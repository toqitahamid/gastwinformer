# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import constant_init, normal_init, trunc_normal_init
from mmcv.cnn.bricks.transformer import FFN
from mmseg.registry import MODELS
from mmseg.models.utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from mmengine.runner import CheckpointLoader


class MixFFN(BaseModule):
    """An implementation of MixFFN of Segformer.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Conv to encode positional information.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims,
        feedforward_channels,
        act_cfg=dict(type="GELU"),
        ffn_drop=0.0,
        dropout_layer=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels,
        )
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
        )

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(MultiheadAttention):
    """An implementation of Efficient Multi-head Attention of Segformer.

    This module is modified from MultiheadAttention which is a module from
    mmcv.cnn.bricks.transformer.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        dropout_layer=None,
        init_cfg=None,
        batch_first=True,
        qkv_bias=False,
        norm_cfg=dict(type="LN"),
        sr_ratio=1,
    ):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias,
        )

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio,
            )
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        # handle the BC-breaking from https://github.com/open-mmlab/mmcv/pull/1418 # noqa
        from mmseg import digit_version, mmcv_version

        if mmcv_version < digit_version("1.3.17"):
            warnings.warn(
                "The legacy version of forward function in"
                "EfficientMultiheadAttention is deprecated in"
                "mmcv>=1.3.17 and will no longer support in the"
                "future. Please upgrade your mmcv."
            )
            self.forward = self.legacy_forward

    def forward(self, x, hw_shape, identity=None):
        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            x_q = x_q.transpose(0, 1)
            x_kv = x_kv.transpose(0, 1)

        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))

    def legacy_forward(self, x, hw_shape, identity=None):
        """multi head attention forward in mmcv version < 1.3.17."""

        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        # `need_weights=True` will let nn.MultiHeadAttention
        # `return attn_output, attn_output_weights.sum(dim=1) / num_heads`
        # The `attn_output_weights.sum(dim=1)` may cause cuda error. So, we set
        # `need_weights=False` to ignore `attn_output_weights.sum(dim=1)`.
        # This issue - `https://github.com/pytorch/pytorch/issues/37583` report
        # the error that large scale tensor sum operation may cause cuda error.
        out = self.attn(query=x_q, key=x_kv, value=x_kv, need_weights=False)[0]

        return identity + self.dropout_layer(self.proj_drop(out))


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Segformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default:None.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        qkv_bias=True,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN"),
        batch_first=True,
        sr_ratio=1,
        with_cp=False,
    ):
        super().__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type="DropPath", drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio,
        )

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type="DropPath", drop_prob=drop_path_rate),
            act_cfg=act_cfg,
        )

        self.with_cp = with_cp

    def forward(self, x, hw_shape):
        def _inner_forward(x):
            x = self.attn(self.norm1(x), hw_shape, identity=x)
            x = self.ffn(self.norm2(x), hw_shape, identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class LocallyGroupedSelfAttention(BaseModule):
    """Locally-grouped Self Attention (LSA) module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        window_size(int): Window size of LSA. Default: 1.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(
        self,
        embed_dims,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        window_size=1,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        assert embed_dims % num_heads == 0, (
            f"dim {embed_dims} should be divided by num_heads {num_heads}."
        )
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.window_size = window_size

    def forward(self, x, hw_shape):
        b, n, c = x.shape
        h, w = hw_shape
        x = x.view(b, h, w, c)

        # pad feature maps to multiples of Local-groups
        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        # calculate attention mask for LSA
        Hp, Wp = x.shape[1:-1]
        _h, _w = Hp // self.window_size, Wp // self.window_size
        mask = torch.zeros((1, Hp, Wp), device=x.device)
        mask[:, -pad_b:, :].fill_(1)
        mask[:, :, -pad_r:].fill_(1)

        # [B, _h, _w, window_size, window_size, C]
        x = x.reshape(b, _h, self.window_size, _w, self.window_size, c).transpose(2, 3)
        mask = (
            mask.reshape(1, _h, self.window_size, _w, self.window_size)
            .transpose(2, 3)
            .reshape(1, _h * _w, self.window_size * self.window_size)
        )
        # [1, _h*_w, window_size*window_size, window_size*window_size]
        attn_mask = mask.unsqueeze(2) - mask.unsqueeze(3)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-1000.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        # [3, B, _w*_h, nhead, window_size*window_size, dim]
        qkv = (
            self.qkv(x)
            .reshape(
                b,
                _h * _w,
                self.window_size * self.window_size,
                3,
                self.num_heads,
                c // self.num_heads,
            )
            .permute(3, 0, 1, 4, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        # [B, _h*_w, n_head, window_size*window_size, window_size*window_size]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + attn_mask.unsqueeze(2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (
            (attn @ v)
            .transpose(2, 3)
            .reshape(b, _h, _w, self.window_size, self.window_size, c)
        )
        x = attn.transpose(2, 3).reshape(
            b, _h * self.window_size, _w * self.window_size, c
        )
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x.reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LSAEncoderLayer(BaseModule):
    """Implements one encoder layer in Twins-SVT.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
           Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): Enable bias for qkv if True. Default: True
        qk_scale (float | None, optional): Override default qk scale of
           head_dim ** -0.5 if set. Default: None.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        window_size (int): Window size of LSA. Default: 1.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        num_fcs=2,
        qkv_bias=True,
        qk_scale=None,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN"),
        window_size=1,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)[1]
        self.attn = LocallyGroupedSelfAttention(
            embed_dims,
            num_heads,
            qkv_bias,
            qk_scale,
            attn_drop_rate,
            drop_rate,
            window_size,
        )

        self.norm2 = build_norm_layer(norm_cfg, embed_dims, postfix=2)[1]
        # self.ffn = FFN(
        #     embed_dims=embed_dims,
        #     feedforward_channels=feedforward_channels,
        #     num_fcs=num_fcs,
        #     ffn_drop=drop_rate,
        #     dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
        #     act_cfg=act_cfg,
        #     add_identity=False)

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type="DropPath", drop_prob=drop_path_rate),
            act_cfg=act_cfg,
        )

        self.drop_path = (
            build_dropout(dict(type="DropPath", drop_prob=drop_path_rate))
            if drop_path_rate > 0.0
            else nn.Identity()
        )

    def forward(self, x, hw_shape):
        x = x + self.drop_path(self.attn(self.norm1(x), hw_shape))
        x = self.drop_path(self.ffn(self.norm2(x), hw_shape, identity=x))
        # x = x + self.drop_path(self.ffn(self.norm2(x), hw_shape))
        return x


@MODELS.register_module()
class MixTwinVisionTransformerV3(BaseModule):
    """The backbone of Segformer.

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        window_sizes (Sequence[int]): Window size of LSA for each stage.
            Default: [7, 7, 7, 7].
        use_lsa (bool): Whether to use LSA (Locally-grouped Self Attention)
            in alternating layers. This is a legacy parameter, use layer_patterns
            for more flexibility. Default: False.
        layer_patterns (Sequence[str | List] | None, optional): Flexible layer
            patterns for each stage. Can be strings like "LLG" or lists like
            ["L", "L", "G"] where L=LSA, G=TransformerEncoder. Each pattern
            length must match corresponding num_layers. If None, falls back
            to use_lsa behavior. Default: None.
    """

    def __init__(
        self,
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 4, 8],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN", eps=1e-6),
        pretrained=None,
        init_cfg=None,
        with_cp=False,
        window_sizes=[7, 7, 7, 7],
        use_lsa=False,
        layer_patterns=None,
    ):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), (
            "init_cfg and pretrained cannot be set at the same time"
        )
        if isinstance(pretrained, str):
            warnings.warn(
                "DeprecationWarning: pretrained is deprecated, "
                'please use "init_cfg" instead'
            )
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError("pretrained must be a str or None")

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        self.window_sizes = window_sizes
        self.use_lsa = use_lsa
        self.layer_patterns = layer_patterns

        assert (
            num_stages
            == len(num_layers)
            == len(num_heads)
            == len(patch_sizes)
            == len(strides)
            == len(sr_ratios)
        )
        # Process and validate layer patterns first
        self.processed_patterns = self._process_layer_patterns()

        # Check if LSA layers are used in any pattern
        self.has_lsa = any("L" in pattern for pattern in self.processed_patterns)
        if self.has_lsa:
            assert len(window_sizes) == num_stages, (
                "window_sizes length must equal num_stages when LSA layers are used"
            )

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # patch_embed
        self.patch_embed1 = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims * num_heads[0],
            kernel_size=patch_sizes[0],
            stride=strides[0],
            padding=patch_sizes[0] // 2,
            norm_cfg=norm_cfg,
        )
        self.patch_embed2 = PatchEmbed(
            in_channels=embed_dims * num_heads[0],
            embed_dims=embed_dims * num_heads[1],
            kernel_size=patch_sizes[1],
            stride=strides[1],
            padding=patch_sizes[1] // 2,
            norm_cfg=norm_cfg,
        )
        self.patch_embed3 = PatchEmbed(
            in_channels=embed_dims * num_heads[1],
            embed_dims=embed_dims * num_heads[2],
            kernel_size=patch_sizes[2],
            stride=strides[2],
            padding=patch_sizes[2] // 2,
            norm_cfg=norm_cfg,
        )
        self.patch_embed4 = PatchEmbed(
            in_channels=embed_dims * num_heads[2],
            embed_dims=embed_dims * num_heads[3],
            kernel_size=patch_sizes[3],
            stride=strides[3],
            padding=patch_sizes[3] // 2,
            norm_cfg=norm_cfg,
        )

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic depth decay rule
        cur = 0

        # Stage 1 blocks
        block1_layers = []
        stage_pattern = self.processed_patterns[0]
        for i in range(num_layers[0]):
            layer_type = stage_pattern[i]
            layer = self._create_layer(
                layer_type=layer_type,
                stage_idx=0,
                embed_dims=embed_dims * num_heads[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[cur + i],
                qkv_bias=qkv_bias,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                sr_ratio=sr_ratios[0],
                window_size=window_sizes[0],
            )
            block1_layers.append(layer)
        self.block1 = ModuleList(block1_layers)
        self.norm1 = build_norm_layer(norm_cfg, embed_dims * num_heads[0])[1]

        cur += num_layers[0]
        # Stage 2 blocks
        block2_layers = []
        stage_pattern = self.processed_patterns[1]
        for i in range(num_layers[1]):
            layer_type = stage_pattern[i]
            layer = self._create_layer(
                layer_type=layer_type,
                stage_idx=1,
                embed_dims=embed_dims * num_heads[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[cur + i],
                qkv_bias=qkv_bias,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                sr_ratio=sr_ratios[1],
                window_size=window_sizes[1],
            )
            block2_layers.append(layer)
        self.block2 = ModuleList(block2_layers)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims * num_heads[1])[1]

        cur += num_layers[1]
        # Stage 3 blocks
        block3_layers = []
        stage_pattern = self.processed_patterns[2]
        for i in range(num_layers[2]):
            layer_type = stage_pattern[i]
            layer = self._create_layer(
                layer_type=layer_type,
                stage_idx=2,
                embed_dims=embed_dims * num_heads[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[cur + i],
                qkv_bias=qkv_bias,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                sr_ratio=sr_ratios[2],
                window_size=window_sizes[2],
            )
            block3_layers.append(layer)
        self.block3 = ModuleList(block3_layers)
        self.norm3 = build_norm_layer(norm_cfg, embed_dims * num_heads[2])[1]

        cur += num_layers[2]
        # Stage 4 blocks
        block4_layers = []
        stage_pattern = self.processed_patterns[3]
        for i in range(num_layers[3]):
            layer_type = stage_pattern[i]
            layer = self._create_layer(
                layer_type=layer_type,
                stage_idx=3,
                embed_dims=embed_dims * num_heads[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[cur + i],
                qkv_bias=qkv_bias,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                sr_ratio=sr_ratios[3],
                window_size=window_sizes[3],
            )
            block4_layers.append(layer)
        self.block4 = ModuleList(block4_layers)
        self.norm4 = build_norm_layer(norm_cfg, embed_dims * num_heads[3])[1]

    def _process_layer_patterns(self):
        """Process and validate layer patterns for each stage.

        Returns:
            List[List[str]]: Processed patterns where each stage has a list of
                layer types ('L' for LSA, 'G' for TransformerEncoder).
        """
        if self.layer_patterns is None:
            # Fallback to legacy use_lsa behavior
            if self.use_lsa:
                # Alternating pattern: LSA for even indices, TransformerEncoder for odd
                patterns = []
                for stage_layers in self.num_layers:
                    stage_pattern = []
                    for i in range(stage_layers):
                        if i % 2 == 0:
                            stage_pattern.append("L")  # LSA
                        else:
                            stage_pattern.append("G")  # TransformerEncoder
                    patterns.append(stage_pattern)
                return patterns
            else:
                # All TransformerEncoder layers
                return [["G"] * stage_layers for stage_layers in self.num_layers]

        # Process custom layer patterns
        if len(self.layer_patterns) != self.num_stages:
            raise ValueError(
                f"layer_patterns length ({len(self.layer_patterns)}) must equal "
                f"num_stages ({self.num_stages})"
            )

        processed = []
        for stage_idx, pattern in enumerate(self.layer_patterns):
            expected_length = self.num_layers[stage_idx]

            if isinstance(pattern, str):
                # String pattern like "LLG", "GLGL"
                if len(pattern) != expected_length:
                    raise ValueError(
                        f"Stage {stage_idx} pattern length ({len(pattern)}) must equal "
                        f"num_layers[{stage_idx}] ({expected_length})"
                    )

                # Validate characters
                for char in pattern:
                    if char not in ["L", "G"]:
                        raise ValueError(
                            f"Invalid character '{char}' in pattern. Use 'L' for LSA "
                            "and 'G' for TransformerEncoder"
                        )

                processed.append(list(pattern))

            elif isinstance(pattern, (list, tuple)):
                # List pattern like ["L", "L", "G"], [True, False, True]
                if len(pattern) != expected_length:
                    raise ValueError(
                        f"Stage {stage_idx} pattern length ({len(pattern)}) must equal "
                        f"num_layers[{stage_idx}] ({expected_length})"
                    )

                stage_pattern = []
                for item in pattern:
                    if isinstance(item, bool):
                        stage_pattern.append("L" if item else "G")
                    elif isinstance(item, str):
                        if item not in ["L", "G"]:
                            raise ValueError(
                                f"Invalid string '{item}' in pattern. Use 'L' for LSA "
                                "and 'G' for TransformerEncoder"
                            )
                        stage_pattern.append(item)
                    else:
                        raise ValueError(
                            f"Invalid pattern item type: {type(item)}. "
                            "Use strings ('L'/'G') or booleans (True/False)"
                        )

                processed.append(stage_pattern)
            else:
                raise ValueError(
                    f"Invalid pattern type for stage {stage_idx}: {type(pattern)}. "
                    "Use string or list/tuple"
                )

        return processed

    def _create_layer(
        self,
        layer_type,
        stage_idx,
        embed_dims,
        num_heads,
        mlp_ratio,
        drop_rate,
        attn_drop_rate,
        drop_path_rate,
        qkv_bias,
        act_cfg,
        norm_cfg,
        with_cp,
        sr_ratio,
        window_size,
    ):
        """Create a single layer based on type specification.

        Args:
            layer_type (str): 'L' for LSAEncoderLayer, 'G' for TransformerEncoderLayer
            stage_idx (int): Stage index for dimension calculation

        Returns:
            nn.Module: The created layer
        """
        if layer_type == "L":
            return LSAEncoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=mlp_ratio * embed_dims,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                qkv_bias=qkv_bias,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                window_size=window_size,
            )
        elif layer_type == "G":
            return TransformerEncoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=mlp_ratio * embed_dims,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                qkv_bias=qkv_bias,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                sr_ratio=sr_ratio,
            )
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    def _create_segformer_key_mapping(self):
        """Create mapping from SegFormer checkpoint keys to our model keys."""
        mapping = {}

        # Map patch embeddings
        for i in range(4):
            stage_idx = i
            mapping[f"layers.{stage_idx}.0.projection.weight"] = (
                f"patch_embed{i + 1}.projection.weight"
            )
            mapping[f"layers.{stage_idx}.0.projection.bias"] = (
                f"patch_embed{i + 1}.projection.bias"
            )
            mapping[f"layers.{stage_idx}.0.norm.weight"] = (
                f"patch_embed{i + 1}.norm.weight"
            )
            mapping[f"layers.{stage_idx}.0.norm.bias"] = f"patch_embed{i + 1}.norm.bias"

        # Map transformer blocks (only non-LSA blocks)
        for stage_idx in range(4):
            stage_num_layers = self.num_layers[stage_idx]
            block_attr = getattr(self, f"block{stage_idx + 1}")

            checkpoint_block_idx = 0  # Track checkpoint block index
            for layer_idx in range(stage_num_layers):
                layer = block_attr[layer_idx]

                # Only map TransformerEncoderLayer (not LSA layers)
                if isinstance(layer, TransformerEncoderLayer):
                    # Map transformer encoder layer weights
                    base_key = f"layers.{stage_idx}.1.{checkpoint_block_idx}"
                    target_key = f"block{stage_idx + 1}.{layer_idx}"

                    # Attention weights
                    mapping[f"{base_key}.norm1.weight"] = f"{target_key}.norm1.weight"
                    mapping[f"{base_key}.norm1.bias"] = f"{target_key}.norm1.bias"
                    mapping[f"{base_key}.attn.attn.in_proj_weight"] = (
                        f"{target_key}.attn.attn.in_proj_weight"
                    )
                    mapping[f"{base_key}.attn.attn.in_proj_bias"] = (
                        f"{target_key}.attn.attn.in_proj_bias"
                    )
                    mapping[f"{base_key}.attn.attn.out_proj.weight"] = (
                        f"{target_key}.attn.attn.out_proj.weight"
                    )
                    mapping[f"{base_key}.attn.attn.out_proj.bias"] = (
                        f"{target_key}.attn.attn.out_proj.bias"
                    )

                    # SR (spatial reduction) weights if sr_ratio > 1
                    if hasattr(layer.attn, "sr"):
                        mapping[f"{base_key}.attn.sr.weight"] = (
                            f"{target_key}.attn.sr.weight"
                        )
                        mapping[f"{base_key}.attn.sr.bias"] = (
                            f"{target_key}.attn.sr.bias"
                        )
                        mapping[f"{base_key}.attn.norm.weight"] = (
                            f"{target_key}.attn.norm.weight"
                        )
                        mapping[f"{base_key}.attn.norm.bias"] = (
                            f"{target_key}.attn.norm.bias"
                        )

                    # FFN weights
                    mapping[f"{base_key}.norm2.weight"] = f"{target_key}.norm2.weight"
                    mapping[f"{base_key}.norm2.bias"] = f"{target_key}.norm2.bias"
                    mapping[f"{base_key}.ffn.layers.0.weight"] = (
                        f"{target_key}.ffn.layers.0.weight"
                    )
                    mapping[f"{base_key}.ffn.layers.0.bias"] = (
                        f"{target_key}.ffn.layers.0.bias"
                    )
                    mapping[f"{base_key}.ffn.layers.1.weight"] = (
                        f"{target_key}.ffn.layers.1.weight"
                    )
                    mapping[f"{base_key}.ffn.layers.1.bias"] = (
                        f"{target_key}.ffn.layers.1.bias"
                    )
                    mapping[f"{base_key}.ffn.layers.4.weight"] = (
                        f"{target_key}.ffn.layers.4.weight"
                    )
                    mapping[f"{base_key}.ffn.layers.4.bias"] = (
                        f"{target_key}.ffn.layers.4.bias"
                    )

                    checkpoint_block_idx += 1
                # Skip LSA layers - they will remain randomly initialized

            # Map stage normalization
            mapping[f"layers.{stage_idx}.2.weight"] = f"norm{stage_idx + 1}.weight"
            mapping[f"layers.{stage_idx}.2.bias"] = f"norm{stage_idx + 1}.bias"

        return mapping

    def load_segformer_weights(self, checkpoint_path):
        """Load SegFormer weights into compatible layers of the hybrid model."""
        print(f"Loading SegFormer weights from: {checkpoint_path}")

        # Load checkpoint
        checkpoint = CheckpointLoader.load_checkpoint(
            checkpoint_path, map_location="cpu"
        )
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Create key mapping
        key_mapping = self._create_segformer_key_mapping()

        # Load compatible weights
        model_dict = self.state_dict()
        loaded_keys = []
        skipped_keys = []

        for checkpoint_key, model_key in key_mapping.items():
            if checkpoint_key in state_dict and model_key in model_dict:
                checkpoint_weight = state_dict[checkpoint_key]
                model_weight = model_dict[model_key]

                # Check if shapes match
                if checkpoint_weight.shape == model_weight.shape:
                    model_dict[model_key] = checkpoint_weight
                    loaded_keys.append(f"{checkpoint_key} -> {model_key}")
                else:
                    skipped_keys.append(
                        f"{checkpoint_key} -> {model_key} (shape mismatch: {checkpoint_weight.shape} vs {model_weight.shape})"
                    )
            else:
                if checkpoint_key not in state_dict:
                    skipped_keys.append(f"{checkpoint_key} (not in checkpoint)")
                if model_key not in model_dict:
                    skipped_keys.append(f"{model_key} (not in model)")

        # Load the updated state dict
        self.load_state_dict(model_dict, strict=False)

        print(f"Successfully loaded {len(loaded_keys)} weight mappings")
        print(f"Skipped {len(skipped_keys)} mappings due to incompatibility")

        if loaded_keys:
            print("\nLoaded weights:")
            for key in loaded_keys[:10]:  # Show first 10
                print(f"  ✓ {key}")
            if len(loaded_keys) > 10:
                print(f"  ... and {len(loaded_keys) - 10} more")

        if skipped_keys:
            print(f"\nSkipped weights:")
            for key in skipped_keys[:5]:  # Show first 5
                print(f"  ✗ {key}")
            if len(skipped_keys) > 5:
                print(f"  ... and {len(skipped_keys) - 5} more")

        # Initialize LSA layers properly
        self._init_lsa_layers()

        print("SegFormer weight loading completed!")

    def _init_lsa_layers(self):
        """Initialize LSA layers with proper weights."""
        print("Initializing LSA layers...")

        for stage_idx in range(4):
            block_attr = getattr(self, f"block{stage_idx + 1}")
            for layer_idx, layer in enumerate(block_attr):
                if isinstance(layer, LSAEncoderLayer):
                    # Initialize LSA layer weights
                    if hasattr(layer.attn, "qkv"):
                        trunc_normal_init(layer.attn.qkv, std=0.02, bias=0.0)
                    if hasattr(layer.attn, "proj"):
                        trunc_normal_init(layer.attn.proj, std=0.02, bias=0.0)
                    print(
                        f"  ✓ Initialized LSA layer: block{stage_idx + 1}.{layer_idx}"
                    )

    def init_weights(self):
        """Initialize weights with special handling for SegFormer checkpoints."""
        # Handle custom SegFormer weight loading
        if (
            self.init_cfg is not None
            and self.init_cfg.get("type") == "Pretrained"
            and "mit_b" in self.init_cfg.get("checkpoint", "")
        ):
            # This is a SegFormer checkpoint - use custom loading
            print("Detected SegFormer checkpoint - using custom weight loading")
            self.load_segformer_weights(self.init_cfg["checkpoint"])
        elif self.init_cfg is None:
            # Default initialization when no checkpoint specified
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=0.02, bias=0.0)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
            # Initialize LSA layers even when no checkpoint
            self._init_lsa_layers()
        else:
            # Standard initialization for other checkpoint types
            super().init_weights()

    def forward(self, x):
        outs = []

        # stage 1
        x, hw_shape = self.patch_embed1(x)
        for block in self.block1:
            x = block(x, hw_shape)
        x = self.norm1(x)
        x = nlc_to_nchw(x, hw_shape)
        if 0 in self.out_indices:
            outs.append(x)

        # stage 2
        x, hw_shape = self.patch_embed2(x)
        for block in self.block2:
            x = block(x, hw_shape)
        x = self.norm2(x)
        x = nlc_to_nchw(x, hw_shape)
        if 1 in self.out_indices:
            outs.append(x)

        # stage 3
        x, hw_shape = self.patch_embed3(x)
        for block in self.block3:
            x = block(x, hw_shape)
        x = self.norm3(x)
        x = nlc_to_nchw(x, hw_shape)
        if 2 in self.out_indices:
            outs.append(x)

        # stage 4
        x, hw_shape = self.patch_embed4(x)
        for block in self.block4:
            x = block(x, hw_shape)
        x = self.norm4(x)
        x = nlc_to_nchw(x, hw_shape)
        if 3 in self.out_indices:
            outs.append(x)

        return outs
