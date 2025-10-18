# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.utils import is_tuple_of

from mmseg.registry import MODELS
from mmseg.models.utils import resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@MODELS.register_module()
class AdaptiveLRASPPHead(BaseDecodeHead):
    """Adaptive Lite R-ASPP (LRASPP) head that works with any input size.

    This is an adaptation of the original LRASPPHead that uses adaptive pooling
    instead of fixed kernel sizes, making it compatible with different backbone
    architectures and input sizes.

    Args:
        branch_channels (tuple[int]): The number of output channels in every
            each branch. Default: (32, 64).
        pool_output_size (int): Output size for adaptive pooling. Default: 1.
    """

    def __init__(self, branch_channels=(32, 64), pool_output_size=1, **kwargs):
        super().__init__(**kwargs)
        if self.input_transform != "multiple_select":
            raise ValueError(
                "in Adaptive Lite R-ASPP (LRASPP) head, input_transform "
                f"must be 'multiple_select'. But received "
                f"'{self.input_transform}'"
            )
        assert is_tuple_of(branch_channels, int)
        assert len(branch_channels) == len(self.in_channels) - 1
        self.branch_channels = branch_channels
        self.pool_output_size = pool_output_size

        self.convs = nn.Sequential()
        self.conv_ups = nn.Sequential()
        for i in range(len(branch_channels)):
            self.convs.add_module(
                f"conv{i}",
                nn.Conv2d(self.in_channels[i], branch_channels[i], 1, bias=False),
            )
            self.conv_ups.add_module(
                f"conv_up{i}",
                ConvModule(
                    self.channels + branch_channels[i],
                    self.channels,
                    1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=False,
                ),
            )

        self.conv_up_input = nn.Conv2d(self.channels, self.channels, 1)

        self.aspp_conv = ConvModule(
            self.in_channels[-1],
            self.channels,
            1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            bias=False,
        )

        # Use adaptive pooling instead of fixed kernel size
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pool_output_size),
            ConvModule(
                self.in_channels[-1],  # Use last channel instead of hardcoded index
                self.channels,
                1,
                act_cfg=dict(type="Sigmoid"),
                bias=False,
            ),
        )

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        x = inputs[-1]

        x = self.aspp_conv(x) * resize(
            self.image_pool(x),
            size=x.size()[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        x = self.conv_up_input(x)

        for i in range(len(self.branch_channels) - 1, -1, -1):
            x = resize(
                x,
                size=inputs[i].size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            x = torch.cat([x, self.convs[i](inputs[i])], 1)
            x = self.conv_ups[i](x)

        return self.cls_seg(x)
