# Model settings for GasTwinFormer
# Paper: "GasTwinFormer: A Hybrid Vision Transformer for Livestock Methane 
#         Emission Segmentation and Dietary Classification in Optical Gas Imaging"
# ICCV 2025
#
# Architecture:
# - Mix Twin encoder with EL-EL-EL-EL pattern (Efficient attention + Local attention)
# - 4 stages: 32→64→160→256 channels, H/4→H/32 spatial reduction
# - LSA window size: 5×5
# - Hierarchical LR-ASPP decoder with 128 channels
# - Simple classification head for dietary treatment prediction
#
# Performance:
# - mIoU: 74.47%, mF1: 83.63%, Diet Acc: 100%
# - Params: 3.348M, FLOPs: 3.428G, FPS: 114.9

norm_cfg = dict(type="SyncBN", requires_grad=True)

data_preprocessor = dict(
    type="MultitaskDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)

model = dict(
    type="ConfigurableSegmentor",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="MixTwinVisionTransformerV3",
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        layer_patterns=["GL", "GL", "GL", "GL"],  # Global+Local pattern
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        use_lsa=True,
        window_sizes=[5, 5, 5, 5],
        pretrained=None,
    ),
    decode_head=dict(
        type="AdaptiveLRASPPHead",
        in_channels=(32, 64, 160, 256),
        in_index=(0, 1, 2, 3),
        branch_channels=(32, 64, 160),
        channels=128,
        input_transform="multiple_select",
        dropout_ratio=0.1,
        num_classes=2,  # background and plume
        norm_cfg=norm_cfg,
        act_cfg=dict(type="ReLU"),
        align_corners=False,
        loss_decode=dict(
            type="GaussianPlumeWeightedDiceLoss",
            use_sigmoid=False,
            activate=True,
            reduction="mean",
            loss_weight=1.0,
            eps=1e-6,
            gas_class_idx=1,  # Index of gas/plume class
            loss_name="loss_gaussian_plume_dice",
        ),
    ),
    cls_head=dict(
        type="SimpleClsHead",
        num_classes=3,  # Adjust based on your task
        in_index=-1,
        in_channels=256,
        hidden_channels=256,
        dropout_rate=0.5,
        loss_cfg=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.5),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

