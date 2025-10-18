"""GasTwinFormer configuration with 80k iterations training

This is the main configuration file for GasTwinFormer model trained on
gas leak detection dataset.
"""

_base_ = [
    "./_base_/models/gastwinformer.py",
    "./_base_/datasets/example_dataset.py",
    "./_base_/schedules/schedule_80k.py",
    "./_base_/default_runtime.py",
]

crop_size = (512, 512)

# Complete data_preprocessor configuration
data_preprocessor = dict(size=crop_size)

# Pretrained backbone checkpoint (SegFormer MIT-B0)
checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth"

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type="Pretrained", checkpoint=checkpoint)),
    decode_head=dict(num_classes=2),  # Adjust for your task
    cls_head=dict(num_classes=3),  # Adjust for your task or remove if not using classification
)

# Work directory
work_dir = "./work_dirs/gastwinformer_80k"

# Override visualizer name for this specific experiment
visualizer = dict(
    vis_backends=[
        dict(type="LocalVisBackend", save_dir="work_dirs"),
        dict(type="TensorboardVisBackend"),
    ]
)

