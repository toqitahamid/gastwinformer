# Set default scope to mmseg
default_scope = "mmseg"

# Custom imports for the project
custom_imports = dict(
    imports=[
        "gastwinformer",  # Import the package to trigger registrations
        "mmseg.utils.set_env",
        "mmseg.visualization",
        "mmseg.engine",
    ],
    allow_failed_imports=False,
)

# Runtime configuration
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

# Visualization backends
vis_backends = [
    dict(type="LocalVisBackend", save_dir="work_dirs"),
    dict(type="TensorboardVisBackend"),
]

# Visualizer configuration
visualizer = dict(
    type="SegLocalVisualizer", 
    vis_backends=vis_backends, 
    name="visualizer"
)

# Logging configuration
log_processor = dict(by_epoch=False)
log_level = "INFO"
load_from = None
resume = False

# Random seed
randomness = dict(seed=42)

