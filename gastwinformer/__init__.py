"""GasTwinFormer - Hybrid Vision Transformer for Livestock Methane Emission Segmentation

This package contains the official implementation of GasTwinFormer, a hybrid vision 
transformer for real-time methane emission segmentation and dietary classification 
in optical gas imaging (OGI).

Paper: "GasTwinFormer: A Hybrid Vision Transformer for Livestock Methane Emission 
       Segmentation and Dietary Classification in Optical Gas Imaging"
Conference: ICCV 2025
Authors: Toqi Tahamid Sarker, Mohamed Embaby, Taminul Islam, Amer AbuGhazaleh, 
         Khaled R Ahmed

Key Features:
- Mix Twin encoder with EL-EL-EL-EL attention pattern (EMA + LSA)
- Hierarchical LR-ASPP decoder for multi-scale feature aggregation
- Gaussian Plume-Weighted Dice Loss for physics-informed segmentation
- 74.47% mIoU, 83.63% mF1, 100% dietary classification accuracy
- 3.348M parameters, 3.428G FLOPs, 114.9 FPS on NVIDIA A100

Dataset: 11,694 annotated beef cattle methane emission frames across three dietary treatments
"""

__version__ = '1.0.0'
__author__ = 'Toqi Tahamid Sarker'
__email__ = 'toqitahamid.sarker@siu.edu'

from . import datasets
from . import models
from . import transforms
from . import evaluation
from . import hooks

# Ensure mmseg modules are registered when gastwinformer is imported
try:
    import mmseg.utils.set_env
    mmseg.utils.set_env.register_all_modules(init_default_scope=False)
    print("✓ Successfully registered mmseg modules")
except Exception as e:
    print(f"✗ Failed to register mmseg modules: {e}")
    pass

__all__ = ['datasets', 'models', 'transforms', 'evaluation', 'hooks']

