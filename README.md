# GasTwinFormer: Gaussian Plume-Weighted Twin Transformer for Gas Leak Segmentation

[![ICCV 2025](https://img.shields.io/badge/ICCV-2025-blue)](https://iccv2025.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official PyTorch implementation** of GASTwinFormer accepted at ICCV 2025.

[Paper]() | [Project Page]() | [Demo]()

## 📋 Overview

GasTwinFormer is a hybrid vision transformer for real-time livestock methane emission segmentation and dietary classification in optical gas imaging (OGI). The model combines:

- **Hybrid Twin Transformer Architecture**: Mix Twin encoder alternating between Efficient Multi-head Attention (EMA) and Locally-grouped Self-Attention (LSA) with EL-EL-EL-EL pattern
- **Gaussian Plume-Weighted Loss**: Physics-informed loss function leveraging gas dispersion behavior
- **Hierarchical LR-ASPP Decoder**: Lightweight decoder for multi-scale feature aggregation
- **Multitask Learning**: Simultaneous methane segmentation and dietary treatment classification

### Key Features

✨ **State-of-the-art Performance**: **74.47% mIoU** and **83.63% mF1** for segmentation  
⚡ **Real-time Inference**: **114.9 FPS** on NVIDIA A100 GPU  
🔬 **Physics-Informed**: Incorporates Gaussian plume model through weighted Dice loss  
🎯 **Efficient Architecture**: Only **3.348M parameters** with **3.428 GFLOPs**  
🎓 **Perfect Classification**: **100% accuracy** for dietary treatment prediction  
📦 **Easy to Use**: Built on MMSegmentation for seamless integration  

## 🚀 Getting Started

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.0 (for GPU training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gastwinformer.git
cd gastwinformer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install GASTwinFormer**
```bash
pip install -e .
```

### Quick Start

#### Inference with Pre-trained Model

```python
import torch
from mmseg.apis import init_model, inference_model, show_result_pyplot

# Load config and checkpoint
config_file = 'gastwinformer/configs/gastwinformer_80k.py'
checkpoint_file = 'path/to/checkpoint.pth'  # Download from releases

# Initialize model
model = init_model(config_file, checkpoint_file, device='cuda:0')

# Run inference
img = 'path/to/your/image.jpg'
result = inference_model(model, img)

# Visualize results
show_result_pyplot(model, img, result, show=True)
```

#### Training

```bash
# Single GPU
python tools/train.py gastwinformer/configs/gastwinformer_80k.py

# Multi-GPU (4 GPUs)
bash tools/dist_train.sh gastwinformer/configs/gastwinformer_80k.py 4
```

#### Evaluation

```bash
python tools/test.py gastwinformer/configs/gastwinformer_80k.py \
    path/to/checkpoint.pth \
    --eval mIoU
```

## 📊 Model Architecture

### Backbone: Mix Twin Encoder

The backbone combines Efficient Multi-head Attention (EMA) from SegFormer and Locally-grouped Self-Attention (LSA) from Twins in a novel hybrid architecture:

- **4 hierarchical stages** with progressive downsampling (H/4 → H/32) and channel expansion (32 → 64 → 160 → 256)
- **EL-EL-EL-EL pattern**: Each stage contains one EMA block followed by one LSA block (8 total blocks)
- **Locally-grouped Self Attention (LSA)**: 5×5 window size for fine-grained local structure
- **Efficient Multi-head Attention (EMA)**: Spatial reduction ratios [8, 4, 2, 1] for global context
- **Mix Feed-Forward Network**: Incorporates 3×3 depthwise convolution for spatial inductive bias

### Decoder: Hierarchical LR-ASPP

Lightweight decoder for multi-scale feature aggregation:

- Processes multi-scale features {F₁, F₂, F₃, F₄} from all encoder stages
- Adaptive average pooling for resolution independence
- Progressive fusion: F₄ through ASPP path, F₁-F₃ through 1×1 conv branches
- 128 internal channels for optimal accuracy-efficiency balance

### Classification Head

Simple yet effective scene-level dietary classification:

- Operates on Stage 4 features (highest semantic level)
- Two-layer fully connected network with ReLU and dropout
- Predicts dietary treatment: High Forage (HF), Mixed Diet (MD), High Grain (HG)

### Loss Function: Gaussian Plume-Weighted Dice Loss

Physics-informed loss incorporating gas dispersion behavior:

```
L_weighted = 1 - (2∑(w(p)·yₚ·ŷₚ) + ε) / (∑(w(p)·yₚ) + ∑(w(p)·ŷₚ) + ε)

where w(p) = exp(-((pₓ-μₓ)²/(2σₓ²) + (pᵧ-μᵧ)²/(2σᵧ²)))
```

The Gaussian weights are computed from predicted mask center-of-mass (μₓ, μᵧ) and weighted standard deviations (σₓ, σᵧ), emphasizing plume center regions while attenuating towards edges.

## 📁 Dataset Preparation

GasTwinFormer expects data in the following structure:

```
data/
├── train/
│   ├── images/
│   │   ├── FLIR0001_frame_00001.png
│   │   └── ...
│   └── masks/
│       ├── FLIR0001_frame_00001.png
│       └── ...
├── val/
│   ├── images/
│   └── masks/
├── test/
│   ├── images/
│   └── masks/
├── combined_train.csv
├── combined_val.csv
└── combined_test.csv
```

**CSV Format** (for multitask learning with dietary classification):
```csv
img_name,diet_type
FLIR0001_frame_00001.png,high_forage
FLIR0002_frame_00001.png,control
FLIR0003_frame_00001.png,low_forage
...
```

**Dataset Statistics**:
- **Total Frames**: 11,694 annotated frames
- **Dietary Treatments**: 3 (High Forage, Mixed Diet/Control, High Grain/Low Forage)
- **Source**: Beef cattle methane emissions captured via Optical Gas Imaging (OGI)
- **Technology**: FLIR thermal infrared cameras (7-8.5 μm spectral range)

## 🔧 Configuration

The model is highly configurable through MMSegmentation's config system. Key configuration files:

- `gastwinformer/configs/gastwinformer_80k.py`: Main config
- `gastwinformer/configs/_base_/models/gastwinformer.py`: Model architecture
- `gastwinformer/configs/_base_/datasets/example_dataset.py`: Dataset settings
- `gastwinformer/configs/_base_/schedules/schedule_80k.py`: Training schedule

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 6×10⁻⁵ | Base learning rate (AdamW) |
| Beta | (0.9, 0.999) | Adam beta parameters |
| Weight Decay | 0.01 | L2 regularization |
| Warmup | 1,500 iterations | Linear warmup from 10⁻⁶ |
| LR Schedule | Polynomial (power=1.0) | Learning rate decay |
| Total Iterations | 80,000 | Training iterations |
| Validation Interval | 8,000 | Validation frequency |
| Batch Size | 8 | Training batch size |
| Input Size | 512×512 | Image resolution |
| Drop Path Rate | 0.1 | Stochastic depth |
| Decoder LR Multiplier | 10× | For faster head convergence |

## 📈 Results

### Methane Emission Segmentation

| Model | Backbone | mIoU (%) | mF1 (%) | Diet Acc (%) | Params (M) | FLOPs (G) | FPS |
|-------|----------|----------|---------|--------------|------------|-----------|-----|
| SegFormer | MiT-B0 | 72.11 | 81.57 | 100.0 | 3.782 | 7.885 | 119.66 |
| Twins | PCPVT-S | 74.05 | 83.25 | 100.0 | 27.906 | 44.34 | 61.60 |
| Twins | SVT-S | 72.06 | 81.62 | 100.0 | 27.846 | 38.471 | 51.64 |
| GasFormer | MiT-B0 | 72.25 | 81.69 | 100.0 | 3.716 | 9.913 | 102.29 |
| DeepLabV3 | ResNet-50 | 70.36 | 80.03 | 100.0 | 68.625 | 270.0 | 91.79 |
| DDRNet | DDRNet | 68.91 | 78.65 | 99.94 | 5.766 | 4.56 | 156.38 |
| **GasTwinFormer** | **MixTwin** | **74.47** | **83.63** | **100.0** | **3.348** | **3.428** | **114.9** |

**Key Achievements**:
- 🏆 **Highest Accuracy**: 74.47% mIoU, 83.63% mF1 (best among all methods)
- ⚡ **Most Efficient**: 3.348M parameters, 3.428G FLOPs (smallest transformer model)
- 🚀 **Real-time**: 114.9 FPS enables continuous monitoring
- 🎯 **Perfect Classification**: 100% dietary treatment accuracy

### Performance Comparison

**vs. SegFormer-B0**: +2.36% mIoU, -11.5% parameters, -56.5% FLOPs  
**vs. GasFormer**: +2.22% mIoU, -9.9% parameters, -65.4% FLOPs  
**vs. Twins PCPVT-S**: +0.42% mIoU, -8.3× parameters, -12.9× FLOPs  
**vs. DeepLabV3**: +4.11% mIoU, -20× parameters, -78× FLOPs

## 🛠️ Advanced Usage

### Custom Dataset

```python
from gastwinformer.datasets import DietPlumeDataset
from mmseg.registry import DATASETS

@DATASETS.register_module()
class MyCustomDataset(DietPlumeDataset):
    METAINFO = dict(
        classes=('background', 'gas'),
        palette=[[0, 0, 0], [255, 0, 0]]
    )
    
    # Override methods as needed
```

### Custom Loss Function

```python
from gastwinformer.models.losses import GaussianPlumeWeightedDiceLoss

# Use in config
loss_decode = dict(
    type='GaussianPlumeWeightedDiceLoss',
    loss_weight=1.0,
    gas_class_idx=1
)
```

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{sarker2025gastwinformer,
  title={GasTwinFormer: A Hybrid Vision Transformer for Livestock Methane Emission Segmentation and Dietary Classification in Optical Gas Imaging},
  author={Sarker, Toqi Tahamid and Embaby, Mohamed and Islam, Taminul and AbuGhazaleh, Amer and Ahmed, Khaled R},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

## 🙏 Acknowledgements

This project is built on top of several excellent open-source projects:

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): Semantic segmentation toolbox
- [SegFormer](https://github.com/NVlabs/SegFormer): Transformer baseline
- [Twins-SVT](https://github.com/Meituan-AutoML/Twins): Twin attention mechanism

We thank the authors for their contributions to the community.

## 📄 License

This project is released under the [MIT License](LICENSE).

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📧 Contact

For questions and feedback:
- **Issues**: [GitHub Issues](https://github.com/yourusername/gastwinformer/issues)
- **Email**: toqitahamid.sarker@siu.edu
- **Project Page**: [gastwinformer.github.io](https://gastwinformer.github.io)

---

**Note**: This is research code. It is provided as-is for reproducibility and further research. For production use, additional testing and validation are recommended.

