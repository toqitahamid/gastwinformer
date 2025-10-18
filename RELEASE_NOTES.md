# GasTwinFormer v1.0.0 - ICCVW 2025 Release

## 🎉 Initial Release

This is the official code release for the paper **"GasTwinFormer: A Hybrid Vision Transformer for Livestock Methane Emission Segmentation and Dietary Classification in Optical Gas Imaging"** accepted at ICCVW 2025.

📄 **arXiv**: [2508.15057](https://arxiv.org/abs/2508.15057)  
🌐 **Project Page**: [gastwinformer.github.io](https://gastwinformer.github.io)

**Authors**: Toqi Tahamid Sarker, Mohamed Embaby, Taminul Islam, Amer AbuGhazaleh, Khaled R Ahmed

**Institutions**: Southern Illinois University Carbondale, University of California Davis

## 📦 Package Contents

### Core Model Components
- **Backbone**: `MixTwinVisionTransformerV3` - Hybrid transformer with GL attention pattern
- **Decode Head**: `AdaptiveLRASPPHead` - Lightweight segmentation head
- **Loss Function**: `GaussianPlumeWeightedDiceLoss` - Physics-informed loss
- **Data Preprocessor**: `MultitaskDataPreProcessor` - For multitask learning
- **Segmentor**: `ConfigurableSegmentor` - Flexible model architecture

### Datasets & Transforms
- `DietPlumeDataset` - Custom dataset loader
- `LoadDietLabel`, `PackMultitaskInputs` - Data transforms
- `MultitaskEvaluator` - Evaluation metrics

### Hooks & Visualization
- `FlexibleSegVisualizationHook` - Advanced visualization with classification overlay

### Configuration Files
- **Model Config**: `gastwinformer/configs/_base_/models/gastwinformer.py`
- **Dataset Config**: `gastwinformer/configs/_base_/datasets/example_dataset.py`
- **Schedule**: `gastwinformer/configs/_base_/schedules/schedule_80k.py`
- **Runtime**: `gastwinformer/configs/_base_/default_runtime.py`
- **Main Config**: `gastwinformer/configs/gastwinformer_80k.py`

### Tools & Scripts
- `tools/train.py` - Training script (single and multi-GPU)
- `tools/test.py` - Testing/evaluation script
- `tools/dist_train.sh` - Distributed training launcher
- `tools/dist_test.sh` - Distributed testing launcher
- `demo/inference_demo.py` - Standalone inference demo

### Documentation
- `README.md` - Comprehensive documentation
- `CONTRIBUTING.md` - Contribution guidelines
- `LICENSE` - MIT License
- `requirements.txt` - Package dependencies
- `setup.py` - Installation script

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/gastwinformer.git
cd gastwinformer
pip install -r requirements.txt
pip install -e .
```

### Training
```bash
# Single GPU
python tools/train.py gastwinformer/configs/gastwinformer_80k.py

# Multi-GPU (4 GPUs)
bash tools/dist_train.sh gastwinformer/configs/gastwinformer_80k.py 4
```

### Inference
```bash
python demo/inference_demo.py \
    --config gastwinformer/configs/gastwinformer_80k.py \
    --checkpoint checkpoints/gastwinformer_80k.pth \
    --image path/to/image.jpg \
    --output demo/results/
```

## 📊 Model Performance

### Benchmark Results on Beef Cattle Methane Dataset (11,694 frames)

| Model | mIoU | mF1 | Diet Acc | Params | FLOPs | FPS |
|-------|------|-----|----------|--------|-------|-----|
| **GasTwinFormer** | **74.47%** | **83.63%** | **100.0%** | **3.348M** | **3.428G** | **114.9** |
| SegFormer-B0 | 72.11% | 81.57% | 100.0% | 3.782M | 7.885G | 119.66 |
| Twins PCPVT-S | 74.05% | 83.25% | 100.0% | 27.906M | 44.34G | 61.60 |
| GasFormer | 72.25% | 81.69% | 100.0% | 3.716M | 9.913G | 102.29 |
| DeepLabV3 | 70.36% | 80.03% | 100.0% | 68.625M | 270.0G | 91.79 |

**Key Achievements**:
- 🏆 State-of-the-art segmentation accuracy (74.47% mIoU)
- ⚡ Most efficient transformer model (3.348M params, 3.428G FLOPs)
- 🚀 Real-time inference (114.9 FPS on NVIDIA A100)
- 🎯 Perfect dietary classification (100% accuracy)

## 🔧 System Requirements

**Tested Configuration**:
- Python 3.8+
- PyTorch 2.0.0+
- CUDA 11.0+
- Intel Xeon Gold 6338 (2.00GHz) CPU
- NVIDIA A100 80GB GPU
- 512GB RAM

**Minimum Requirements**:
- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.0
- 16GB RAM
- GPU with >= 8GB VRAM (for training)

## 📝 Known Issues

None reported yet. Please open an issue on GitHub if you encounter any problems.

## 🛣️ Roadmap

- [ ] Pre-trained model weights release
- [ ] ONNX export support
- [ ] TensorRT optimization
- [ ] Web demo
- [ ] Extended documentation with tutorials

## 🙏 Acknowledgments

Built on top of:
- MMSegmentation
- SegFormer
- Twins-SVT

## 📧 Contact

For questions, please open an issue or contact:
- **Lead Author**: Toqi Tahamid Sarker (toqitahamid.sarker@siu.edu)
- **Project Page**: [gastwinformer.github.io](https://gastwinformer.github.io)
- **GitHub**: [github.com/toqitahamid/gastwinformer](https://github.com/toqitahamid/gastwinformer)

---

**Enjoy using GASTwinFormer! ⭐ Star us on GitHub if you find this useful!**

