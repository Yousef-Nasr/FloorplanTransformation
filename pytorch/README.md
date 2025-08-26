# PyTorch Floorplan Transformation (Modernized)

This is a modernized PyTorch implementation of the floorplan transformation algorithm from the paper "Raster-to-Vector: Revisiting Floorplan Transformation" (ICCV 2017).

## 🚀 Quick Start

### Requirements
- Python 3.8+ (tested with Python 3.12)
- PyTorch 2.0+
- CUDA (optional, CPU inference supported)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Yousef-Nasr/FloorplanTransformation.git
cd FloorplanTransformation/pytorch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the setup demo (optional):
```bash
./setup_demo.sh
```

## 🎯 Usage

### Streamlit Web App

Launch the interactive web interface:
```bash
streamlit run streamlit_app.py
```

This will open a web interface where you can:
- Upload floorplan images
- Load trained model checkpoints
- Run inference with real-time visualization
- Download results and vector representations

### Command Line Inference

For batch processing or scripting:
```bash
python inference.py \
    --image path/to/floorplan.png \
    --checkpoint path/to/model.pth \
    --output results/ \
    --visualize \
    --reconstruct
```

**Arguments:**
- `--image`: Path to input floorplan image
- `--checkpoint`: Path to trained model checkpoint (.pth file)
- `--output`: Output directory for results
- `--visualize`: Generate visualization plots
- `--reconstruct`: Generate vector representation

## 🧠 Model Architecture

The model consists of:
- **DRN-54 Backbone**: Dilated ResNet for feature extraction
- **Pyramid Module**: Multi-scale feature fusion
- **CBAM**: Convolutional Block Attention Module
- **Segmentation Head**: Predicts corners, icons, and rooms

**Output Channels:**
- **Corners**: 70 channels (wall corners + opening corners + icon corners)
- **Icons**: 9 channels (7 icon types + 2 background classes)
- **Rooms**: 12 channels (10 room types + 2 background classes)

## 🔧 Key Modernization Updates

### PyTorch Compatibility
- ✅ Updated from PyTorch 1.0 → 2.8+
- ✅ Fixed deprecated `align_corners` in `Upsample`
- ✅ CPU/GPU flexible inference
- ✅ Modern dependency versions

### New Features
- 🆕 **Streamlit Web Interface**: Interactive web app for easy inference
- 🆕 **CPU Inference Support**: No CUDA requirement for inference
- 🆕 **Modular Inference Script**: Standalone inference without training dependencies
- 🆕 **Automatic Device Detection**: Automatically uses GPU if available, falls back to CPU
- 🆕 **Optional Pretrained Loading**: Can run without downloading pretrained weights

### Code Quality
- 🔧 **Python 3.12 Compatible**: Modern Python features and compatibility
- 🔧 **Type Hints**: Better code documentation
- 🔧 **Error Handling**: Robust error handling and user feedback
- 🔧 **Memory Efficient**: Proper memory management for inference

## 📊 Model Outputs

The model produces three types of predictions:

1. **Corner Heatmaps**: Probability maps for different corner types (walls, openings, icons)
2. **Icon Segmentation**: Classification of icon types (doors, windows, fixtures, etc.)
3. **Room Segmentation**: Classification of room types (bedroom, kitchen, bathroom, etc.)

## 🔄 Training (Optional)

To train your own model:

```bash
python train.py --restore=0
```

**Training options:**
- `--restore=0`: Train from scratch
- `--restore=1`: Resume from checkpoint
- `--batchSize=16`: Batch size
- `--LR=2.5e-4`: Learning rate
- `--numEpochs=1000`: Number of epochs

## 📁 Project Structure

```
pytorch/
├── models/                 # Model definitions
│   ├── model.py           # Main model class
│   ├── drn.py            # DRN backbone
│   ├── modules.py        # Building blocks
│   └── ...
├── datasets/              # Data loading
├── inference.py           # Command-line inference
├── streamlit_app.py       # Web interface
├── train.py              # Training script
├── utils.py              # Utilities
├── options.py            # Configuration
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🎮 Interactive Demo

The Streamlit app provides:

1. **Model Loading**: Upload or specify checkpoint path
2. **Image Upload**: Drag & drop floorplan images
3. **Real-time Processing**: Instant inference results
4. **Visualization**: Interactive plots of predictions
5. **Download Results**: Save predictions and vector representations

## 🔍 Troubleshooting

### Common Issues

**"No module named 'models'"**: Make sure you're in the `pytorch/` directory.

**"CUDA out of memory"**: The model automatically uses CPU if GPU memory is insufficient.

**"Checkpoint not found"**: Ensure the checkpoint path is correct. You need a trained model file (.pth).

**"Module load error"**: Install all dependencies with `pip install -r requirements.txt`.

### Performance Tips

- Use GPU for faster inference if available
- For batch processing, use the command-line interface
- Large images are automatically resized to 256×256

## 📚 Dependencies

Core dependencies:
- PyTorch 2.0+
- torchvision 0.15+
- NumPy 1.21+
- OpenCV 4.5+
- Streamlit 1.25+
- Pillow 9.0+
- matplotlib 3.5+

See `requirements.txt` for complete list.

## 🤝 Contributing

This modernization maintains compatibility with the original research while adding practical usability improvements. Contributions are welcome!

## 📜 License

This project maintains the same license as the original repository.

## 🙏 Acknowledgments

- Original paper: "Raster-to-Vector: Revisiting Floorplan Transformation" (ICCV 2017)
- Original authors: Chen Liu, Jiajun Wu, Pushmeet Kohli, and Yasutaka Furukawa
- Original repository: [FloorplanTransformation](https://github.com/art-programmer/FloorplanTransformation)

---

**Note**: This is a modernized version focused on inference and usability. For the original research implementation, please refer to the main repository.
