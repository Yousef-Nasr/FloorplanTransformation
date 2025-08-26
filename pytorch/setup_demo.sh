#!/bin/bash

echo "🏠 Floorplan Transformation - Setup and Demo"
echo "============================================="

# Check Python version
echo "📋 Checking Python version..."
python --version

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Create a simple test image
echo "🖼️  Creating test image..."
python -c "
from PIL import Image, ImageDraw
import numpy as np

# Create a simple floorplan-like image
img = Image.new('RGB', (256, 256), color='white')
draw = ImageDraw.Draw(img)

# Draw some walls (black lines)
draw.rectangle([50, 50, 200, 200], outline='black', width=3)
draw.line([50, 125, 200, 125], fill='black', width=3)
draw.line([125, 50, 125, 200], fill='black', width=3)

# Add some doors (openings)
draw.rectangle([95, 50, 155, 53], fill='white')
draw.rectangle([50, 95, 53, 155], fill='white')

img.save('test_floorplan.png')
print('Test floorplan saved as test_floorplan.png')
"

# Test model creation
echo "🧠 Testing model creation..."
python -c "
from models.model import Model
from options import parse_args

options = parse_args()
options.width = 256
options.height = 256

model = Model(options, pretrained=False)
print(f'✅ Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters')
"

# Test streamlit app
echo "🚀 Testing Streamlit app..."
echo "Run the following command to start the Streamlit app:"
echo "streamlit run streamlit_app.py"

echo ""
echo "📚 Usage Instructions:"
echo "====================="
echo "1. To run the Streamlit app: streamlit run streamlit_app.py"
echo "2. To run inference from command line:"
echo "   python inference.py --image test_floorplan.png --checkpoint /path/to/checkpoint.pth --output results --visualize --reconstruct"
echo ""
echo "Note: You'll need a trained model checkpoint (.pth file) to run actual inference."
echo "The repository includes training code in train.py if you want to train your own model."