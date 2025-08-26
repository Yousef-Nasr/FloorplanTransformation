#!/bin/bash

echo "🏠 Floorplan Transformation - Setup and Demo"
echo "============================================="

# Check Python version
echo "📋 Checking Python version..."
python --version

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

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

# Run demo
echo "🚀 Running complete demo..."
python demo.py

echo ""
echo "📚 Usage Instructions:"
echo "====================="
echo "1. To run the Streamlit app: streamlit run streamlit_app.py"
echo "2. To run inference from command line:"
echo "   python inference.py --image demo_floorplan.png --checkpoint /path/to/checkpoint.pth --output results --visualize --reconstruct"
echo "3. To run the interactive demo: python demo.py"
echo ""
echo "Note: You'll need a trained model checkpoint (.pth file) to run actual inference."
echo "The repository includes training code in train.py if you want to train your own model."
echo ""
echo "✅ Setup complete! Check demo_floorplan.png and demo_results.png for example output."