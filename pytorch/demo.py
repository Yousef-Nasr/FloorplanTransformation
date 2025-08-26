#!/usr/bin/env python3

"""
Demo script to show the modernized floorplan transformation capabilities
"""

import os
import sys
from pathlib import Path
import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model import Model
from options import parse_args
from inference import preprocess_image, get_device

def create_demo_floorplan():
    """Create a simple demo floorplan image"""
    img = Image.new('RGB', (256, 256), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw room outline
    draw.rectangle([30, 30, 226, 226], outline='black', width=4)
    
    # Draw internal walls
    draw.line([30, 128, 226, 128], fill='black', width=4)  # Horizontal wall
    draw.line([128, 30, 128, 226], fill='black', width=4)  # Vertical wall
    
    # Add doors (gaps in walls)
    draw.rectangle([118, 30, 138, 34], fill='white')  # Top door
    draw.rectangle([30, 118, 34, 138], fill='white')  # Left door
    draw.rectangle([222, 118, 226, 138], fill='white')  # Right door
    
    # Add some fixtures
    draw.rectangle([50, 50, 70, 60], fill='brown', outline='black')  # Table
    draw.ellipse([180, 180, 200, 200], fill='blue', outline='black')  # Fixture
    
    return img

def demo_inference():
    """Demonstrate the inference pipeline"""
    print("🏠 Floorplan Transformation Demo")
    print("=" * 40)
    
    # Device setup
    device = get_device()
    print(f"📱 Using device: {device}")
    
    # Create demo image
    print("🖼️  Creating demo floorplan...")
    demo_img = create_demo_floorplan()
    demo_img.save('demo_floorplan.png')
    print("✅ Demo floorplan saved as 'demo_floorplan.png'")
    
    # Initialize model
    print("🧠 Initializing model...")
    options = parse_args()
    options.width = 256
    options.height = 256
    options.heatmapThreshold = 0.5
    
    model = Model(options, pretrained=False)
    model.to(device)
    model.eval()
    print(f"✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Preprocess image
    print("🔄 Preprocessing image...")
    img_tensor = preprocess_image(demo_img, (options.width, options.height))
    img_tensor = img_tensor.to(device)
    print(f"✅ Image preprocessed: {img_tensor.shape}")
    
    # Run inference
    print("🚀 Running inference...")
    with torch.no_grad():
        corner_pred, icon_pred, room_pred = model(img_tensor)
    
    # Convert to numpy for visualization
    corner_heatmaps = torch.sigmoid(corner_pred).cpu().numpy()[0]
    icon_prob = torch.softmax(icon_pred, dim=-1).cpu().numpy()[0] 
    room_prob = torch.softmax(room_pred, dim=-1).cpu().numpy()[0]
    
    print(f"✅ Inference complete!")
    print(f"   Corner predictions: {corner_heatmaps.shape}")
    print(f"   Icon predictions: {icon_prob.shape}")
    print(f"   Room predictions: {room_prob.shape}")
    
    # Create visualization
    print("📊 Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Floorplan Transformation Demo Results', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(demo_img)
    axes[0, 0].set_title('Original Floorplan')
    axes[0, 0].axis('off')
    
    # Corner heatmaps
    corner_sum = np.sum(corner_heatmaps, axis=-1)
    im1 = axes[0, 1].imshow(corner_sum, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('Corner Detections')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Icon predictions
    icon_max = np.argmax(icon_prob, axis=-1)
    im2 = axes[1, 0].imshow(icon_max, cmap='tab10')
    axes[1, 0].set_title('Icon Classifications')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Room predictions
    room_max = np.argmax(room_prob, axis=-1)
    im3 = axes[1, 1].imshow(room_max, cmap='tab10')
    axes[1, 1].set_title('Room Classifications')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualization saved as 'demo_results.png'")
    
    # Summary statistics
    print("\n📈 Results Summary:")
    print(f"   Total corner detections (>0.5): {np.sum(corner_sum > 0.5)}")
    print(f"   Max icon confidence: {np.max(icon_prob):.3f}")
    print(f"   Max room confidence: {np.max(room_prob):.3f}")
    print(f"   Dominant icon class: {np.argmax(np.sum(icon_prob, axis=(0,1)))}")
    print(f"   Dominant room class: {np.argmax(np.sum(room_prob, axis=(0,1)))}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Train a model or obtain a trained checkpoint")
    print("2. Use 'python inference.py' for real inference")
    print("3. Use 'streamlit run streamlit_app.py' for web interface")

if __name__ == "__main__":
    demo_inference()