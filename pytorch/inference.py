import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import os
import argparse
from pathlib import Path

from utils import *
from options import parse_args
from models.model import Model
from IP import reconstructFloorplan


def get_device():
    """Get the best available device (CUDA if available, otherwise CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_model(checkpoint_path, options, pretrained=False):
    """Load the trained model from checkpoint"""
    device = get_device()
    
    model = Model(options, pretrained=pretrained)
    
    # Load the checkpoint
    if os.path.exists(checkpoint_path):
        print(f'Loading model from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    model.to(device)
    model.eval()
    return model, device


def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess input image for inference"""
    # Load image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    # Resize image
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy array and normalize
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor


def postprocess_predictions(corner_pred, icon_pred, room_pred, options):
    """Postprocess model predictions"""
    # Apply sigmoid to corner predictions
    corner_heatmaps = torch.sigmoid(corner_pred).detach().cpu().numpy()
    
    # Apply softmax to icon and room predictions
    icon_prob = F.softmax(icon_pred, dim=-1).detach().cpu().numpy()
    room_prob = F.softmax(room_pred, dim=-1).detach().cpu().numpy()
    
    return corner_heatmaps, icon_prob, room_prob


def infer_single_image(model, image_path, options, device):
    """Run inference on a single image"""
    # Preprocess image
    image_tensor = preprocess_image(image_path, (options.width, options.height))
    image_tensor = image_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        corner_pred, icon_pred, room_pred = model(image_tensor)
    
    # Postprocess predictions
    corner_heatmaps, icon_prob, room_prob = postprocess_predictions(
        corner_pred, icon_pred, room_pred, options
    )
    
    return {
        'corner_heatmaps': corner_heatmaps[0],  # Remove batch dimension
        'icon_prob': icon_prob[0],
        'room_prob': room_prob[0],
        'original_image': image_tensor.cpu().numpy()[0]
    }


def visualize_predictions(predictions, output_path=None):
    """Visualize the predictions"""
    corner_heatmaps = predictions['corner_heatmaps']
    icon_prob = predictions['icon_prob']
    room_prob = predictions['room_prob']
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    original_img = predictions['original_image'].transpose(1, 2, 0)
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Corner heatmaps (sum across all corner types)
    corner_sum = np.sum(corner_heatmaps, axis=-1)
    axes[0, 1].imshow(corner_sum, cmap='hot')
    axes[0, 1].set_title('Corner Heatmaps')
    axes[0, 1].axis('off')
    
    # Icon predictions (max across classes)
    icon_max = np.argmax(icon_prob, axis=-1)
    axes[1, 0].imshow(icon_max, cmap='tab20')
    axes[1, 0].set_title('Icon Predictions')
    axes[1, 0].axis('off')
    
    # Room predictions (max across classes)
    room_max = np.argmax(room_prob, axis=-1)
    axes[1, 1].imshow(room_max, cmap='tab20')
    axes[1, 1].set_title('Room Predictions')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    return fig


def reconstruct_floorplan(predictions, options):
    """Reconstruct vector floorplan from predictions"""
    try:
        corner_heatmaps = predictions['corner_heatmaps']
        icon_prob = predictions['icon_prob']
        room_prob = predictions['room_prob']
        
        # Use the IP solver to reconstruct the floorplan
        representation = reconstructFloorplan(
            corner_heatmaps, 
            icon_prob, 
            room_prob,
            options.heatmapThreshold
        )
        
        return representation
    except Exception as e:
        print(f"Warning: Floorplan reconstruction failed: {e}")
        return None


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Floorplan Transformation Inference')
    parser.add_argument('--image', required=True, help='Path to input floorplan image')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    parser.add_argument('--reconstruct', action='store_true', help='Reconstruct vector representation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Create options (using defaults from training)
    options = parse_args()
    options.width = 256
    options.height = 256
    options.heatmapThreshold = 0.5
    
    try:
        # Load model
        model, device = load_model(args.checkpoint, options, pretrained=False)
        print(f"Model loaded successfully on device: {device}")
        
        # Run inference
        predictions = infer_single_image(model, args.image, options, device)
        print("Inference completed successfully")
        
        # Save predictions
        output_base = output_dir / Path(args.image).stem
        
        # Visualize if requested
        if args.visualize:
            try:
                import matplotlib.pyplot as plt
                fig = visualize_predictions(predictions, f"{output_base}_visualization.png")
                plt.close(fig)
            except ImportError:
                print("Warning: matplotlib not available for visualization")
        
        # Reconstruct floorplan if requested
        if args.reconstruct:
            representation = reconstruct_floorplan(predictions, options)
            if representation is not None:
                # Save representation to file
                np.save(f"{output_base}_representation.npy", representation)
                print(f"Vector representation saved to {output_base}_representation.npy")
        
        # Save raw predictions
        np.save(f"{output_base}_predictions.npy", predictions)
        print(f"Raw predictions saved to {output_base}_predictions.npy")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())