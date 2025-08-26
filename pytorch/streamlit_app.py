import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import os
import tempfile
from pathlib import Path

# Import our inference functions
from inference import load_model, infer_single_image, preprocess_image, reconstruct_floorplan
from options import parse_args


def init_session_state():
    """Initialize session state variables"""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'device' not in st.session_state:
        st.session_state.device = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False


def load_model_cached(checkpoint_path):
    """Load model and cache it in session state"""
    if not st.session_state.model_loaded or st.session_state.model is None:
        options = parse_args()
        options.width = 256
        options.height = 256
        options.heatmapThreshold = 0.5
        
        try:
            model, device = load_model(checkpoint_path, options, pretrained=False)
            st.session_state.model = model
            st.session_state.device = device
            st.session_state.options = options
            st.session_state.model_loaded = True
            return True, "Model loaded successfully!"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
    return True, "Model already loaded"


def create_visualization(predictions):
    """Create visualization of predictions"""
    try:
        import matplotlib.pyplot as plt
        
        corner_heatmaps = predictions['corner_heatmaps']
        icon_prob = predictions['icon_prob']
        room_prob = predictions['room_prob']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original image
        original_img = predictions['original_image'].transpose(1, 2, 0)
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Corner heatmaps (sum across all corner types)
        corner_sum = np.sum(corner_heatmaps, axis=-1)
        im1 = axes[0, 1].imshow(corner_sum, cmap='hot')
        axes[0, 1].set_title('Corner Heatmaps')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Icon predictions (max across classes)
        icon_max = np.argmax(icon_prob, axis=-1)
        im2 = axes[1, 0].imshow(icon_max, cmap='tab20')
        axes[1, 0].set_title('Icon Predictions')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Room predictions (max across classes)
        room_max = np.argmax(room_prob, axis=-1)
        im3 = axes[1, 1].imshow(room_max, cmap='tab20')
        axes[1, 1].set_title('Room Predictions')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig
    except ImportError:
        st.error("Matplotlib is required for visualization")
        return None


def main():
    st.set_page_config(
        page_title="Floorplan Transformation",
        page_icon="🏠",
        layout="wide"
    )
    
    st.title("🏠 Floorplan Transformation")
    st.markdown("Upload a floorplan image to extract vector representations of walls, icons, and rooms.")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    # Model checkpoint upload/path
    checkpoint_option = st.sidebar.radio(
        "Model Checkpoint Source",
        ["Upload checkpoint file", "Use local path"]
    )
    
    checkpoint_path = None
    if checkpoint_option == "Upload checkpoint file":
        uploaded_checkpoint = st.sidebar.file_uploader(
            "Upload model checkpoint (.pth file)", 
            type=['pth', 'pt']
        )
        if uploaded_checkpoint is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                tmp_file.write(uploaded_checkpoint.read())
                checkpoint_path = tmp_file.name
    else:
        checkpoint_path = st.sidebar.text_input(
            "Path to checkpoint file",
            placeholder="/path/to/checkpoint.pth"
        )
    
    # Load model button
    if checkpoint_path and st.sidebar.button("Load Model"):
        with st.sidebar:
            with st.spinner("Loading model..."):
                success, message = load_model_cached(checkpoint_path)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    # Model status
    if st.session_state.model_loaded:
        st.sidebar.success("✅ Model loaded and ready")
        device_info = str(st.session_state.device)
        st.sidebar.info(f"Device: {device_info}")
    else:
        st.sidebar.warning("⚠️ Please load a model first")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose a floorplan image",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a rasterized floorplan image"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Floorplan", use_container_width=True)
            
            # Processing options
            st.subheader("Processing Options")
            visualize = st.checkbox("Generate visualization", value=True)
            reconstruct = st.checkbox("Reconstruct vector representation", value=True)
            
            # Process button
            if st.button("Process Floorplan", type="primary", disabled=not st.session_state.model_loaded):
                if not st.session_state.model_loaded:
                    st.error("Please load a model first")
                else:
                    with st.spinner("Processing floorplan..."):
                        try:
                            # Run inference
                            predictions = infer_single_image(
                                st.session_state.model,
                                image,
                                st.session_state.options,
                                st.session_state.device
                            )
                            
                            # Store results in session state for display
                            st.session_state.predictions = predictions
                            st.session_state.show_results = True
                            
                            st.success("Processing completed!")
                            
                        except Exception as e:
                            st.error(f"Error during processing: {str(e)}")
    
    with col2:
        st.header("Results")
        
        if hasattr(st.session_state, 'show_results') and st.session_state.show_results:
            predictions = st.session_state.predictions
            
            # Visualization
            if visualize:
                st.subheader("Predictions Visualization")
                try:
                    fig = create_visualization(predictions)
                    if fig:
                        st.pyplot(fig)
                        # Clear the figure to free memory
                        import matplotlib.pyplot as plt
                        plt.close(fig)
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
            
            # Statistics
            st.subheader("Prediction Statistics")
            corner_heatmaps = predictions['corner_heatmaps']
            icon_prob = predictions['icon_prob']
            room_prob = predictions['room_prob']
            
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("Corner detections", f"{np.sum(corner_heatmaps > 0.5)}")
            with col2b:
                st.metric("Icon confidence", f"{np.max(icon_prob):.3f}")
            with col2c:
                st.metric("Room confidence", f"{np.max(room_prob):.3f}")
            
            # Vector reconstruction
            if reconstruct:
                st.subheader("Vector Representation")
                try:
                    representation = reconstruct_floorplan(predictions, st.session_state.options)
                    if representation is not None:
                        st.success("Vector representation reconstructed successfully!")
                        
                        # Show some basic info about the representation
                        if hasattr(representation, 'shape'):
                            st.info(f"Representation shape: {representation.shape}")
                        
                        # Download button for the representation
                        representation_bytes = io.BytesIO()
                        np.save(representation_bytes, representation)
                        representation_bytes.seek(0)
                        
                        st.download_button(
                            label="Download Vector Representation",
                            data=representation_bytes.getvalue(),
                            file_name="floorplan_representation.npy",
                            mime="application/octet-stream"
                        )
                    else:
                        st.warning("Vector reconstruction failed")
                except Exception as e:
                    st.error(f"Error during vector reconstruction: {str(e)}")
            
            # Download raw predictions
            st.subheader("Download Results")
            predictions_bytes = io.BytesIO()
            np.save(predictions_bytes, predictions)
            predictions_bytes.seek(0)
            
            st.download_button(
                label="Download Raw Predictions",
                data=predictions_bytes.getvalue(),
                file_name="predictions.npy",
                mime="application/octet-stream"
            )
        
        else:
            st.info("Upload an image and click 'Process Floorplan' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Floorplan Transformation** - Raster-to-Vector Floorplan Conversion using Deep Learning\n\n"
        "Based on the paper: *Raster-to-Vector: Revisiting Floorplan Transformation* (ICCV 2017)"
    )


if __name__ == "__main__":
    main()