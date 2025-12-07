"""
Deepfake Forensic Detective - Streamlit App
Enhanced with modular architecture support for ResNet, EfficientNet, and ViT
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import sys

# Import modular components
from src.models import get_model, load_checkpoint
from src.preprocessing import process_uploaded_image
from src.xai_utils import compute_xai_visualization, get_gemini_explanation
from configs.config import Config

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Deepfake Forensic Detective",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. Custom CSS - Forensic Detective Theme
# ==========================================
st.markdown("""
<style>
    /* Global Font & Theme */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #E0E0E0;
    }

    h1, h2, h3 {
        font-family: 'Roboto Mono', monospace;
        font-weight: 700;
        color: #00E5FF; /* Neon Cyan */
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #1E1E1E;
    }

    /* Custom Card Container */
    .forensic-card {
        background-color: transparent;
        border: 1px transparent;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: none;
    }

    /* Result Badges */
    .badge-fake {
        background: linear-gradient(135deg, #FF6B6B 0%, #EE5A6F 100%);
        color: white;
        padding: 12px 28px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1.3em;
        border: none;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .badge-real {
        background: linear-gradient(135deg, #51CF66 0%, #37B24D 100%);
        color: white;
        padding: 12px 28px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1.3em;
        border: none;
        box-shadow: 0 4px 15px rgba(81, 207, 102, 0.4);
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .badge-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 30px 0;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        height: 3em;
        background-color: #00E5FF;
        color: #000;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00B8D4;
        box-shadow: 0 0 10px #00E5FF;
    }

    /* Image Captions */
    .img-caption {
        text-align: center;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.9em;
        color: #9E9E9E;
        margin-top: 8px;
    }

    /* Report Box */
    .report-box {
        background-color: #2D3748;
        border-left: 4px solid #00E5FF;
        padding: 15px;
        border-radius: 4px;
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        color: #E2E8F0;
    }

    .report-box h4 {
        color: #00E5FF;
        margin-top: 0;
    }

    /* Status badges */
    .status-ok {
        color: #51CF66;
        font-weight: 600;
    }

    .status-error {
        color: #FF6B6B;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. Model Loading with Architecture Support
# ==========================================

@st.cache_resource
def load_deepfake_model(model_name: str, model_path: str):
    """
    Load trained deepfake detection model with architecture-specific handling

    Args:
        model_name: Model architecture name
        model_path: Path to checkpoint file

    Returns:
        model: Loaded model
        device: Device (cuda/mps/cpu)
    """
    try:
        # Get device
        device = Config.get_device()

        # Initialize model architecture
        model = get_model(
            model_name=model_name,
            num_classes=Config.NUM_CLASSES,
            pretrained=False,
            device=device
        )

        # 1. Handle path: Ensure it looks in 'models/' directory if not provided
        if not os.path.exists(model_path):
            # Try joining with 'models/' folder
            potential_path = os.path.join("models", os.path.basename(model_path))
            if os.path.exists(potential_path):
                model_path = potential_path

        # 2. Load weights directly (No checkpoint dictionary logic)
        print(f"Loading weights from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        
        # Load state dictionary into model
        model.load_state_dict(state_dict)

        model.eval()

        return model, device

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None


# ==========================================
# 4. Main Application
# ==========================================

def main():
    # --- Sidebar Configuration ---
    with st.sidebar:
        st.title("🕵️‍♂️ Forensic Toolkit")
        st.markdown("---")

        # Model Selection
        st.markdown("### 🧠 Model Configuration")

        model_name = st.selectbox(
            "Select Model Architecture",
            options=['resnet34', 'efficientnet_b4', 'vit_b_16'],
            index=0,
            help="Choose the model architecture for analysis"
        )

        # Determine model type
        model_type = 'vit' if 'vit' in model_name else 'cnn'

        # Model file path
        model_path = st.text_input(
            "Model Path",
            value=f"models/{model_name}.pth",
            help="Path to the trained model file"
        )

        st.markdown("---")

        # API Key Input
        st.markdown("### 🔑 Gemini API Settings")
        api_key = st.text_input(
            "Gemini API Key (Optional)",
            type="password",
            value=Config.GEMINI_API_KEY,
            help="Required for AI-powered forensic analysis report"
        )

        st.markdown("---")

        # System Status
        st.markdown("### 🛠️ System Status")

        # Check model file
        if os.path.exists(model_path):
            st.markdown(f'<span class="status-ok">✓ Model file found</span>', unsafe_allow_html=True)

            # Load model
            with st.spinner("Loading model..."):
                model, device = load_deepfake_model(model_name, model_path)

            if model is not None:
                st.markdown(f'<span class="status-ok">✓ Model loaded: {model_name}</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="status-ok">✓ Device: {device}</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-error">✗ Model loading failed</span>', unsafe_allow_html=True)
                return
        else:
            st.markdown(f'<span class="status-error">✗ Model not found: {model_path}</span>', unsafe_allow_html=True)
            st.info("💡 Place your .pth file in the app directory or update the path above")
            return

        st.markdown("---")

        # About Section
        st.markdown("### ℹ️ About")
        st.info(
            f"""
            **Deepfake Forensic Detective**

            This tool uses deep learning models trained on mixed datasets
            (Celeb-DF, YouTube, FFHQ, StyleGAN, Stable Diffusion) to detect
            synthetic media.

            **Current Model:** {model_name}
            **Architecture:** {model_type.upper()}

            Features:
            - Multi-architecture support (ResNet, EfficientNet, ViT)
            - XAI visualization (Grad-CAM for CNNs, Attention Rollout for ViT)
            - AI-powered forensic analysis (Gemini)
            """
        )

    # --- Main Content Area ---
    st.markdown("# 🔍 Deepfake Forensic Detective")
    st.markdown("### Analyze suspicious images using advanced Computer Vision & Explainable AI")

    # File Upload
    st.markdown('<div class="forensic-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📎 Upload Suspect Image (JPG/PNG/JPEG)",
        type=["jpg", "png", "jpeg"],
        help="Upload an image to analyze for deepfake artifacts"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Process uploaded image
    if uploaded_file is not None:
        # Display analysis in centered column
        col_1, col_main, col_3 = st.columns([1, 3, 1])

        with col_main:
            with st.spinner("🔬 Running forensic analysis algorithms..."):
                # Load image
                image = Image.open(uploaded_file).convert("RGB")

                # Preprocess for model
                input_tensor = process_uploaded_image(
                    image,
                    model_type=model_type,
                    img_size=Config.IMG_SIZE,
                    device=device
                )

                # Run inference
                with torch.no_grad():
                    output = model(input_tensor)
                    if model_type == 'vit': 
                      probs = torch.softmax(output.logits, dim=1)
                    else:
                      probs = torch.softmax(output, dim=1)
                    conf, pred_class = torch.max(probs, 1)

                # Get prediction
                LABELS = {0: "REAL", 1: "FAKE"}
                label = LABELS[pred_class.item()]
                score = conf.item()

                # Generate XAI visualization (auto-switches between Grad-CAM and Attention Rollout)
                try:
                    heatmap_img, heatmap_raw, xai_method = compute_xai_visualization(
                        model=model,
                        input_tensor=input_tensor,
                        model_name=model_name,
                        target_layer=None,  # Auto-select
                        original_size=image.size  # Resize to original image dimensions
                    )
                    heatmap_pil = Image.fromarray(heatmap_img.astype('uint8'))
                    xai_success = True
                except Exception as e:
                    st.warning(f"⚠️ XAI visualization generation failed: {e}")
                    heatmap_pil = None
                    xai_success = False
                    xai_method = "Unknown"

        # --- Display Results ---
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")

        # Result Badge
        if label == "FAKE":
            st.markdown(f"""
            <div class="badge-container">
                <span class="badge-fake">⚠️ DETECTED: SYNTHETIC MEDIA</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="badge-container">
                <span class="badge-real">✓ VERIFIED: REAL IMAGE</span>
            </div>
            """, unsafe_allow_html=True)

        # Confidence Bar
        st.progress(score, text=f"Model Confidence: {score:.1%}")

        # Visual Evidence: Side-by-side comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="forensic-card" style="text-align:center;">', unsafe_allow_html=True)
            st.markdown('<p class="img-caption">📷 Exhibit A: Original Image</p>', unsafe_allow_html=True)
            st.image(image, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="forensic-card" style="text-align:center;">', unsafe_allow_html=True)
            st.markdown(f'<p class="img-caption">🔥 Exhibit B: AI Attention Heatmap ({xai_method if xai_success else "N/A"})</p>', unsafe_allow_html=True)
            if xai_success and heatmap_pil:
                st.image(heatmap_pil, width="stretch")
                st.caption(f"Red/yellow regions indicate areas the model focused on for its decision (using {xai_method})")
            else:
                st.warning("XAI visualization not available")
            st.markdown('</div>', unsafe_allow_html=True)

        # --- Gemini Forensic Report ---
        st.markdown("---")
        if api_key and xai_success:
            st.markdown("### 📝 AI-Powered Forensic Report")

            with st.spinner("🤖 Generative AI is analyzing the evidence..."):
                try:
                    # Resize images for API efficiency
                    img_resized = image.resize((Config.IMG_SIZE, Config.IMG_SIZE))

                    report = get_gemini_explanation(
                        api_key=api_key,
                        original_image=img_resized,
                        heatmap_image=heatmap_pil,
                        prediction=label,
                        confidence=score
                    )

                    st.markdown(f"""
                    <div class="report-box">
                        <h4>🔍 Forensic Analysis Report (via {xai_method})</h4>
                        <p>{report}</p>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Error generating Gemini report: {e}")

        elif not api_key:
            st.info("💡 Enter your Gemini API key in the sidebar to unlock AI-powered forensic analysis")
        elif not xai_success:
            st.warning("⚠️ Gemini analysis requires successful XAI visualization generation")

        # --- Metrics ---
        st.markdown("---")
        st.markdown("### 📈 Metrics")

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.metric("Prediction", label)

        with col_m2:
            st.metric("Confidence", f"{score:.2%}")


# ==========================================
# 5. Entry Point
# ==========================================

if __name__ == "__main__":
    main()
