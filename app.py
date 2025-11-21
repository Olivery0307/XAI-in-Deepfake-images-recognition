import streamlit as st
import torch
import timm
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import os

# ==========================================
# 1. Configuration & Constants
# ==========================================
st.set_page_config(
    page_title="Deepfake Forensic Detective",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Visual Appeal ---
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
</style>
""", unsafe_allow_html=True)

# --- Constants ---
MODEL_NAME = "resnet34" # Must match your training
NUM_CLASSES = 2
IMG_SIZE = 224
LABELS = {0: "REAL", 1: "FAKE"}

# ==========================================
# 2. Helper Functions
# ==========================================

@st.cache_resource
def load_model(model_path):
    """Loads the trained model (cached for speed)."""
    try:
        # Initialize model architecture
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        
        # Load weights
        # Ensure map_location handles CPU/GPU correctly
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        model.eval()
        model.to(device)
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def process_image(image):
    """Prepares image for model inference and Grad-CAM."""
    # Resize for display/processing
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    rgb_img = np.float32(img_resized) / 255
    
    # Standard ImageNet normalization
    input_tensor = preprocess_image(
        rgb_img, 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    return img_resized, rgb_img, input_tensor

def get_gradcam(model, input_tensor, rgb_img, target_layer, device, original_size=None):
    """Generates Grad-CAM heatmap overlay."""
    cam = GradCAM(model=model, target_layers=[target_layer])

    # We don't specify targets, so it defaults to the highest scoring class
    grayscale_cam = cam(input_tensor=input_tensor.to(device))[0, :]

    # Overlay heatmap on original image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    heatmap_img = Image.fromarray(visualization)

    # Resize heatmap to match original image dimensions if provided
    if original_size is not None:
        heatmap_img = heatmap_img.resize(original_size, Image.LANCZOS)

    return heatmap_img

def ask_gemini(api_key, original_img, heatmap_img, prediction_label, confidence):
    """Sends images to Gemini for qualitative analysis."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash') # Faster, cheaper model
        
        prompt = f"""
        You are a forensic AI analyst. A deepfake detection model has analyzed an image.
        
        - **Model Prediction:** {prediction_label}
        - **Confidence Score:** {confidence:.1%}
        
        **Visual Evidence:**
        1.  **Image 1:** The original image being analyzed.
        2.  **Image 2:** A Grad-CAM heatmap. The RED/YELLOW areas show exactly where the AI model found "suspicious" or "significant" features.
        
        **Your Task:**
        Write a short, professional forensic report (3-4 sentences).
        1.  Acknowledge the model's prediction.
        2.  Analyze the heatmap: What specific facial features is the model focusing on? (e.g., eyes, mouth, hairline, background).
        3.  Explain WHY: If it's FAKE, do these highlighted areas correspond to common deepfake artifacts (blurring, inconsistencies)? If REAL, does the focus look natural?
        
        Keep the tone objective and technical.
        """
        
        response = model.generate_content([prompt, original_img, heatmap_img])
        return response.text
    except Exception as e:
        return f"Error generating report: {str(e)}"

# ==========================================
# 3. Main App Layout
# ==========================================

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.title("🕵️‍♂️ Forensic Toolkit")
        st.markdown("---")
        
        # API Key Input
        api_key = st.text_input("Gemini API Key (Optional)", type="password", help="Required for the text explanation report.")
        
        st.markdown("### 🛠️ System Status")
        model_path = "resnet34.pth" # HARDCODED: Update this filename if needed
        
        # Check if model exists
        if os.path.exists(model_path):
            st.success(f"Model Loaded: {MODEL_NAME}")
            model, device = load_model(model_path)
        else:
            st.error(f"Model not found: {model_path}")
            st.info("Please place your .pth file in the same directory as app.py")
            return # Stop execution if no model

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info(
            "This tool uses three pretrained models trained on a mixed dataset (Celeb-DF + StyleGAN + Diffusion) "
            "to detect synthetic faces. You can switch between each model using the sidebar. "
            "It uses **Grad-CAM** to visualize decision regions."
        )

    # --- Main Content ---
    st.markdown("# 🔍 Deepfake Forensic Detective")
    st.markdown("### Analyze suspicious images using advanced Computer Vision & XAI")
    
    # File Uploader in a "Card"
    st.markdown('<div class="forensic-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a suspect image (JPG/PNG)", type=["jpg", "png", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # --- 1. Preprocessing ---
        col_1, col_2, col_3 = st.columns([1, 2, 1])
        with col_2:
            with st.spinner("Running forensic analysis algorithms..."):
                image = Image.open(uploaded_file).convert("RGB")
                img_resized, rgb_img, input_tensor = process_image(image)
                
                # --- 2. Inference ---
                output = model(input_tensor.to(device))
                probs = torch.softmax(output, dim=1)
                conf, pred_class = torch.max(probs, 1)
                
                label = LABELS[pred_class.item()]
                score = conf.item()
                
                # --- 3. Grad-CAM ---
                # ResNet-50 target layer is typically 'layer4' (last conv block)
                target_layer = model.layer4[-1]
                heatmap = get_gradcam(model, input_tensor, rgb_img, target_layer, device, original_size=image.size)

        # --- 4. Display Results ---
        st.markdown("### 📊 Analysis Results")
        
        # Result Banner
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

        # Visual Evidence Columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="forensic-card" style="text-align:center;">', unsafe_allow_html=True)
            st.markdown('<p class="img-caption">Exhibit A: Original Image</p>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="forensic-card" style="text-align:center;">', unsafe_allow_html=True)
            st.markdown('<p class="img-caption">Exhibit B: AI Attention Heatmap</p>', unsafe_allow_html=True)
            st.image(heatmap, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- 5. Gemini Report ---
        if api_key:
            st.markdown("### 📝 Automated Forensic Report")
            with st.spinner("Generative AI is compiling the report..."):
                report = ask_gemini(api_key, img_resized, heatmap, label, score)
                
                st.markdown(f"""
                <div class="report-box">
                    <p>{report}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Enter your Gemini API key in the sidebar to unlock the qualitative analysis report.")

if __name__ == "__main__":
    main()