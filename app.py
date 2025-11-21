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
# [UI升級] 設定 page_icon 為偵探圖示
st.set_page_config(
    page_title="Deepfake Forensic Detective", 
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# [UI升級] 注入自定義 CSS 讓介面更專業
st.markdown("""
<style>
    /* 全域字體優化 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* 按鈕樣式優化 */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        height: 3em;
        transition: all 0.3s ease;
    }
    
    /* 結果卡片樣式 */
    .result-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* 報告區塊樣式 */
    .report-box {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 25px;
        margin-top: 20px;
        color: #212529;
    }
    
    /* 圖片標題樣式 */
    .img-caption {
        text-align: center;
        font-weight: 600;
        color: #495057;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_NAME = 'tf_efficientnetv2_s'
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Core Functions (完全保留您的邏輯)
# ==========================================

@st.cache_resource
def load_model(model_path):
    try:
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def process_and_predict(model, image_pil):
    img_cv2 = np.array(image_pil)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR) 
    img_cv2_resized = cv2.resize(img_cv2, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_cv2_resized, cv2.COLOR_BGR2RGB)
    
    img_float_norm = np.float32(img_rgb) / 255.0
    input_tensor = preprocess_image(img_rgb, 
                                    mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225]).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
    
    is_fake = prob > 0.5
    label = "Fake" if is_fake else "Real"
    confidence = prob if is_fake else 1 - prob
    
    target_layers = [model.conv_head]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    visualization = show_cam_on_image(img_float_norm, grayscale_cam, use_rgb=True)
    heatmap_pil = Image.fromarray(visualization)
    
    return label, confidence, heatmap_pil

def ask_gemini(api_key, original_img, heatmap_img, label, confidence):
    if not api_key:
        return "⚠️ Error: Please enter your Gemini API Key in the sidebar settings."
    
    genai.configure(api_key=api_key)
    
    # 保留您的變數設定
    model_name = 'gemini-2.5-flash-lite' 
    
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        return f"Model Error: {e}. Try changing the model name in the code."

    prompt_text = [
        "You are a senior digital forensics expert specializing in Deepfake detection.",
        f"My AI model (EfficientNetV2) analyzed an image and predicted: **{label}** (Confidence: {confidence:.2%}).",
        "",
        "I am providing:",
        "1. The original image.",
        "2. A Grad-CAM heatmap (Red/Yellow areas = Model focus).",
        "",
        "Your Task:",
        "1. Analyze why the model likely made this decision based on the heatmap.",
        "2. Look for visual artifacts in eyes, mouth, skin texture, or boundaries.",
        "3. CRITICAL: If the model predicted REAL, but you see artifacts, provide your counter-opinion.",
        "4. Please only provide a concise and brief summary (only 3 bullet points) of your final verdict."
    ]

    try:
        with st.spinner(f'🤖 AI Forensic Expert is analyzing the evidence...'):
            response = model.generate_content([*prompt_text, original_img, heatmap_img])
        return response.text
    except Exception as e:
        return f"Gemini API Error: {e}"

# ==========================================
# 3. Main UI Layout (大幅美化版)
# ==========================================
def main():
    # --- Header Section ---
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        st.image("https://cdn-icons-png.flaticon.com/512/7504/7504250.png", width=80) # 示意圖示
    with col_title:
        st.title("Deepfake Forensic Detective")
        st.markdown("#### 🛡️ Advanced AI-Powered Image Authentication System")
    
    st.markdown("---")

    # --- Sidebar: Professional Settings ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("Enter your credentials and system parameters below.")
        
        api_key = st.text_input("Gemini API Key", type="password", help="Required for forensic report generation.")
        
        default_model_path = "efficientnetv2_finetuned_v4.pth"
        model_path = st.text_input("Model Path (.pth)", value=default_model_path)
        
        if os.path.exists(model_path):
            st.success(f"✅ Model Linked: {os.path.basename(model_path)}")
        else:
            st.error("❌ Model Not Found")

        st.markdown("---")
        st.markdown("**System Status**")
        st.caption(f"Device: `{device}`")
        st.caption(f"Model Architecture: `{MODEL_NAME}`")

    # --- Main Content ---
    
    # 使用 Expander 讓介面保持乾淨，隱藏詳細說明
    with st.expander("ℹ️ How this system works"):
        st.markdown("""
        1. **Visual Analysis (CNN):** Uses EfficientNetV2 to scan for pixel-level artifacts.
        2. **Heatmap Generation:** Creates a Grad-CAM heatmap to visualize *where* the AI is looking.
        3. **Semantic Reasoning (LLM):** Uses Google Gemini to interpret the findings and provide a human-readable report.
        """)

    st.subheader("1. Evidence Upload")
    uploaded_file = st.file_uploader("Upload a suspect image (JPG/PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load Image
        image = Image.open(uploaded_file).convert('RGB')
        
        # 顯示上傳預覽 (放在一個乾淨的容器中)
        with st.container():
            col_preview, col_info = st.columns([1, 2])
            with col_preview:
                st.image(image, caption="Uploaded Evidence", width=200)
            with col_info:
                st.info(f"**File Name:** {uploaded_file.name}")
                st.info(f"**Resolution:** {image.size[0]} x {image.size[1]} px")

        # 檢查模型
        if not os.path.exists(model_path):
            st.error(f"❌ Critical Error: Model file missing at `{model_path}`")
            st.stop()
            
        model = load_model(model_path)
        
        if model:
            st.markdown("---")
            st.subheader("2. Forensic Analysis")
            
            # [UI升級] 使用全寬的主要按鈕 (Primary Button)
            if st.button("🔍 Initiate Deepfake Analysis", type="primary", use_container_width=True):
                
                # 進度條
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Processing Visual Artifacts...")
                progress_bar.progress(30)
                
                # Step A: CNN Prediction
                label, conf, heatmap = process_and_predict(model, image)
                progress_bar.progress(60)
                
                status_text.text("Generating Heatmap Visualization...")
                progress_bar.progress(100)
                status_text.empty() # 清除文字
                progress_bar.empty() # 清除進度條
                
                # [UI升級] 結果顯示區 - 使用 HTML/CSS 製作漂亮的卡片
                bg_color = "#ffebee" if label == "Fake" else "#e8f5e9"
                text_color = "#c62828" if label == "Fake" else "#2e7d32"
                border_color = "#ef5350" if label == "Fake" else "#66bb6a"
                
                st.markdown(f"""
                <div class="result-card" style="background-color: {bg_color}; border: 2px solid {border_color};">
                    <h2 style="color: {text_color}; margin: 0;">Prediction: {label}</h2>
                    <p style="font-size: 1.2em; color: {text_color}; margin-top: 5px;">Confidence: <strong>{conf:.1%}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # 圖片並排顯示區
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, use_container_width=True)
                    st.markdown('<div class="img-caption">Original Evidence</div>', unsafe_allow_html=True)
                with col2:
                    st.image(heatmap, use_container_width=True)
                    st.markdown('<div class="img-caption">AI Attention Heatmap (Grad-CAM)</div>', unsafe_allow_html=True)
                
                # [UI升級] Gemini 報告區
                if api_key:
                    st.markdown("---")
                    st.subheader("3. Gemini Forensic Report")
                    
                    report = ask_gemini(api_key, image, heatmap, label, conf)
                    
                    # 使用自定義樣式的盒子包裹報告
                    st.markdown(f"""
                    <div class="report-box">
                        <h4>📝 Expert Analysis Summary</h4>
                        {report}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Please enter your Gemini API Key in the sidebar to unlock the full forensic report.")

if __name__ == "__main__":
    main()