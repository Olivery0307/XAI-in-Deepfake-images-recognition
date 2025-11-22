# 🔍 Deepfake Detection with Explainable AI

A production-ready deepfake detection system that combines advanced deep learning models (ResNet, EfficientNet, ViT) with Explainable AI (XAI) techniques for forensic analysis.

## 🌟 Features

- **Multi-Architecture Support**: ResNet34/50, EfficientNet-B0/B4, Vision Transformer (ViT-B/16, ViT-B/32)
- **Mixed Dataset Training**: Supports Celeb-DF, YouTube, FFHQ, StyleGAN, and Stable Diffusion datasets
- **Smart Data Splitting**: Video-aware splitting for Celeb-DF/YouTube (prevents leakage), file-level for GAN images
- **Explainable AI**: Grad-CAM visualization with automatic layer selection for different architectures
- **Gemini AI Integration**: AI-powered forensic analysis reports
- **Streamlit Web App**: Professional "Forensic Detective" themed interface
- **Google Colab Ready**: Optimized for training on Colab Pro with GCS integration

## 📁 Project Structure

```
deepfake_project/
├── configs/
│   ├── __init__.py
│   └── config.py              # Central configuration (paths, hyperparameters, API keys)
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Dataset classes and smart splitting logic
│   ├── preprocessing.py       # Image transforms (training/inference)
│   ├── models.py              # Model factory (ResNet/EfficientNet/ViT)
│   ├── trainer.py             # Training loops with best model checkpointing
│   └── xai_utils.py           # Grad-CAM and Gemini API integration
├── notebooks/
│   └── main_training.ipynb    # Google Colab training orchestration notebook
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── EfficientNet.ipynb         # Original research notebook (legacy)
├── Resnet.ipynb              # Original research notebook (legacy)
└── ViT.ipynb                 # Original research notebook (legacy)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd XAI-in-Deepfake-images-recognition

# Install dependencies
pip install -r requirements.txt
```

### Training on Google Colab

1. Open [notebooks/main_training.ipynb](notebooks/main_training.ipynb) in Google Colab
2. Update the GCS bucket configuration:
   ```python
   PROJECT_ID = "your-project-id"
   BUCKET_NAME = "your-bucket-name"
   ```
3. Run all cells to:
   - Mount GCS bucket with gcsfuse
   - Load and split datasets
   - Train selected model
   - Save checkpoints and splits
4. Download the best model checkpoint for inference

### Running the Streamlit App

```bash
# Make sure you have a trained model checkpoint
streamlit run app.py
```

Then:
1. Select your model architecture from the sidebar
2. Update the model path (e.g., `resnet34.pth`)
3. (Optional) Enter your Gemini API key for forensic reports
4. Upload an image to analyze

## 🎯 Usage Guide

### Training a Model

```python
from src.data_loader import get_data_mixed_structure, LocalImageDataset, save_splits
from src.preprocessing import get_transforms
from src.models import get_model
from src.trainer import main_training_loop
from configs.config import Config
import torch
from torch.utils.data import DataLoader

# 1. Load and split data
train_data, val_data, test_data = get_data_mixed_structure(
    celeb_real_path=Config.PATHS['celeb_real'],
    youtube_real_path=Config.PATHS['youtube_real'],
    celeb_synthesis_path=Config.PATHS['celeb_synthesis'],
    ffhq_real_path=Config.PATHS['ffhq_real'],
    stylegan_fake_path=Config.PATHS['stylegan_fake'],
    stablediffusion_fake_path=Config.PATHS['stablediffusion_fake'],
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
)

# 2. Create datasets
train_transform = get_transforms(split='train', model_type='cnn')
val_transform = get_transforms(split='val', model_type='cnn')

train_dataset = LocalImageDataset(train_data[0], train_data[1], transform=train_transform)
val_dataset = LocalImageDataset(val_data[0], val_data[1], transform=val_transform)

# 3. Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. Initialize model
device = Config.get_device()
model = get_model('resnet34', num_classes=2, pretrained=True, device=device)

# 5. Train
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

history = main_training_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=10,
    device=device,
    checkpoint_dir='./checkpoints',
    model_name='resnet34'
)
```

### Running Inference

```python
from src.models import get_model, load_checkpoint
from src.preprocessing import process_uploaded_image
from src.xai_utils import compute_gradcam, get_gemini_explanation
from PIL import Image

# Load model
device = 'cuda'
model = get_model('resnet34', num_classes=2, pretrained=False, device=device)
model = load_checkpoint(model, 'resnet34_best.pth', device=device)

# Load and preprocess image
image = Image.open('test_image.jpg')
input_tensor = process_uploaded_image(image, model_type='cnn', device=device)

# Predict
output = model(input_tensor)
probs = torch.softmax(output, dim=1)
pred_class = torch.argmax(probs, dim=1)

# Generate Grad-CAM
heatmap_img, heatmap_raw = compute_gradcam(
    model=model,
    input_tensor=input_tensor,
    model_name='resnet34'
)
```

## 🏗️ Architecture Details

### Supported Models

| Model | Architecture | Input Size | Parameters | Special Notes |
|-------|--------------|------------|------------|---------------|
| ResNet34 | CNN | 224x224 | 21M | Fast, good baseline |
| ResNet50 | CNN | 224x224 | 25M | Better accuracy |
| EfficientNet-B0 | CNN | 224x224 | 5M | Efficient, mobile-ready |
| EfficientNet-B4 | CNN | 224x224 | 19M | Higher accuracy |
| ViT-B/16 | Transformer | 224x224 | 86M | Patch size 16, attention-based |
| ViT-B/32 | Transformer | 224x224 | 88M | Patch size 32, faster |

### Dataset Structure

The system handles two types of datasets differently:

**Video-based datasets** (Celeb-DF, YouTube):
- Split by **folder** (video ID) to prevent data leakage
- Ensures frames from the same video stay in the same split

**Image-based datasets** (FFHQ, StyleGAN, Stable Diffusion):
- Split by **file** for random distribution
- Uses stratified splitting for balanced classes

### Explainable AI (XAI)

**Grad-CAM Implementation:**
- Automatic target layer selection per architecture:
  - ResNet: `layer4[-1]`
  - EfficientNet: `features[-1]`
  - ViT: Last encoder block with reshape transform
- Handles ViT's unique token-based architecture
- Generates heatmaps showing model attention

**Gemini Integration:**
- Sends original image + heatmap to Gemini AI
- Receives forensic analysis correlating heatmap with facial features
- Identifies potential deepfake artifacts

## ⚙️ Configuration

Edit [configs/config.py](configs/config.py) to customize:

```python
# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
IMG_SIZE = 224

# Early stopping
PATIENCE = 3
MIN_DELTA = 0.001

# Paths (for Colab training)
GCS_MOUNT_POINT = "/content/gcs_data/"
```

## 🔑 API Keys

For Gemini AI forensic reports:

```bash
# Set environment variable
export GEMINI_API_KEY="your-api-key-here"

# Or update configs/config.py
GEMINI_API_KEY = "your-api-key-here"
```

Get your API key from: https://makersuite.google.com/app/apikey

## 📊 Training Best Practices

1. **Start with ResNet34**: Fast baseline, good for prototyping
2. **Use Colab Pro**: GPU essential for large datasets
3. **Save splits**: Use `save_splits()` for reproducibility
4. **Monitor validation**: Early stopping prevents overfitting
5. **Try data augmentation**: Already included in training transforms
6. **Experiment with architectures**: ViT may capture different artifacts than CNNs

## 🐛 Troubleshooting

**Model loading fails:**
- Check the model checkpoint format (state_dict vs model_state_dict)
- Ensure `num_classes=2` matches training

**Grad-CAM errors with ViT:**
- ViT requires `reshape_transform` - already implemented
- Check that the target layer exists

**Out of memory:**
- Reduce `BATCH_SIZE` in config
- Use smaller models (ResNet34, EfficientNet-B0)
- Enable gradient checkpointing (not yet implemented)

**GCS mount fails in Colab:**
- Verify authentication: `gcloud auth list`
- Check bucket permissions
- Ensure gcsfuse is installed

## 📚 Resources

- [PyTorch Grad-CAM Documentation](https://github.com/jacobgil/pytorch-grad-cam)
- [Gemini API Guide](https://ai.google.dev/tutorials/python_quickstart)
- [Streamlit Documentation](https://docs.streamlit.io)
- [ViT Paper (Dosovitskiy et al.)](https://arxiv.org/abs/2010.11929)

## 🤝 Contributing

This project was refactored from research notebooks into a production-ready codebase. Contributions welcome!

## 📄 License

[Add your license here]

## 👥 Authors

Oliver Squad - Initial work

---

**Note:** This is Oliver Squad. Everyone should watch Broadway show! 🎭
