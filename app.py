import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageEnhance
import json
from pathlib import Path
import numpy as np
import cv2

# Page config
st.set_page_config(
    page_title="Blood Group Detection",
    page_icon="🩸",
    layout="centered"
)

# Paths
MODEL1_PATH = Path(r'C:\Users\phoga\Desktop\HemoPrint\model_results\model1_resnet18')
MODEL2_PATH = Path(r'C:\Users\phoga\Desktop\HemoPrint\model_results\model2_vgg16')
PREP_PATH = Path(r'C:\Users\phoga\Desktop\HemoPrint\model_results\preprocessing')

# Load config
@st.cache_data
def load_config():
    with open(PREP_PATH / 'config.json', 'r') as f:
        config = json.load(f)
    return config

config = load_config()
BLOOD_GROUPS = config['classes']
NUM_CLASSES = config['num_classes']
IMAGE_SIZE = config['image_size']

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
@st.cache_resource
def load_models():
    # Load ResNet-18
    model1 = models.resnet18(pretrained=False)
    model1.fc = nn.Linear(model1.fc.in_features, NUM_CLASSES)
    model1.load_state_dict(torch.load(MODEL1_PATH / 'best_model.pth', map_location=device))
    model1 = model1.to(device)
    model1.eval()
    
    # Load VGG-16
    model2 = models.vgg16(pretrained=False)
    model2.classifier[6] = nn.Linear(model2.classifier[6].in_features, NUM_CLASSES)
    model2.load_state_dict(torch.load(MODEL2_PATH / 'best_model.pth', map_location=device))
    model2 = model2.to(device)
    model2.eval()
    
    return model1, model2

# Image preprocessing
def preprocess_image(image, enhance_for_phone=False):
    """
    Preprocess image for model input
    enhance_for_phone: Apply aggressive preprocessing for phone photos (JPG/JPEG) to match BMP training data quality
    """
    if enhance_for_phone:
        # Convert PIL to numpy for OpenCV processing
        img_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale (fingerprints are grayscale features)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Remove JPEG compression artifacts with bilateral filter
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Step 2: Aggressive CLAHE for ridge enhancement (models trained on high-contrast BMP)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Step 3: Morphological operations to enhance ridge structure
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # Step 4: Strong unsharp masking for ridge clarity
        gaussian = cv2.GaussianBlur(morph, (0, 0), 2.0)
        sharpened = cv2.addWeighted(morph, 2.5, gaussian, -1.5, 0)
        
        # Step 5: Enhance edges (ridge patterns)
        sobelx = cv2.Sobel(sharpened, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(sharpened, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(sobel / sobel.max() * 255)
        
        # Combine original with edge-enhanced
        combined = cv2.addWeighted(sharpened, 0.7, sobel, 0.3, 0)
        
        # Step 6: Histogram matching to simulate BMP-like distribution
        combined = cv2.equalizeHist(combined)
        
        # Step 7: Final noise reduction while preserving ridges
        final = cv2.fastNlMeansDenoising(combined, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Convert back to RGB (3 channels) for model input
        img_rgb = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)
        
        # Convert back to PIL
        image = Image.fromarray(img_rgb)
    else:
        # Apply basic contrast enhancement for regular images
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
    
    # Standard transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(image, model1, model2, use_fusion=True, enhance_for_phone=False):
    # Preprocess
    img_tensor = preprocess_image(image, enhance_for_phone).to(device)
    
    with torch.no_grad():
        # Get predictions from both models
        outputs1 = model1(img_tensor)
        outputs2 = model2(img_tensor)
        
        # Convert to probabilities
        probs1 = F.softmax(outputs1, dim=1)
        probs2 = F.softmax(outputs2, dim=1)
        
        if use_fusion:
            # Fusion: Average probabilities
            avg_probs = (probs1 + probs2) / 2
            probs = avg_probs
            model_name = "Fusion Model (ResNet-18 + VGG-16)"
        else:
            # Use ResNet-18 only
            probs = probs1
            model_name = "ResNet-18"
        
        # Get prediction
        confidence, predicted = probs.max(1)
        predicted_class = BLOOD_GROUPS[predicted.item()]
        confidence_score = confidence.item() * 100
        
        # Get all class probabilities
        all_probs = probs.cpu().numpy()[0] * 100
        
    return predicted_class, confidence_score, all_probs, model_name

# UI
st.title("🩸 Blood Group Detection from Fingerprint")
st.write("Upload a fingerprint image to detect the blood group using AI")

# Sidebar
st.sidebar.header("Settings")
use_fusion = st.sidebar.checkbox("Use Fusion Model", value=True, 
                                  help="Combines ResNet-18 + VGG-16 for better accuracy")
enhance_phone = st.sidebar.checkbox("📱 Phone Photo Mode", value=False,
                                    help="Enable for photos taken with phone camera (JPG/JPEG). Applies extra preprocessing: contrast enhancement, denoising, and sharpening.")

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    "This app uses deep learning models trained on fingerprint images "
    "to predict blood groups. The fusion model combines ResNet-18 and VGG-16 "
    "for improved accuracy."
)

# Load models
with st.spinner("Loading models..."):
    model1, model2 = load_models()

st.success("✅ Models loaded successfully!")

# File upload
uploaded_file = st.file_uploader("Choose a fingerprint image...", 
                                 type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, width=350)
    
    with col2:
        st.subheader("Prediction")
        
        # Make prediction
        with st.spinner("Analyzing fingerprint..."):
            predicted_class, confidence, all_probs, model_name = predict(
                image, model1, model2, use_fusion, enhance_phone
            )
        
        # Display results
        st.markdown(f"**Model:** {model_name}")
        st.markdown("---")
        
        # Main prediction
        st.markdown(f"### 🎯 Predicted Blood Group:")
        st.markdown(f"# **{predicted_class}**")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        
        # Progress bar for confidence
        st.progress(float(confidence / 100))
        
        # Show all probabilities (always show)
        st.markdown("---")
        st.subheader("All Blood Group Probabilities")
        
        # Sort by probability
        prob_dict = {BLOOD_GROUPS[i]: all_probs[i] for i in range(NUM_CLASSES)}
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        
        for blood_group, prob in sorted_probs:
            # Highlight predicted class
            if blood_group == predicted_class:
                st.markdown(f"**{blood_group}:** {prob:.2f}% ⭐")
            else:
                st.markdown(f"{blood_group}: {prob:.2f}%")
            st.progress(float(prob / 100))

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "<p>🔬 Blood Group Detection System | Powered by PyTorch & Streamlit</p>"
    "</div>",
    unsafe_allow_html=True
)
