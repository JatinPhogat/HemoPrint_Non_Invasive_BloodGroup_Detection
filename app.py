import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path
import numpy as np

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
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(image, model1, model2, use_fusion=True):
    # Preprocess
    img_tensor = preprocess_image(image).to(device)
    
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
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Prediction")
        
        # Make prediction
        with st.spinner("Analyzing fingerprint..."):
            predicted_class, confidence, all_probs, model_name = predict(
                image, model1, model2, use_fusion
            )
        
        # Display results
        st.markdown(f"**Model:** {model_name}")
        st.markdown("---")
        
        # Main prediction
        st.markdown(f"### 🎯 Predicted Blood Group:")
        st.markdown(f"# **{predicted_class}**")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        
        # Progress bar for confidence
        st.progress(confidence / 100)
        
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
            st.progress(prob / 100)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "<p>🔬 Blood Group Detection System | Powered by PyTorch & Streamlit</p>"
    "</div>",
    unsafe_allow_html=True
)
