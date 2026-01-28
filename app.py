import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# Set page config
st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

# Define the model architecture (must match training)
class BalancedEmotionCNN(nn.Module):
    def __init__(self):
        super(BalancedEmotionCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            # Block 2: 112 -> 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            # Block 3: 56 -> 28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),

            # Block 4: 28 -> 14
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Constants
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EMOTION_CLASSES = ['happy', 'neutral', 'surprise']  # Adjust if different

# Emotion emoji mapping
EMOTION_EMOJIS = {
    'happy': '😊',
    'neutral': '😐',
    'surprise': '😮',
    'sad': '😢',
    'angry': '😠',
    'fear': '😨',
    'disgust': '🤢'
}

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BalancedEmotionCNN().to(device)
    
    try:
        model.load_state_dict(torch.load('Balanced_Emotion_Model (1).pt', map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

def predict_emotion(image, model, device):
    """Make prediction on an image"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Transform image
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, prediction = probs.max(1)
    
    # Get emotion
    emotion = EMOTION_CLASSES[prediction.item()]
    conf_value = confidence.item()
    
    # Get all probabilities
    all_probs = {EMOTION_CLASSES[i]: probs[0, i].item() for i in range(len(EMOTION_CLASSES))}
    
    return emotion, conf_value, all_probs

def main():
    # Header
    st.title("😊 Facial Emotion Recognition")
    st.markdown("Upload an image to detect the emotion")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info(
            "This app uses a Convolutional Neural Network (CNN) to classify facial expressions "
            "into different emotion categories."
        )
        
        st.header("Model Details")
        st.write("**Architecture:** Custom CNN")
        st.write("**Input Size:** 224x224")
        st.write("**Classes:**")
        for emotion in EMOTION_CLASSES:
            emoji = EMOTION_EMOJIS.get(emotion, '🙂')
            st.write(f"- {emoji} {emotion.capitalize()}")
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check if 'Balanced_Emotion_Model (1).pt' exists.")
        return
    
    st.success(f"✅ Model loaded successfully! Using device: {device}")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📸 Upload Image", "📹 Use Webcam"])
    
    with tab1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a facial image for emotion detection"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Prediction Results")
                
                with st.spinner("Analyzing emotion..."):
                    emotion, confidence, all_probs = predict_emotion(image, model, device)
                
                # Display results
                emoji = EMOTION_EMOJIS.get(emotion, '🙂')
                st.markdown(f"### {emoji} **{emotion.upper()}**")
                st.metric("Confidence", f"{confidence:.1%}")
                
                # Progress bars for all emotions
                st.markdown("---")
                st.subheader("All Probabilities")
                
                # Sort by probability
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                
                for emo, prob in sorted_probs:
                    emoji = EMOTION_EMOJIS.get(emo, '🙂')
                    st.write(f"{emoji} **{emo.capitalize()}**")
                    st.progress(prob)
                    st.caption(f"{prob:.1%}")
    
    with tab2:
        st.subheader("📹 Webcam Capture")
        st.info("Click the button below to capture an image from your webcam")
        
        camera_image = st.camera_input("Take a picture")
        
        if camera_image is not None:
            # Display captured image
            image = Image.open(camera_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Captured Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Prediction Results")
                
                with st.spinner("Analyzing emotion..."):
                    emotion, confidence, all_probs = predict_emotion(image, model, device)
                
                # Display results
                emoji = EMOTION_EMOJIS.get(emotion, '🙂')
                st.markdown(f"### {emoji} **{emotion.upper()}**")
                st.metric("Confidence", f"{confidence:.1%}")
                
                # Progress bars for all emotions
                st.markdown("---")
                st.subheader("All Probabilities")
                
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                
                for emo, prob in sorted_probs:
                    emoji = EMOTION_EMOJIS.get(emo, '🙂')
                    st.write(f"{emoji} **{emo.capitalize()}**")
                    st.progress(prob)
                    st.caption(f"{prob:.1%}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center'>"
        "<p>Built with Streamlit and PyTorch</p>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
