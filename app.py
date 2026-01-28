import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Emotion Recognition AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean styling
st.markdown("""
<style>
    /* Headers */
    h1 {
        color: #1a202c;
        font-weight: 700;
        text-align: center;
    }
    
    h3 {
        color: #2d3748;
        font-weight: 600;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f7fafc;
    }
    
    /* Image styling */
    [data-testid="stImage"] {
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

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
EMOTION_CLASSES = ['angry', 'happy', 'neutral']  # Must match model training order

# Emotion color mapping
EMOTION_COLORS = {
    'angry': '#dc2626',    # Red
    'happy': '#16a34a',    # Green
    'neutral': '#2563eb',  # Blue
    'sad': '#6b7280',      # Gray
    'surprise': '#ea580c', # Orange
    'fear': '#9333ea',     # Purple
    'disgust': '#0891b2'   # Cyan
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

def create_probability_chart(all_probs):
    """Create a modern bar chart for emotion probabilities"""
    emotions = list(all_probs.keys())
    probabilities = [all_probs[e] * 100 for e in emotions]
    colors = [EMOTION_COLORS.get(e, '#667eea') for e in emotions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probabilities,
            y=emotions,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(255, 255, 255, 0.5)', width=2)
            ),
            text=[f'{p:.1f}%' for p in probabilities],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='Emotion Confidence Levels',
            font=dict(size=16, color='#1a202c')
        ),
        xaxis=dict(
            title='Confidence (%)',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            range=[0, 100]
        ),
        yaxis=dict(
            title='',
            categoryorder='total ascending'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(size=12, color='#4a5568')
    )
    
    return fig

def main():
    # Header
    st.title("Facial Emotion Recognition")
    st.write("Upload an image or use your webcam to detect emotions")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info(
            "This application uses a Convolutional Neural Network (CNN) to analyze "
            "facial expressions and classify emotions."
        )
        
        st.header("Model Info")
        st.write("**Architecture:** Custom CNN")
        st.write("**Input Size:** 224×224")
        st.write("**Parameters:** ~5M")
        
        st.header("Detectable Emotions")
        for emotion in EMOTION_CLASSES:
            st.write(f"• {emotion.capitalize()}")
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check if 'Balanced_Emotion_Model (1).pt' exists.")
        return
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, rgba(72, 187, 120, 0.2), rgba(56, 178, 172, 0.2)); 
    padding: 1rem; border-radius: 10px; text-align: center; color: #2D3748; margin-bottom: 2rem;
    border-left: 4px solid #48BB78;'>
    <b>Model Status:</b> Loaded Successfully | <b>Device:</b> {device}
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Upload Image", "Use Webcam"])
    
    with tab1:
        st.markdown("### Upload a facial image for analysis")
        
        # File uploader
        uploaded_file = st.file_uploade")
        return
    
    st.success(f"✓ Model loaded | Device: {device}"uploaded_file)
            
            col1, col2 = st.columns([1, 1], gap="large")
            
            with col1:
                st.markdown("#### Input Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("#### Analysis Results")
                
                with st.spinner("Analyzing facial expression..."):
           subheader("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png']}; margin-bottom: 1.5rem;'>
                    <h2 style='color: {color}; margin: 0; font-size: 2.5rem; text-transform: uppercase;'>
                    {emotion}
                    </h2>
                    <p style='color: #2D3748; font-size: 1.5rem; margin-top: 0.5rem; font-weight: 600;'>
                    {confidence:.1%} Confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
                2)
            
            with col1:
                st.write("**Input Image**")
                st.image(image, use_container_width=True)
            
            with col2:
                st.write("**Results**")
                
                with st.spinner("Analyzing..."):
                    emotion, confidence, all_probs = predict_emotion(image, model, device)
                
                # Display prediction
                color = EMOTION_COLORS.get(emotion, '#2563eb')
                st.markdown(f"""
                <div style='background-color: {color}15; padding: 1.5rem; border-radius: 8px; 
                text-align: center; border-left: 4px solid {color}; margin-bottom: 1rem;'>
                    <h2 style='color: {color}; margin: 0; text-transform: uppercase;'>{emotion}</h2>
                    <p style='color: #374151; font-size: 1.2rem; margin-top: 0.5rem;'>
                    {confidence:.1%} confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # C
            with col2:
                st.markdown("#### Analysis Results")
                
                with st.spinner("Analyzing facial expression..."):
                    emotion, confidence, all_probs = predict_emotion(image, model, device)
                
                # Display main prediction with styled box
           subheader("Use Webcam")
        st.info("Take a photo using your webcam")
        
        camera_image = st.camera_input("Take a picture
                </div>
                """, unsafe_allow_html=True)
                
                # Probability chart
                fig = create_probability_chart(all_probs)
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); 
    border-radius: 15px; margin-top: 2rem;'>
        <p style='color: white; font2)
            
            with col1:
                st.write("**Captured Image**")
                st.image(image, use_container_width=True)
            
            with col2:
                st.write("**Results**")
                
                with st.spinner("Analyzing..."):
                    emotion, confidence, all_probs = predict_emotion(image, model, device)
                
                # Display prediction
                color = EMOTION_COLORS.get(emotion, '#2563eb')
                st.markdown(f"""
                <div style='background-color: {color}15; padding: 1.5rem; border-radius: 8px; 
                text-align: center; border-left: 4px solid {color}; margin-bottom: 1rem;'>
                    <h2 style='color: {color}; margin: 0; text-transform: uppercase;'>{emotion}</h2>
                    <p style='color: #374151; font-size: 1.2rem; margin-top: 0.5rem;'>
                    {confidence:.1%} confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # C---")
    st.caption("Built with PyTorch and Streamlit"