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

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main background and theme */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Content containers */
    div[data-testid="stVerticalBlock"] > div {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    /* Headers */
    h1 {
        color: #2D3748;
        font-weight: 700;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    h2 {
        color: #4A5568;
        font-weight: 600;
        font-size: 1.8rem !important;
    }
    
    h3 {
        color: #667eea;
        font-weight: 600;
        font-size: 1.4rem !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li {
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        padding: 1rem;
        border: 2px dashed #667eea;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #667eea;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(102, 126, 234, 0.1);
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        color: #4A5568;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    
    /* Image containers */
    [data-testid="stImage"] {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
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

# Emotion color mapping for modern UI
EMOTION_COLORS = {
    'angry': '#E53E3E',    # Red
    'happy': '#48BB78',    # Green
    'neutral': '#4299E1',  # Blue
    'sad': '#718096',      # Gray
    'surprise': '#ED8936', # Orange
    'fear': '#9F7AEA',     # Purple
    'disgust': '#38B2AC'   # Teal
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
            text='Emotion Probability Distribution',
            font=dict(size=18, color='#2D3748', family='Arial Black')
        ),
        xaxis=dict(
            title='Confidence (%)',
       markdown(f"""
    <div style='background: linear-gradient(135deg, rgba(72, 187, 120, 0.2), rgba(56, 178, 172, 0.2)); 
    padding: 1rem; border-radius: 10px; text-align: center; color: #2D3748; margin-bottom: 2rem;
    border-left: 4px solid #48BB78;'>
    <b>Model Status:</b> Loaded Successfully | <b>Device:</b> {device}
    </div>
    """,st.markdown("### Upload a facial image for analysis")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1], gap="large")
            
            with col1:
                st.markdown("#### Input Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("#### Analysis Results")
                
                with st.spinner("Analyzing facial expression..."):
                    emotion, confidence, all_probs = predict_emotion(image, model, device)
                
                # Display main prediction with styled box
                color = EMOTION_COLORS.get(emotion, '#667eea')
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {color}20, {color}40); 
                padding: 2rem; border-radius: 15px; text-align: center; 
                border: 3px solid {color}; margin-bottom: 1.5rem;'>
                    <h2 style='color: {color}; margin: 0; font-size: 2.5rem; text-transform: uppercase;'>
                    {emotion}
                    </h2>
                    <p style='color: #2D3748; font-size: 1.5rem; margin-top: 0.5rem; font-weight: 600;'>
                    {confidence:.1%} Confidence
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability chart
           markdown("### Capture image directly from your webcam")
        st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; 
        margin-bottom: 1rem; border-left: 4px solid #667eea;'>
        <p style='margin: 0; color: #4A5568;'>
        Click the camera button below to take a picture for real-time emotion analysis
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        camera_image = st.camera_input("Take a picture", label_visibility="collapsed")
        
        if camera_image is not None:
            # Display captured image
            image = Image.open(camera_image)
            
            col1, col2 = st.columns([1, 1], gap="large")
            
            with col1:
                st.markdown("#### Captured Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("#### Analysis Results")
                
                with st.spinner("Analyzing facial expression..."):
                    emotion, confidence, all_probs = predict_emotion(image, model, device)
                
                # Display main prediction with styled box
                color = EMOTION_COLORS.get(emotion, '#667eea')
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {color}20, {color}40); 
                padding: 2rem; border-radius: 15px; text-align: center; 
                border: 3px solid {color}; margin-bottom: 1.5rem;'>
                    <h2 style='color: {color}; margin: 0; font-size: 2.5rem; text-transform: uppercase;'>
                    {emotion}
                    </h2>
                    <p style='color: #2D3748; font-size: 1.5rem; margin-top: 0.5rem; font-weight: 600;'>
                    {confidence:.1%} Confidence
                 <br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); 
    border-radius: 15px; margin-top: 2rem;'>
        <p style='color: white; font-size: 0.9rem; margin: 0;'>
        Powered by <b>PyTorch</b> & <b>Streamlit</b> | Deep Learning Emotion Recognition System
        </p>
    </div>
    """, unsafe_allow_html=True            st.plotly_chart(fig, use_container_width=Trueemotion detection"
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
