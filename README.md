# 😊 Facial Emotion Recognition App

A deep learning-powered web application that detects emotions from facial images using a custom CNN model built with PyTorch.

## 🎯 Features

- **Real-time Emotion Detection**: Upload images or use webcam to detect emotions
- **3 Emotion Classes**: Happy, Neutral, and Surprise
- **Confidence Scores**: See prediction probabilities for all emotion classes
- **User-Friendly Interface**: Built with Streamlit for an intuitive experience

## 🚀 Live Demo

[Visit the App](https://your-app-url.streamlit.app) *(Update with your Streamlit Cloud URL)*

## 🛠️ Installation

### Local Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/emotion-recognition.git
cd emotion-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
emotion-recognition/
├── app.py                          # Main Streamlit application
├── Balanced_Emotion_Model (1).pt   # Trained PyTorch model
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🧠 Model Architecture

- **Type**: Custom Convolutional Neural Network (CNN)
- **Input Size**: 224x224 RGB images
- **Architecture**: 4 convolutional blocks with batch normalization and max pooling
- **Output**: 3 emotion classes

### Model Details:
- Block 1: Conv2d(3→64) + ReLU + BatchNorm + MaxPool
- Block 2: Conv2d(64→128) + ReLU + BatchNorm + MaxPool
- Block 3: Conv2d(128→256) + ReLU + BatchNorm + MaxPool
- Block 4: Conv2d(256→512) + ReLU + BatchNorm + MaxPool
- Classifier: Fully connected layers (512→256→3) with dropout

## 📊 Emotions Detected

| Emotion | Emoji |
|---------|-------|
| Happy | 😊 |
| Neutral | 😐 |
| Surprise | 😮 |

## 🎨 Usage

1. **Upload Mode**: Click "Browse files" to upload a facial image
2. **Webcam Mode**: Use the webcam tab to capture real-time photos
3. View the predicted emotion with confidence scores

## 🔧 Technologies Used

- **PyTorch**: Deep learning framework
- **Streamlit**: Web application framework
- **Torchvision**: Image transformations
- **PIL**: Image processing

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 👤 Author

Your Name - [Your GitHub](https://github.com/yourusername)

---

Made with ❤️ using PyTorch and Streamlit
