# VisionAssist 🔊👁️

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)](https://flask.palletsprojects.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> An AI-powered vision assistant with natural voice conversation capabilities.

VisionAssist is an accessible application that uses computer vision and natural language processing to help users understand their surroundings through AI-generated image descriptions and voice-based conversations.

![VisionAssist Demo](https://github.com/ysham123/VisionAssist/raw/main/static/demo.gif)

## ✨ Features

- **Real-time Camera Integration** - Instant access to device camera
- **AI Image Captioning** - Powered by state-of-the-art BLIP vision model
- **Natural Voice Conversations** - Ask questions about what you see
- **Hands-free Operation** - Full voice input and output support
- **Accessible Interface** - Designed with accessibility in mind
- **Conversation History** - Review past interactions
- **Cross-browser Support** - Works on major browsers

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Web browser with camera and microphone access
- Internet connection (for initial model download)

### Installation

1. **Clone and Navigate**:
```bash
git clone https://github.com/ysham123/VisionAssist.git
cd VisionAssist
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download NLTK Data** (for text processing):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

4. **Start the Server**:
```bash
python server.py
```

5. **Access the Application**:
   - Open browser to `http://localhost:5000`
   - Grant camera permissions for real-time capture

## 🔧 API Documentation

### Core Endpoints

#### Image Captioning
```http
POST /api/v1/vision/caption
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "include_attention": true,
  "include_gradcam": false
}
```

**Response**:
```json
{
  "success": true,
  "caption": "A person walking down a city street with tall buildings",
  "source": "ml_model",
  "model_info": {
    "architecture": "MobileNet + LSTM with Attention",
    "feature_extractor": "MobileNetV2",
    "decoder": "LSTM with Bahdanau Attention"
  },
  "attention_weights": [[0.1, 0.3, 0.2, ...]],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Advanced Analysis
```http
POST /api/v1/vision/analyze
Content-Type: application/json
4. The AI will respond both visually and with speech

### Voice Commands

- "What do you see?"
- "Can you describe this in more detail?"
- "What color is the [object]?"
- "How many [objects] are there?"
- "Where is the [object] located?"

## 🔧 Technical Architecture

### Frontend

- **Interface**: HTML5, CSS3, JavaScript
- **Camera Access**: WebRTC API
- **Voice Input**: Web Speech API (SpeechRecognition)
- **Voice Output**: Web Speech API (SpeechSynthesis)

### Backend

- **Server**: Flask (Python)
- **Image Captioning**: BLIP model (Transformers)
- **Conversation**: LLM-based response generation
- **Image Processing**: PIL/OpenCV

## 🌐 Browser Compatibility

| Browser | Camera | Voice Input | Voice Output |
|---------|--------|-------------|-------------|
| Chrome  | ✅     | ✅          | ✅          |
| Edge    | ✅     | ✅          | ✅          |
| Firefox | ✅     | ⚠️ Limited  | ✅          |
| Safari  | ✅     | ⚠️ Limited  | ✅          |

## ⚠️ Troubleshooting

- **Microphone Access**: Ensure you grant microphone permissions when prompted
- **Camera Access**: Ensure you grant camera permissions when prompted
- **Speech Recognition Errors**: If you encounter network errors, check your internet connection
- **No Sound**: Check that your device's volume is turned up and not muted
- **Model Loading**: First-time startup may take longer as models are downloaded

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


