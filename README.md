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

1. Clone the repository
   ```bash
   git clone https://github.com/ysham123/VisionAssist.git
   cd VisionAssist
   ```

2. Create and activate a virtual environment
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on `.env.example`
   ```bash
   cp .env.example .env
   ```

5. Run the application
   ```bash
   python server.py
   ```

6. Open your browser and navigate to http://localhost:5000

## 🎯 Usage Guide

### Basic Mode

1. **Start Camera**: Click the camera button to activate your device's camera
2. **Capture Image**: Take a snapshot of what you want to understand
3. **View Caption**: The AI will automatically describe what it sees
4. **Ask Questions**: Type or speak questions about the image
5. **Toggle Speech**: Enable/disable voice responses

### Voice Conversation Mode

1. Click "Start Conversation Mode" to begin an interactive session
2. The system will automatically:
   - Start your camera (if not already active)
   - Capture an image
   - Describe what it sees
   - Activate voice input for follow-up questions
3. Speak naturally to ask questions about what the camera sees
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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

Yosef Shammout - [@ysham123](https://github.com/ysham123)

Project Link: [https://github.com/ysham123/VisionAssist](https://github.com/ysham123/VisionAssist)

---

<p align="center">Made with ❤️ for accessibility</p>
