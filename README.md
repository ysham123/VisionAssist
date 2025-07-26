# VisionAssist - Interactive AI Visual Assistant

An AI-powered real-time camera-based image captioning system with conversational voice features, designed to assist visually impaired users.

## 🌟 Features

### Core Functionality
- **Real-time Camera Captioning**: Live AI-generated descriptions of camera feed
- **Interactive Voice Conversations**: ChatGPT-style voice interactions about visual content
- **Speech Recognition**: Browser-based voice input for hands-free operation
- **Audio Feedback**: Text-to-speech for all responses and captions
- **Accessible UI**: WCAG-compliant interface optimized for screen readers

### AI Technologies
- **Vision AI**: BLIP (Bootstrapped Language-Image Pre-training) for image captioning
- **Conversational AI**: Ollama with Llama 3.2 for local, private conversations
- **Speech Processing**: Web Speech API for voice recognition and synthesis

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+
- Webcam/camera access
- Modern web browser (Chrome, Firefox, Edge)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/VisionAssist.git
   cd VisionAssist
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama** (for conversational features)
   ```bash
   # Windows
   winget install Ollama.Ollama
   
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

4. **Download AI model**
   ```bash
   ollama pull llama3.2
   ```

5. **Install frontend dependencies**
   ```bash
   cd visionassist-frontend
   npm install
   ```

### Running the Application

1. **Start Ollama service**
   ```bash
   ollama serve
   ```

2. **Start the conversational backend** (Terminal 1)
   ```bash
   python conversational_backend.py
   ```

3. **Start the original backend** (Terminal 2)
   ```bash
   python api_backend.py
   ```

4. **Start the frontend** (Terminal 3)
   ```bash
   cd visionassist-frontend
   npm start
   ```

5. **Open your browser** to `http://localhost:3000`

## 🎮 How to Use

### Basic Captioning
1. Click "📹 Start Real-Time Captioning"
2. Allow camera permissions
3. Listen to automatic scene descriptions

### Conversational Mode
1. Start basic captioning first
2. Click "💬 Chat Mode" to enable conversations
3. Use voice input to ask questions:
   - "What do you see?"
   - "How many people are here?"
   - "What colors are visible?"
   - "Describe this scene in detail"
   - "Read any text you can see"

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend │    │  Flask Backends  │    │   AI Services   │
│                 │    │                  │    │                 │
│ • Camera Feed   │◄──►│ • Vision API     │◄──►│ • BLIP Model    │
│ • Voice Input   │    │   (Port 5000)    │    │ • Transformers  │
│ • Speech Output │    │                  │    │                 │
│ • Chat History  │    │ • Conversation   │◄──►│ • Ollama        │
│                 │    │   API (Port 5001)│    │ • Llama 3.2     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
VisionAssist/
├── conversational_backend.py   # Ollama integration API
├── api_backend.py              # Original vision API
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── visionassist-frontend/      # React application
│   ├── src/
│   │   ├── App.js             # Main application
│   │   ├── components/
│   │   │   ├── VoiceInput.js  # Speech recognition
│   │   │   └── VoiceInput.css # Voice UI styles
│   │   └── App.css            # Main styles
│   └── package.json           # Node dependencies
└── README.md                   # This file
```

## 🔧 Configuration

### API Endpoints
- **Frontend**: http://localhost:3000
- **Vision API**: http://localhost:5000
- **Conversation API**: http://localhost:5001
- **Ollama**: http://localhost:11434

### Customization
Edit `config.py` to adjust:
- Model parameters
- API settings
- Performance options

## 🎯 Target Users

- **Visually impaired individuals** seeking real-time scene understanding
- **Accessibility advocates** exploring AI-assisted navigation
- **Developers** interested in multimodal AI applications
- **Researchers** studying human-AI interaction patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BLIP**: Salesforce Research for the vision-language model
- **Ollama**: For local AI inference capabilities
- **Web Speech API**: For browser-based speech recognition
- **React**: For the accessible frontend framework

## 🐛 Known Issues

- Speech recognition accuracy varies by browser and accent
- Initial model loading may take 30-60 seconds
- Camera permissions required for full functionality

## 📞 Support

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**VisionAssist** - Empowering independence through AI-powered visual assistance. 
