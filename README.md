# VisionAssist 🔊👁️

AI-powered vision assistant with real-time image captioning and voice interaction.

## Features

- **Real-time Camera Capture** - Automatic 3-second interval image analysis
- **AI Image Captioning** - BLIP model for accurate scene description
- **Voice Interaction** - Speech recognition and synthesis for hands-free operation
- **Conversation Mode** - Context-aware chat with visual understanding
- **Response Logger** - Track all AI responses and analysis

## Quick Start

### Requirements

- Python 3.8+
- Modern web browser (Chrome/Edge recommended)
- Webcam and microphone

### Installation

```bash
# Clone repository
git clone https://github.com/ysham123/VisionAssist.git
cd VisionAssist

# Run the start script
chmod +x start.sh
./start.sh
```

Or manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Access the Application

1. Open browser to `http://localhost:5000`
2. Allow camera and microphone permissions
3. The app will automatically start capturing and analyzing images

## Usage

### Auto-Capture Mode (Default)
- Images are automatically captured every 3 seconds
- AI captions appear in real-time
- All responses are logged in the timeline

### Voice Mode
- Click the microphone button to enable voice interaction
- Speak naturally to ask questions about what the camera sees
- Responses are spoken back and logged

### Manual Controls
- **Pause/Resume** - Toggle auto-capture
- **Clear Log** - Clear the response timeline
- **Export Log** - Download responses as JSON

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/vision/caption` | POST | Generate image caption |
| `/api/v1/vision/analyze` | POST | Detailed image analysis |
| `/api/v1/conversation/sessions` | POST | Create chat session |
| `/api/v1/conversation/chat` | POST | Send chat message |

### Example Request

```bash
curl -X POST http://localhost:5000/api/v1/vision/caption \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,..."}'
```

## Configuration

Create a `.env` file based on `.env.example`:

```env
HOST=127.0.0.1
PORT=5000
DEBUG=false
ML_BACKEND_ENABLED=true
```

## Troubleshooting

### Camera Not Working
- Check browser permissions for camera access
- Ensure no other app is using the camera
- Try refreshing the page

### Voice Recognition Issues
- Use Chrome or Edge for best compatibility
- Check microphone permissions
- Ensure microphone is not muted

### ML Model Not Loading
- First run may take time to download models
- Check internet connection for model download
- Verify sufficient disk space (~1GB for models)

## Browser Support

| Feature | Chrome | Edge | Firefox | Safari |
|---------|--------|------|---------|--------|
| Camera | ✅ | ✅ | ✅ | ✅ |
| Voice Input | ✅ | ✅ | ⚠️ | ⚠️ |
| Voice Output | ✅ | ✅ | ✅ | ✅ |

## Performance Tips

- Use GPU if available (automatically detected)
- Close other camera-using applications
- Use good lighting for better image analysis
- Keep browser tabs to minimum for best performance

## License

MIT License - See LICENSE file for details

## Support

For issues or questions, please open an issue on GitHub.