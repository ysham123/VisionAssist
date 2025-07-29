# VisionAssist Simplified Prototype

A minimal implementation of VisionAssist that demonstrates real-time image captioning using a BLIP model.

## Features

- Camera access and image capture
- Real-time image captioning using BLIP model
- Basic conversation functionality
- Speech output using browser's built-in speech synthesis

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:

## Usage

### Basic Usage
1. Click "Start Camera" to enable your webcam
2. Click "Capture Image" to take a photo
3. The AI will generate a caption for the image
4. Type questions about the image in the chat box
5. Toggle speech output with the switch

### Voice-Based Conversation Mode
1. Click "Start Conversation Mode" to begin
2. The system will automatically start the camera if needed
3. It will capture an image and ask "What can you see in this image?"
4. After the AI responds, it will activate voice input
5. Speak naturally to ask follow-up questions about what the camera sees
6. The AI will respond both visually and with speech

### Voice Input
- Click the "Voice Input" button to start speaking
- Click again to stop if needed
- Your speech will be converted to text and sent to the AI
- If voice input fails, you can always type your questions

## Technologies

- Python/Flask for the backend
- HTML/CSS/JavaScript for the frontend
- BLIP model for image captioning
- Transformers for conversation
- Web Speech API for voice input/output
- WebRTC for camera access

## Browser Compatibility

For the best experience with voice features, use one of these browsers:
- Google Chrome (recommended)
- Microsoft Edge
- Safari (limited voice recognition support)

## Troubleshooting

- **Microphone Access**: Ensure you grant microphone permissions when prompted
- **Camera Access**: Ensure you grant camera permissions when prompted
- **Speech Recognition Errors**: If you encounter network errors, check your internet connection
- **No Sound**: Check that your device's volume is turned up and not muted
