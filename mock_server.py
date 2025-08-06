from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import time
from datetime import datetime
import random

app = Flask(__name__)
CORS(app)

# Store conversation history
conversation_history = {}

# Fallback captions for mock responses
FALLBACK_CAPTIONS = [
    "A person walking down a city street with tall buildings in the background.",
    "A dog playing in a grassy park on a sunny day.",
    "A coffee cup sitting on a wooden table next to a laptop.",
    "A group of friends smiling and taking a selfie together.",
    "A beautiful sunset over the ocean with orange and purple skies."
]

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/api/v1/vision/caption', methods=['POST'])
def caption():
    """Generate caption for an image"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing image data',
                'error_code': 'MISSING_DATA'
            }), 400
        
        # Generate mock caption
        caption_text = random.choice(FALLBACK_CAPTIONS)
        
        return jsonify({
            'success': True,
            'caption': caption_text,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'SERVER_ERROR'
        }), 500

@app.route('/api/v1/conversation/session/create', methods=['POST'])
def create_session():
    """Create a new conversation session"""
    session_id = f"session_{int(time.time())}_{int(time.time() % 1000)}"
    return jsonify({
        'success': True,
        'session_id': session_id,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/v1/conversation/chat', methods=['POST'])
def chat():
    """Process a chat message with context awareness"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing message data',
                'error_code': 'MISSING_DATA'
            }), 400
        
        user_message = data['message']
        session_id = data.get('session_id', 'default_session')
        
        # Initialize session history if it doesn't exist
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        # Get image caption if provided
        image_context = ""
        if 'image' in data and data['image']:
            try:
                caption = random.choice(FALLBACK_CAPTIONS)
                image_context = f"The current image shows: {caption}. "
                # Add the caption to conversation history
                conversation_history[session_id].append({
                    'role': 'system',
                    'content': f"[Image description: {caption}]"
                })
            except Exception as e:
                print(f"Error processing image: {e}")
        
        # Add user message to history
        conversation_history[session_id].append({
            'role': 'user',
            'content': user_message
        })
        
        # Limit history size to prevent context overflow
        if len(conversation_history[session_id]) > 10:
            conversation_history[session_id] = conversation_history[session_id][-10:]
        
        # Generate mock responses based on user input
        responses = [
            f"I can see: {caption if 'caption' in locals() else 'something interesting'}. How can I help you with that?",
            "I'm here to help describe what I see. Could you show me something?",
            "I can help identify what's in your surroundings. What would you like to know?",
            "I'm looking at your camera feed. Is there something specific you'd like me to describe?",
            "I'm analyzing what I see. Is there anything particular you're curious about?"
        ]
        
        # Add some context-awareness to responses
        if "hello" in user_message.lower() or "hi" in user_message.lower():
            response = "Hello! I'm VisionAssist. How can I help you today?"
        elif "what" in user_message.lower() and "see" in user_message.lower():
            response = f"I can see {random.choice(FALLBACK_CAPTIONS).lower()}"
        elif "help" in user_message.lower():
            response = "I can help you identify objects, read text, describe scenes, and answer questions about what I see through your camera."
        elif "thank" in user_message.lower():
            response = "You're welcome! Is there anything else I can help you with?"
        else:
            response = random.choice(responses)
        
        # Add assistant response to history
        conversation_history[session_id].append({
            'role': 'assistant',
            'content': response
        })
        
        return jsonify({
            'success': True,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'SERVER_ERROR'
        }), 500

if __name__ == '__main__':
    print("Starting VisionAssist Mock Backend Server...")
    print("This is a simplified mock server for demonstration purposes.")
    print("Endpoints available:")
    print("  - /api/v1/vision/caption")
    print("  - /api/v1/conversation/session/create")
    print("  - /api/v1/conversation/chat")
    app.run(host='0.0.0.0', port=5000, debug=True)
