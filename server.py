from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import base64
import io
import time
import json
import requests
from datetime import datetime
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Store conversation history
conversation_history = {}

# Initialize conversation pipeline
try:
    print("Loading conversation model...")
    conversation_pipeline = pipeline(
        "text-generation",
        model="HuggingFaceH4/zephyr-7b-beta",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    CONVERSATION_MODEL_LOADED = True
    print("Conversation model loaded successfully!")
except Exception as e:
    print(f"Error loading conversation model: {e}")
    CONVERSATION_MODEL_LOADED = False

# Load the image captioning model
print("Loading image captioning model...")
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    if torch.cuda.is_available():
        model = model.to("cuda")
    print("Image captioning model loaded successfully!")
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL_LOADED = False

# Fallback captions if model fails
FALLBACK_CAPTIONS = [
    "A person walking down a city street with tall buildings in the background.",
    "A dog playing in a grassy park on a sunny day.",
    "A coffee cup sitting on a wooden table next to a laptop.",
    "A group of friends smiling and taking a selfie together.",
    "A beautiful sunset over the ocean with orange and purple skies."
]

def generate_caption(image_data=None):
    """Generate a caption for an image using the BLIP model"""
    if not image_data or not MODEL_LOADED:
        import random
        return random.choice(FALLBACK_CAPTIONS)
    
    try:
        # Remove the data URL prefix
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Process the image and generate caption
        inputs = processor(image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        out = model.generate(**inputs, max_length=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        return caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        import random
        return random.choice(FALLBACK_CAPTIONS)

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
        
        # Generate caption using the BLIP model
        caption_text = generate_caption(data.get('image'))
        
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
                caption = generate_caption(data['image'])
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
        
        # Generate response using the conversation model
        response = ""
        
        if CONVERSATION_MODEL_LOADED:
            try:
                # Format conversation history for the model
                formatted_history = ""
                for entry in conversation_history[session_id]:
                    if entry['role'] == 'user':
                        formatted_history += f"User: {entry['content']}\n"
                    elif entry['role'] == 'assistant':
                        formatted_history += f"Assistant: {entry['content']}\n"
                    else:  # system
                        formatted_history += f"System: {entry['content']}\n"
                
                # Create prompt with context
                system_prompt = """You are VisionAssist, an AI assistant that helps visually impaired users understand their surroundings. 
                You can see through the user's camera and describe what you see. Be concise, helpful, and focus on the most important details."""
                
                prompt = f"{system_prompt}\n\n{image_context}\n\n{formatted_history}\nAssistant: "
                
                # Generate response
                result = conversation_pipeline(
                    prompt,
                    max_length=200,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    do_sample=True
                )
                
                # Extract response
                generated_text = result[0]['generated_text']
                response = generated_text.split("Assistant: ")[-1].strip()
                
            except Exception as e:
                print(f"Error generating response with model: {e}")
                response = f"I can see: {caption if 'caption' in locals() else 'something interesting'}. How can I help you with that?"
        else:
            # Fallback responses if model isn't loaded
            if image_context:
                response = f"I can see: {caption}. How can I help you with that?"
            else:
                responses = [
                    "I'm here to help describe what I see. Could you show me something?",
                    "I can help identify what's in your surroundings. What would you like to know?",
                    "I'm looking at your camera feed. Is there something specific you'd like me to describe?",
                    "I'm analyzing what I see. Is there anything particular you're curious about?",
                    "I can see your surroundings and help describe them. What would you like to know?"
                ]
                import random
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
    app.run(host='0.0.0.0', port=5000, debug=True)
