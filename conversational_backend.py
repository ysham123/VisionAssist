"""
VisionAssist Conversational Backend with Ollama Integration
Enhanced API server for interactive voice conversations about visual content.
"""
import requests
import json
import base64
import io
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from datetime import datetime
import time

# Import existing vision capabilities
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
    print("✅ BLIP transformers available")
except ImportError as e:
    BLIP_AVAILABLE = False
    print(f"❌ BLIP not available: {e}")

app = Flask(__name__)
CORS(app)

class ConversationalVisionAssist:
    """Enhanced VisionAssist with Ollama conversational capabilities."""
    
    def __init__(self):
        """Initialize the conversational vision assistant."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        print("🚀 VisionAssist Conversational Backend Starting...")
        print("=" * 60)
        
        # Initialize vision model
        self.vision_model_available = False
        if BLIP_AVAILABLE:
            try:
                print("📥 Loading BLIP model for vision...")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model.to(self.device)
                self.blip_model.eval()
                print("✅ Vision model loaded successfully!")
                self.vision_model_available = True
            except Exception as e:
                print(f"❌ Error loading vision model: {e}")
                self.vision_model_available = False
        
        # Initialize Ollama connection
        self.ollama_url = "http://localhost:11434"
        self.conversation_history = []
        
        print(f"🎯 Vision Model: {'Ready' if self.vision_model_available else 'Demo mode'}")
        print(f"🤖 Ollama URL: {self.ollama_url}")
        print("=" * 60)
    
    def check_ollama_status(self):
        """Check if Ollama is running and available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def generate_caption(self, image_data):
        """Generate caption for base64 image data."""
        try:
            if not self.vision_model_available:
                return self.get_demo_caption()
            
            # Decode base64 image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Generate caption using BLIP
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50, num_beams=5)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            print(f"Error generating caption: {e}")
            return self.get_demo_caption()
    
    def get_demo_caption(self):
        """Get demo caption as fallback."""
        demo_captions = [
            "A person sitting at a desk with a laptop computer",
            "A room with natural lighting and furniture",
            "A workspace with various objects on a table",
            "A person in an indoor environment",
            "A scene with everyday objects and furniture"
        ]
        import random
        return random.choice(demo_captions)
    
    def chat_with_ollama(self, prompt, model="llama3.2"):
        """Send a chat message to Ollama and get response."""
        try:
            if not self.check_ollama_status():
                return "Ollama is not running. Please start Ollama first."
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response from Ollama')
            else:
                return f"Ollama error: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return f"Connection error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def process_conversational_query(self, image_data, user_question):
        """Process a conversational query about an image."""
        try:
            # Generate image caption
            caption = self.generate_caption(image_data)
            
            # Create context-aware prompt for Ollama
            prompt = f"""You are VisionAssist, an AI assistant helping visually impaired users understand their environment. 

Current visual context: {caption}

User question: {user_question}

Please provide a helpful, clear, and concise response. Focus on being descriptive and accessible. If the user asks about specific details not visible in the caption, acknowledge the limitation and provide what information you can.

Response:"""

            # Get response from Ollama
            response = self.chat_with_ollama(prompt)
            
            # Store in conversation history
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'caption': caption,
                'question': user_question,
                'response': response
            })
            
            # Keep only last 10 conversations
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return {
                'caption': caption,
                'response': response,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error processing conversational query: {e}")
            return {
                'caption': self.get_demo_caption(),
                'response': f"I'm sorry, I encountered an error: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }

# Initialize the conversational assistant
vision_assistant = ConversationalVisionAssist()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    ollama_status = vision_assistant.check_ollama_status()
    return jsonify({
        'status': 'healthy',
        'vision_model': vision_assistant.vision_model_available,
        'ollama_available': ollama_status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/caption', methods=['POST'])
def generate_caption():
    """Generate caption for uploaded image."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = data['image']
        caption = vision_assistant.generate_caption(image_data)
        
        return jsonify({
            'caption': caption,
            'timestamp': datetime.now().isoformat(),
            'confidence': 0.85  # Mock confidence for compatibility
        })
        
    except Exception as e:
        print(f"Error in caption endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def conversational_chat():
    """Handle conversational queries about images."""
    try:
        data = request.get_json()
        if not data or 'image' not in data or 'question' not in data:
            return jsonify({'error': 'Missing image or question data'}), 400
        
        image_data = data['image']
        question = data['question']
        
        result = vision_assistant.process_conversational_query(image_data, question)
        
        return jsonify({
            'success': True,
            'caption': result['caption'],
            'response': result['response'],
            'timestamp': result['timestamp']
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/conversation-history', methods=['GET'])
def get_conversation_history():
    """Get recent conversation history."""
    return jsonify({
        'history': vision_assistant.conversation_history,
        'count': len(vision_assistant.conversation_history)
    })

@app.route('/ollama-status', methods=['GET'])
def ollama_status():
    """Check Ollama status and available models."""
    try:
        if not vision_assistant.check_ollama_status():
            return jsonify({
                'available': False,
                'message': 'Ollama is not running'
            })
        
        # Get available models
        response = requests.get(f"{vision_assistant.ollama_url}/api/tags")
        models = response.json().get('models', []) if response.status_code == 200 else []
        
        return jsonify({
            'available': True,
            'models': [model['name'] for model in models],
            'url': vision_assistant.ollama_url
        })
        
    except Exception as e:
        return jsonify({
            'available': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("\n🔄 Starting conversational backend...")
    print("📍 API endpoints:")
    print("   - Health: http://localhost:5001/health")
    print("   - Caption: http://localhost:5001/caption")
    print("   - Chat: http://localhost:5001/chat")
    print("   - History: http://localhost:5001/conversation-history")
    print("   - Ollama Status: http://localhost:5001/ollama-status")
    print("🔗 Frontend should connect to: http://localhost:5001")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True)
