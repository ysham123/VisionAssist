from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
import traceback

# Import our ML backend
try:
    from ml_backend import get_ml_backend, initialize_ml_backend
    ML_BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML backend not available: {e}")
    ML_BACKEND_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

# Store conversation history (in production, use Redis or database)
conversation_history = {}

# Initialize ML backend on startup
print("\n🚀 Initializing VisionAssist ML Backend...")
if ML_BACKEND_AVAILABLE:
    ML_BACKEND_LOADED = initialize_ml_backend()
    if ML_BACKEND_LOADED:
        print("✅ ML Backend loaded successfully!")
        print("🔬 Architecture: MobileNet + LSTM with Attention")
        print("🎯 Target: Accessibility for visually impaired users")
        print("📊 Features: Grad-CAM explainability, attention visualization")
    else:
        print("❌ Failed to load ML Backend - using fallback responses")
else:
    ML_BACKEND_LOADED = False
    print("❌ ML Backend not available - using fallback responses")

# Fallback responses for when ML backend is not available
FALLBACK_CAPTIONS = [
    "I can see an image with various objects and details that could be helpful for navigation.",
    "There appears to be a scene with people, objects, or text that I'm analyzing for you.",
    "I'm processing visual information that includes important elements for accessibility.",
    "The image contains details about the environment that could assist with understanding your surroundings.",
    "I can detect visual elements that may be relevant for navigation and object recognition."
]

FALLBACK_RESPONSES = [
    "I'm here to help describe what I see through your camera. What would you like to know about your surroundings?",
    "I can analyze images and help with navigation. Please show me what you'd like me to describe.",
    "I'm ready to assist with visual understanding. Is there something specific in your environment you'd like me to identify?",
    "I can help describe objects, text, people, and scenes to assist with accessibility. What can I help you with?",
    "I'm designed to provide detailed descriptions for visually impaired users. How can I assist you today?"
]

def generate_caption_with_ml(image_data: str, include_attention: bool = False, include_gradcam: bool = False) -> Dict:
    """Generate caption using ML backend with advanced features"""
    logger.info(f"generate_caption_with_ml called with ML_BACKEND_LOADED={ML_BACKEND_LOADED}")
    
    if not ML_BACKEND_LOADED:
        logger.warning("ML Backend not loaded, using fallback")
        import random
        return {
            'caption': random.choice(FALLBACK_CAPTIONS),
            'source': 'fallback_no_backend',
            'timestamp': datetime.now().isoformat()
        }
    
    # Retry logic for transient errors
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries + 1}: Getting ML backend instance...")
            ml_backend = get_ml_backend()
            logger.info("ML backend retrieved, generating caption...")
            
            result = ml_backend.generate_caption(
                image_data=image_data,
                include_attention=include_attention,
                include_gradcam=include_gradcam
            )
            
            # Validate result quality
            caption = result.get('caption', '').strip()
            if not caption or len(caption) < 3:
                logger.warning(f"Low quality caption received: '{caption}', retrying...")
                if attempt < max_retries:
                    continue
                else:
                    logger.error("All retries exhausted, caption quality still poor")
                    raise Exception("Generated caption is too short or empty")
            
            result['source'] = 'ml_model'
            result['attempt'] = attempt + 1
            logger.info(f"ML caption generated successfully on attempt {attempt + 1}: {caption[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in 1 second... ({max_retries - attempt} attempts left)")
                import time
                time.sleep(1)
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Final error: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                
                # Only use fallback after all retries exhausted
                import random
                return {
                    'caption': random.choice(FALLBACK_CAPTIONS),
                    'source': 'fallback_after_retries',
                    'error': str(e),
                    'attempts': max_retries + 1,
                    'timestamp': datetime.now().isoformat()
                }

def generate_contextual_response(user_message: str, image_caption: str = None, conversation_context: List[Dict] = None) -> str:
    """Generate contextual response for conversation with intelligent question processing"""
    try:
        if not image_caption:
            # No image context
            if "help" in user_message.lower():
                return "I'm here to help you understand your visual surroundings. Please show me an image through your camera and I can describe what I see to assist with navigation, object identification, or reading text."
            else:
                import random
                return random.choice(FALLBACK_RESPONSES)
        
        # Process specific questions about the image
        user_lower = user_message.lower()
        caption_lower = image_caption.lower()
        
        # Count-related questions
        if any(word in user_lower for word in ['how many', 'count', 'number of']):
            if 'window' in user_lower:
                if 'window' in caption_lower:
                    return f"Looking at the image, I can see windows in the scene. Based on what's visible: {image_caption}. I can see some windows, but for an exact count, the image would need to show them more clearly."
                else:
                    return f"I'm looking for windows in the image. Currently I see: {image_caption}. I don't clearly see any windows that I can count in this view."
            elif 'person' in user_lower or 'people' in user_lower:
                if any(word in caption_lower for word in ['person', 'people', 'man', 'woman', 'individual']):
                    return f"Based on the image, I can see people present. From what I observe: {image_caption}. I can identify people in the scene."
                else:
                    return f"Looking for people in the image. I see: {image_caption}. I don't clearly identify people in this particular view."
            else:
                return f"You're asking about counting something in the image. I can see: {image_caption}. Could you be more specific about what you'd like me to count?"
        
        # Color questions
        elif 'color' in user_lower or 'what color' in user_lower:
            if 'shirt' in user_lower or 'top' in user_lower or 'clothing' in user_lower:
                if any(color in caption_lower for color in ['white', 'black', 'red', 'blue', 'green', 'yellow', 'gray', 'grey']):
                    colors_mentioned = [color for color in ['white', 'black', 'red', 'blue', 'green', 'yellow', 'gray', 'grey'] if color in caption_lower]
                    return f"Looking at the clothing in the image, I can see {', '.join(colors_mentioned)} colors. The image shows: {image_caption}."
                else:
                    return f"I can see clothing in the image: {image_caption}. The specific colors aren't clearly described in my analysis, but I can see what appears to be clothing items."
            else:
                return f"You're asking about colors in the image. I can see: {image_caption}. What specific item's color would you like me to focus on?"
        
        # Clothing/appearance questions
        elif any(word in user_lower for word in ['wearing', 'shirt', 'top', 'clothes', 'clothing', 'tank top']):
            if any(word in caption_lower for word in ['wearing', 'shirt', 'top', 'tank', 'clothing']):
                return f"Looking at what's being worn in the image: {image_caption}. I can see clothing items as described."
            else:
                return f"I'm looking at the clothing in the image. I can see: {image_caption}. Could you ask about a specific clothing item?"
        
        # General description requests
        elif any(phrase in user_lower for phrase in ['what do you see', 'describe', 'tell me about']):
            return f"I can see: {image_caption}. This should help you understand your current surroundings."
        
        # Location questions
        elif any(phrase in user_lower for phrase in ['where am i', 'location', 'where is this']):
            return f"Based on what I can see: {image_caption}. This might help you identify your location or surroundings."
        
        # Text reading requests
        elif 'read' in user_lower or 'text' in user_lower:
            if 'text' in caption_lower or 'sign' in caption_lower:
                return f"I can see text or signage in the image: {image_caption}. Let me know if you need me to focus on specific text elements."
            else:
                return f"I'm looking for text in the image. Currently I see: {image_caption}. If there's specific text you need read, please let me know."
        
        # Default: Try to answer the question contextually
        else:
            return f"Based on your question '{user_message}' and what I can see in the image: {image_caption}. How can I help you with more specific information about what you're looking for?"
                
    except Exception as e:
        logger.error(f"Error generating contextual response: {e}")
        return "I'm here to help with visual assistance. Please let me know what you'd like me to describe or identify."

@app.route('/')
def index():
    """Serve the main application page"""
    return send_from_directory('.', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ml_backend_loaded': ML_BACKEND_LOADED,
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0-ml'
    }), 200

@app.route('/api/v1/vision/caption', methods=['POST'])
def caption():
    """Generate caption for an image using ML backend"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing image data',
                'error_code': 'MISSING_DATA'
            }), 400
        
        # Get optional parameters
        include_attention = data.get('include_attention', False)
        include_gradcam = data.get('include_gradcam', False)
        
        # Generate caption using ML backend
        result = generate_caption_with_ml(
            image_data=data['image'],
            include_attention=include_attention,
            include_gradcam=include_gradcam
        )
        
        return jsonify({
            'success': True,
            **result
        }), 200
        
    except Exception as e:
        logger.error(f"Error in caption endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'SERVER_ERROR'
        }), 500

@app.route('/api/v1/vision/analyze', methods=['POST'])
def analyze_image():
    """Advanced image analysis with ML features"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing image data',
                'error_code': 'MISSING_DATA'
            }), 400
        
        # Generate comprehensive analysis
        result = generate_caption_with_ml(
            image_data=data['image'],
            include_attention=True,
            include_gradcam=data.get('include_gradcam', False)
        )
        
        # Add quality evaluation if ML backend is available
        if ML_BACKEND_LOADED:
            try:
                ml_backend = get_ml_backend()
                quality_metrics = ml_backend.evaluate_caption_quality(result['caption'])
                result['quality_metrics'] = quality_metrics
            except Exception as e:
                logger.warning(f"Could not compute quality metrics: {e}")
        
        return jsonify({
            'success': True,
            'analysis': result
        }), 200
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'SERVER_ERROR'
        }), 500

@app.route('/api/v1/conversation/session', methods=['POST'])
def create_session():
    """Create a new conversation session"""
    session_id = f"session_{int(time.time() * 1000)}"
    conversation_history[session_id] = []
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/v1/conversation/chat', methods=['POST'])
def chat():
    """Process a chat message with ML-powered context awareness"""
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
        image_caption = None
        if 'image' in data and data['image']:
            try:
                caption_result = generate_caption_with_ml(data['image'])
                image_caption = caption_result['caption']
                
                # Add the caption to conversation history
                conversation_history[session_id].append({
                    'role': 'system',
                    'content': f"[Image description: {image_caption}]",
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error processing image in chat: {e}")
        
        # Add user message to history
        conversation_history[session_id].append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Limit history size to prevent context overflow
        if len(conversation_history[session_id]) > 20:
            conversation_history[session_id] = conversation_history[session_id][-20:]
        
        # Generate contextual response
        response = generate_contextual_response(
            user_message=user_message,
            image_caption=image_caption,
            conversation_context=conversation_history[session_id]
        )
        
        # Add assistant response to history
        conversation_history[session_id].append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'response': response,
            'image_processed': image_caption is not None,
            'ml_backend_used': ML_BACKEND_LOADED,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'SERVER_ERROR'
        }), 500

@app.route('/api/v1/model/info', methods=['GET'])
def model_info():
    """Get information about the ML model"""
    try:
        info = {
            'ml_backend_loaded': ML_BACKEND_LOADED,
            'architecture': 'MobileNet + LSTM with Attention',
            'features': [
                'Image captioning with attention mechanism',
                'Grad-CAM explainability',
                'Accessibility-focused descriptions',
                'Real-time inference'
            ],
            'target_use_case': 'Accessibility for visually impaired users',
            'model_components': {
                'feature_extractor': 'MobileNetV2 (pre-trained)',
                'decoder': 'LSTM with Bahdanau Attention',
                'explainability': 'Grad-CAM visualization'
            }
        }
        
        if ML_BACKEND_LOADED:
            try:
                ml_backend = get_ml_backend()
                info['device'] = str(ml_backend.device)
                info['vocab_size'] = len(ml_backend.tokenizer)
            except Exception as e:
                info['backend_error'] = str(e)
        
        return jsonify({
            'success': True,
            'model_info': info,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'SERVER_ERROR'
        }), 500

if __name__ == '__main__':
    print("\n🌟 VisionAssist ML Server Starting...")
    print(f"📡 Server will be available at: http://localhost:5000")
    print(f"🔬 ML Backend Status: {'✅ Loaded' if ML_BACKEND_LOADED else '❌ Fallback Mode'}")
    print("🎯 Ready to assist visually impaired users with AI-powered descriptions\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
