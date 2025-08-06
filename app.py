"""
VisionAssist Main Application
Refactored Flask app with proper service layer architecture
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
import threading

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from config import config, validate_config
from services import VisionService, ConversationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Configure CORS with proper origins
CORS(app, 
     origins=config.cors_origins_list,
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

# Initialize services
vision_service = VisionService()
conversation_service = ConversationService()

def initialize_services_sync():
    """Initialize services before handling requests"""
    logger.info("🚀 Initializing VisionAssist services...")
    
    # Validate configuration
    config_status = validate_config()
    if not config_status['valid']:
        logger.warning("Configuration issues detected:")
        for issue in config_status['issues']:
            logger.warning(f"  - {issue}")
    
    # Initialize vision service (simplified for Flask compatibility)
    try:
        # For now, we'll initialize synchronously
        # In production, use proper async framework like FastAPI
        logger.info("✅ Vision service will initialize on first request")
    except Exception as e:
        logger.warning(f"⚠️ Vision service setup warning: {e}")
    
    logger.info("🌟 VisionAssist services ready!")

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Handle file size limit exceeded"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.',
        'error_code': 'FILE_TOO_LARGE'
    }), 413

@app.errorhandler(500)
def handle_internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'error_code': 'INTERNAL_ERROR'
    }), 500

@app.errorhandler(404)
def handle_not_found(e):
    """Handle not found errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'error_code': 'NOT_FOUND'
    }), 404

# Static file serving
@app.route('/')
def index():
    """Serve the main application page"""
    return send_from_directory('.', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

# Health and status endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check"""
    vision_health = vision_service.get_health_status()
    conversation_health = conversation_service.get_health_status()
    
    overall_status = 'healthy'
    if vision_health['status'] != 'healthy' or conversation_health['status'] != 'healthy':
        overall_status = 'degraded'
    
    return jsonify({
        'status': overall_status,
        'timestamp': datetime.now().isoformat(),
        'services': {
            'vision': vision_health,
            'conversation': conversation_health
        },
        'config': validate_config()
    })

@app.route('/api/v1/status', methods=['GET'])
def api_status():
    """API status endpoint"""
    return jsonify({
        'api_version': config.API_VERSION,
        'status': 'operational',
        'timestamp': datetime.now().isoformat(),
        'endpoints': [
            '/api/v1/vision/caption',
            '/api/v1/vision/analyze',
            '/api/v1/conversation/sessions',
            '/api/v1/conversation/chat',
            '/health',
            '/api/v1/status'
        ]
    })

# Vision API endpoints
@app.route('/api/v1/vision/caption', methods=['POST'])
def caption():
    """Generate caption for an image"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing image data in request body',
                'error_code': 'MISSING_IMAGE_DATA'
            }), 400
        
        # For Flask compatibility, we'll use synchronous calls
        # In production, consider using FastAPI for proper async support
        try:
            from ml_backend import get_ml_backend
            ml_backend = get_ml_backend()
            if ml_backend:
                result = ml_backend.generate_caption(
                    data['image'],
                    include_attention=data.get('include_attention', False),
                    include_gradcam=data.get('include_gradcam', False)
                )
                result['success'] = True
            else:
                result = {'success': False, 'error': 'ML backend not available', 'error_code': 'ML_UNAVAILABLE'}
        except Exception as e:
            result = {'success': False, 'error': str(e), 'error_code': 'ML_ERROR'}
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in caption endpoint: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to process caption request',
            'error_code': 'CAPTION_PROCESSING_ERROR'
        }), 500

@app.route('/api/v1/vision/analyze', methods=['POST'])
def analyze_image():
    """Advanced image analysis with attention and Grad-CAM"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing image data in request body',
                'error_code': 'MISSING_IMAGE_DATA'
            }), 400
        
        # For Flask compatibility, we'll use synchronous calls
        try:
            from ml_backend import get_ml_backend
            ml_backend = get_ml_backend()
            if ml_backend:
                result = ml_backend.generate_caption(
                    data['image'],
                    include_attention=True,
                    include_gradcam=data.get('include_gradcam', False)
                )
                result['success'] = True
            else:
                result = {'success': False, 'error': 'ML backend not available', 'error_code': 'ML_UNAVAILABLE'}
        except Exception as e:
            result = {'success': False, 'error': str(e), 'error_code': 'ML_ERROR'}
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to process analysis request',
            'error_code': 'ANALYSIS_PROCESSING_ERROR'
        }), 500

# Conversation API endpoints
@app.route('/api/v1/conversation/sessions', methods=['POST'])
def create_session():
    """Create a new conversation session"""
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id')
        
        result = conversation_service.create_session(user_id)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to create conversation session',
            'error_code': 'SESSION_CREATION_ERROR'
        }), 500

@app.route('/api/v1/conversation/chat', methods=['POST'])
def chat():
    """Process a chat message with optional visual context"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing message in request body',
                'error_code': 'MISSING_MESSAGE'
            }), 400
        
        session_id = data.get('session_id', 'default')
        message = data['message']
        image_caption = data.get('image_caption')
        
        result = conversation_service.process_message(
            session_id, message, image_caption
        )
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to process chat message',
            'error_code': 'CHAT_PROCESSING_ERROR'
        }), 500

@app.route('/api/v1/conversation/sessions/<session_id>/history', methods=['GET'])
def get_session_history(session_id: str):
    """Get conversation history for a session"""
    try:
        limit = request.args.get('limit', 50, type=int)
        result = conversation_service.get_session_history(session_id, limit)
        
        status_code = 200 if result['success'] else 404
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error getting session history: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve session history',
            'error_code': 'HISTORY_RETRIEVAL_ERROR'
        }), 500

@app.route('/api/v1/model/info', methods=['GET'])
def model_info():
    """Get information about the ML model"""
    try:
        from ml_backend import get_ml_backend
        ml_backend = get_ml_backend()
        
        if ml_backend:
            return jsonify({
                'success': True,
                'model_info': {
                    'name': 'BLIP Image Captioning',
                    'version': 'base',
                    'provider': 'Salesforce',
                    'type': 'vision-language',
                    'capabilities': ['image_captioning', 'visual_question_answering'],
                    'input_formats': ['jpg', 'png', 'webp'],
                    'max_image_size': '10MB',
                    'processing_time': '~100ms CPU, ~20ms GPU'
                },
                'backend_status': 'available',
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'success': True,
                'model_info': {
                    'name': 'Fallback Vision System',
                    'version': '1.0',
                    'type': 'mock',
                    'capabilities': ['basic_captioning'],
                    'note': 'ML backend not available'
                },
                'backend_status': 'fallback',
                'timestamp': datetime.now().isoformat()
            }), 200
            
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve model information',
            'error_code': 'MODEL_INFO_ERROR'
        }), 500

# Background tasks (simplified for Flask)
def cleanup_task():
    """Background task for cleanup operations"""
    import time
    import threading
    
    def run_cleanup():
        while True:
            try:
                # Clean up expired sessions
                cleaned = conversation_service.cleanup_expired_sessions()
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} expired sessions")
                
                # Wait 1 hour before next cleanup
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    # Start cleanup in background thread
    cleanup_thread = threading.Thread(target=run_cleanup, daemon=True)
    cleanup_thread.start()

if __name__ == '__main__':
    logger.info("🌟 Starting VisionAssist Application Server...")
    logger.info(f"📡 Server will be available at: http://{config.HOST}:{config.PORT}")
    logger.info(f"🔧 Debug mode: {config.DEBUG}")
    logger.info(f"🤖 ML Backend enabled: {config.ML_BACKEND_ENABLED}")
    
    # Initialize services
    initialize_services_sync()
    
    # Start background cleanup task
    cleanup_task()
    
    # Run Flask app
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True
    )
