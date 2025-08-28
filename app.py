"""
VisionAssist Main Application
Streamlined Flask application for local development
"""
import logging
import os
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from werkzeug.exceptions import BadRequest

from config import config
from services import VisionService, ConversationService

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Security headers (simplified for local dev)
if not config.DEBUG:
    Talisman(
        app,
        force_https=False,
        content_security_policy={
            'default-src': "'self'",
            'img-src': ["'self'", 'data:', 'blob:'],
            'script-src': ["'self'", "'unsafe-inline'"],
            'style-src': ["'self'", "'unsafe-inline'", 'https://fonts.googleapis.com'],
            'font-src': ["'self'", 'https://fonts.gstatic.com', 'data:']
        }
    )

# Rate limiting (optional for local dev)
if config.RATE_LIMIT_ENABLED:
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["1000 per hour"]
    )

# Configure CORS
CORS(app, origins=config.CORS_ORIGINS)

# App configuration
app.config.update(
    SECRET_KEY=config.SECRET_KEY,
    MAX_CONTENT_LENGTH=config.MAX_CONTENT_LENGTH
)

# Initialize services
vision_service = VisionService()
conversation_service = ConversationService()

# Request logging
@app.before_request
def log_request():
    """Log requests for debugging"""
    if request.path not in ['/static/', '/health', '/favicon.ico']:
        logger.debug(f"Request: {request.method} {request.path}")
    g.start_time = datetime.now()

@app.after_request
def log_response(response):
    """Log response time"""
    if hasattr(g, 'start_time'):
        duration = (datetime.now() - g.start_time).total_seconds() * 1000
        logger.debug(f"Response time: {duration:.2f}ms")
    return response

# Error handlers
@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({'success': False, 'error': str(e)}), 400

@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({'success': False, 'error': 'Not found'}), 404

@app.errorhandler(500)
def handle_server_error(e):
    logger.error(f"Server error: {e}", exc_info=True)
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# Static files
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

# Vision endpoints
@app.route('/api/v1/vision/caption', methods=['POST'])
def caption():
    """Generate caption for an image"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            raise BadRequest('Missing image data')
        
        result = vision_service.generate_caption(
            data['image'],
            include_attention=data.get('include_attention', False)
        )
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Caption error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/vision/analyze', methods=['POST'])
def analyze():
    """Analyze image with additional details"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            raise BadRequest('Missing image data')
        
        result = vision_service.analyze_image(
            data['image'],
            include_gradcam=data.get('include_gradcam', False)
        )
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Conversation endpoints
@app.route('/api/v1/conversation/sessions', methods=['POST'])
def create_session():
    """Create a new conversation session"""
    try:
        # Handle both JSON and empty requests
        data = {}
        if request.is_json:
            data = request.get_json() or {}
        elif request.content_length and request.content_length > 0:
            # Handle form data or other content types
            data = request.form.to_dict()
        
        result = conversation_service.create_session(data.get('user_id'))
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/conversation/chat', methods=['POST'])
def chat():
    """Process a chat message"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            raise BadRequest('Missing message')
        
        result = conversation_service.process_message(
            data.get('session_id', 'default'),
            data['message'],
            data.get('image_caption')
        )
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/conversation/sessions/<session_id>/history', methods=['GET'])
def get_history(session_id):
    """Get conversation history"""
    try:
        limit = request.args.get('limit', 50, type=int)
        result = conversation_service.get_history(session_id, limit)
        return jsonify(result), 200 if result['success'] else 404
    except Exception as e:
        logger.error(f"History error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/v1/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'success': True,
        'model_info': vision_service.get_model_info(),
        'backend_status': 'available' if vision_service.ml_backend else 'fallback'
    })

if __name__ == '__main__':
    logger.info(f"🚀 Starting VisionAssist on http://{config.HOST}:{config.PORT}")
    logger.info(f"📝 Debug mode: {config.DEBUG}")
    logger.info(f"🤖 ML Backend: {config.ML_BACKEND_ENABLED}")
    
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True
    )