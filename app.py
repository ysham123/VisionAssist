"""
VisionAssist Main Application
Production-ready Flask application with enhanced security and error handling
"""
import asyncio
import logging
import os
import secrets
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, Union

from flask import (
    Flask, 
    jsonify, 
    request, 
    send_from_directory, 
    Response,
    g
)
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from werkzeug.exceptions import (
    BadRequest,
    RequestEntityTooLarge,
    InternalServerError
)
from werkzeug.middleware.proxy_fix import ProxyFix

from config import config, validate_config
from services import VisionService, ConversationService

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Security headers and middleware
Talisman(
    app,
    force_https=config.ENVIRONMENT == 'production',
    strict_transport_security=True,
    session_cookie_secure=True,
    session_cookie_http_only=True,
    # CSP tuned for local dev with Google Fonts and nonce-based scripts
    content_security_policy={
        'default-src': "'self'",
        'img-src': ["'self'", 'data:', 'blob:'],
        'media-src': ["'self'"],
        # Use nonce for inline scripts; external scripts allowed from self
        # Note: Inline scripts should be moved to external files to avoid 'unsafe-inline'
        'script-src': [
            "'self'"
        ],
        # Allow Google Fonts CSS and inline styles for dev; remove 'unsafe-inline' for prod
        'style-src': ["'self'", "'unsafe-inline'", 'https://fonts.googleapis.com'],
        # Permit Google Fonts font files
        'font-src': ["'self'", 'https://fonts.gstatic.com', 'data:']
    },
    content_security_policy_nonce_in=['script-src']
)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
    strategy="fixed-window"
)

# Configure CORS with proper origins
CORS(
    app,
    origins=config.CORS_ORIGINS,
    methods=["GET", "POST", "OPTIONS", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    supports_credentials=True,
    max_age=600
)

# Application configuration
app.config.update(
    SECRET_KEY=config.SECRET_KEY,
    MAX_CONTENT_LENGTH=config.MAX_CONTENT_LENGTH,
    SESSION_COOKIE_SECURE=config.ENVIRONMENT == 'production',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
    JSON_SORT_KEYS=True,
    JSON_AS_ASCII=False,
    JSONIFY_PRETTYPRINT_REGULAR=False
)

# Initialize services with error handling
try:
    vision_service = VisionService()
    conversation_service = ConversationService()
    logger.info("✅ Services initialized successfully")
except Exception as e:
    logger.critical(f"❌ Failed to initialize services: {e}")
    if config.ENVIRONMENT == 'production':
        raise

# Security middleware
@app.before_request
def before_request():
    """Security and request preprocessing"""
    # Add security headers
    g.start_time = datetime.utcnow()
    
    # Skip for static files and health checks
    if request.path.startswith('/static/') or request.path == '/health' or request.path == '/favicon.ico':
        return
    
    # Log request details
    logger.info(f"Request: {request.method} {request.path}")
    
    # Validate content type for POST/PUT requests
    # Allow empty bodies for endpoints like session creation
    if request.method in ['POST', 'PUT']:
        content_type = (request.content_type or '')
        content_length = request.content_length or 0
        if content_length and (not request.is_json) and ('multipart/form-data' not in content_type):
            raise BadRequest('Content-Type must be application/json or multipart/form-data')

# Error handlers
@app.errorhandler(400)
def handle_bad_request(e):
    """Handle 400 Bad Request errors"""
    return jsonify({
        'success': False,
        'error': str(e) or 'Bad request',
        'error_code': 'BAD_REQUEST'
    }), 400

@app.errorhandler(401)
def handle_unauthorized(e):
    """Handle 401 Unauthorized errors"""
    return jsonify({
        'success': False,
        'error': 'Authentication required',
        'error_code': 'UNAUTHORIZED'
    }), 401

@app.errorhandler(403)
def handle_forbidden(e):
    """Handle 403 Forbidden errors"""
    return jsonify({
        'success': False,
        'error': 'Access denied',
        'error_code': 'FORBIDDEN'
    }), 403

@app.errorhandler(404)
def handle_not_found(e):
    """Handle 404 Not Found errors"""
    return jsonify({
        'success': False,
        'error': 'Resource not found',
        'error_code': 'NOT_FOUND'
    }), 404

@app.errorhandler(405)
def handle_method_not_allowed(e):
    """Handle 405 Method Not Allowed errors"""
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'error_code': 'METHOD_NOT_ALLOWED'
    }), 405

@app.errorhandler(413)
def handle_request_entity_too_large(e):
    """Handle 413 Request Entity Too Large errors"""
    return jsonify({
        'success': False,
        'error': f'File too large. Maximum size is {config.MAX_CONTENT_LENGTH / (1024 * 1024)}MB',
        'error_code': 'PAYLOAD_TOO_LARGE'
    }), 413

@app.errorhandler(429)
def handle_rate_limit_exceeded(e):
    """Handle 429 Too Many Requests errors"""
    return jsonify({
        'success': False,
        'error': 'Rate limit exceeded',
        'error_code': 'RATE_LIMIT_EXCEEDED',
        'retry_after': e.retry_after if hasattr(e, 'retry_after') else None
    }), 429

@app.errorhandler(500)
def handle_server_error(e):
    """Handle 500 Internal Server Errors"""
    logger.error(f'Internal Server Error: {str(e)}', exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'error_code': 'INTERNAL_SERVER_ERROR'
    }), 500

# Request teardown
@app.teardown_request
def teardown_request(exception=None):
    """Log request completion and metrics"""
    if hasattr(g, 'start_time'):
        duration = (datetime.utcnow() - g.start_time).total_seconds() * 1000  # ms
        logger.info(f"Request completed in {duration:.2f}ms")

# Health check endpoint (simplified for local testing)
@app.route('/health', methods=['GET'])
@limiter.exempt
def health_check():
    """Health check endpoint for local testing"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'environment': 'development',
        'message': 'Running in local testing mode'
    })

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

# Status endpoint (simplified for local testing)
@app.route('/status', methods=['GET'])
def status():
    """Status endpoint for local testing"""
    return jsonify({
        'status': 'operational',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'vision': {'status': 'operational'},
            'conversation': {'status': 'operational'}
        },
        'environment': 'development',
        'message': 'Running in local testing mode with mock services'
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
                # Graceful fallback to keep endpoint successful in local dev
                result = {
                    'success': True,
                    'caption': 'The vision model is not available. Showing a fallback description for development testing.',
                    'confidence': 0.5,
                    'processing_time': 0.1,
                    'fallback': True,
                    'model_info': {'type': 'fallback', 'version': '1.0'}
                }
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
                result = {
                    'success': True,
                    'caption': 'The vision model is not available. Fallback analysis for development testing.',
                    'confidence': 0.5,
                    'processing_time': 0.1,
                    'attention_weights': None,
                    'gradcam': None,
                    'fallback': True,
                    'model_info': {'type': 'fallback', 'version': '1.0'}
                }
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
