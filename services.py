"""
VisionAssist Services
Unified service layer for vision and conversation functionality
"""
import logging
import base64
import io
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from PIL import Image

from config import config

logger = logging.getLogger(__name__)

class VisionService:
    """Handles vision and ML operations"""
    
    def __init__(self):
        self.ml_backend = None
        self._init_ml_backend()
    
    def _init_ml_backend(self):
        """Initialize ML backend if enabled"""
        if config.ML_BACKEND_ENABLED:
            try:
                from ml_backend import MLBackend
                self.ml_backend = MLBackend()
                logger.info("ML backend initialized")
            except Exception as e:
                logger.warning(f"ML backend unavailable: {e}")
                self.ml_backend = None
    
    def generate_caption(self, image_data: str, include_attention: bool = False) -> Dict[str, Any]:
        """Generate caption for image"""
        try:
            # Validate image
            if not self._validate_image(image_data):
                return {'success': False, 'error': 'Invalid image data'}
            
            # Use ML backend if available
            if self.ml_backend:
                result = self.ml_backend.generate_caption(image_data, include_attention)
                return {
                    'success': True,
                    'caption': result.get('caption', 'No caption generated'),
                    'confidence': result.get('confidence', 0.5),
                    'attention_weights': result.get('attention_weights'),
                    'timestamp': datetime.now().isoformat()
                }
            
            # Fallback response
            return {
                'success': True,
                'caption': 'I can see visual content that may be helpful for navigation and understanding your environment.',
                'confidence': 0.5,
                'fallback': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Caption generation error: {e}")
            return {'success': False, 'error': str(e)}
    
    def analyze_image(self, image_data: str, include_gradcam: bool = False) -> Dict[str, Any]:
        """Analyze image with optional Grad-CAM"""
        # For simplicity, reuse caption generation with extra flags
        result = self.generate_caption(image_data, include_attention=True)
        
        if include_gradcam and self.ml_backend:
            # Add Grad-CAM if requested (simplified)
            result['gradcam'] = None  # Placeholder
        
        return result
    
    def _validate_image(self, image_data: str) -> bool:
        """Validate image data"""
        try:
            if not image_data:
                return False
            
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Try to decode and open image
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes))
            
            # Check size constraints
            if img.size[0] > 4096 or img.size[1] > 4096:
                logger.warning("Image too large")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.ml_backend:
            return {
                'name': 'BLIP Image Captioning',
                'type': 'vision-language',
                'status': 'available'
            }
        return {
            'name': 'Fallback System',
            'type': 'mock',
            'status': 'fallback'
        }

class ConversationService:
    """Handles conversation and chat functionality"""
    
    def __init__(self):
        self.sessions = {}
        self.vision_service = None
    
    def create_session(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            'id': session_id,
            'user_id': user_id,
            'created_at': datetime.now(),
            'messages': [],
            'metadata': {}
        }
        
        logger.info(f"Session created: {session_id}")
        return {
            'success': True,
            'session_id': session_id,
            'created_at': datetime.now().isoformat()
        }
    
    def process_message(self, session_id: str, message: str, 
                       image_caption: Optional[str] = None) -> Dict[str, Any]:
        """Process a chat message"""
        try:
            # Get or create session
            if session_id not in self.sessions:
                self.create_session()
                session_id = list(self.sessions.keys())[-1]
            
            session = self.sessions[session_id]
            
            # Generate response
            response = self._generate_response(message, image_caption)
            
            # Store conversation
            session['messages'].append({
                'timestamp': datetime.now().isoformat(),
                'user': message,
                'assistant': response,
                'has_visual_context': bool(image_caption)
            })
            
            return {
                'success': True,
                'response': response,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_response(self, message: str, caption: Optional[str] = None) -> str:
        """Generate AI response (simplified)"""
        message_lower = message.lower()
        
        # Context-aware responses
        if caption:
            return f"Based on what I can see ({caption}), {self._get_contextual_response(message_lower)}"
        
        # Simple keyword-based responses
        if any(word in message_lower for word in ['navigate', 'where', 'direction']):
            return "I can help you navigate. Please show me your surroundings through the camera."
        
        if any(word in message_lower for word in ['read', 'text', 'sign']):
            return "Point your camera at the text you'd like me to read."
        
        if any(word in message_lower for word in ['describe', 'what', 'see']):
            return "I'm ready to describe what I see. Make sure your camera is active."
        
        return "I'm here to help with visual assistance. What would you like to know?"
    
    def _get_contextual_response(self, message: str) -> str:
        """Get contextual response based on visual input"""
        if 'color' in message:
            return "I can identify colors in the image."
        if 'count' in message or 'how many' in message:
            return "let me count the objects for you."
        if 'safe' in message or 'danger' in message:
            return "I'll help you identify any potential hazards."
        return "I can provide more details about what I observe."
    
    def get_history(self, session_id: str, limit: int = 50) -> Dict[str, Any]:
        """Get conversation history"""
        if session_id not in self.sessions:
            return {'success': False, 'error': 'Session not found'}
        
        messages = self.sessions[session_id]['messages'][-limit:]
        return {
            'success': True,
            'messages': messages,
            'total': len(messages)
        }