"""
Vision Service
Handles all computer vision and ML inference operations
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import base64
from PIL import Image
import io

from config import config

logger = logging.getLogger(__name__)

class VisionService:
    """Service for handling vision-related operations"""
    
    def __init__(self):
        self.ml_backend = None
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the vision service asynchronously"""
        if self._initialized:
            return True
            
        try:
            if config.ML_BACKEND_ENABLED:
                # Import ML backend only when needed
                from ml_backend import get_ml_backend, initialize_ml_backend
                
                # Initialize in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(None, initialize_ml_backend)
                
                if success:
                    self.ml_backend = get_ml_backend()
                    logger.info("Vision service initialized with ML backend")
                else:
                    logger.warning("ML backend failed to initialize, using fallback")
                    
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vision service: {e}")
            self._initialized = True  # Mark as initialized even if ML failed
            return False
    
    async def generate_caption(self, image_data: str, **kwargs) -> Dict[str, Any]:
        """
        Generate caption for image
        
        Args:
            image_data: Base64 encoded image
            **kwargs: Additional options (include_attention, include_gradcam, etc.)
            
        Returns:
            Dictionary with caption and metadata
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Validate image data
            validation_result = self._validate_image(image_data)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'error_code': 'INVALID_IMAGE'
                }
            
            # Use ML backend if available
            if self.ml_backend:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    self.ml_backend.generate_caption,
                    image_data,
                    kwargs.get('include_attention', False),
                    kwargs.get('include_gradcam', False)
                )
                
                return {
                    'success': True,
                    'caption': result.get('caption', 'Unable to generate caption'),
                    'confidence': result.get('confidence', 0.0),
                    'processing_time': result.get('processing_time', 0.0),
                    'attention_weights': result.get('attention_weights'),
                    'gradcam': result.get('gradcam'),
                    'timestamp': datetime.now().isoformat(),
                    'model_info': result.get('model_info', {})
                }
            else:
                # Fallback response
                return self._generate_fallback_caption()
                
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return {
                'success': False,
                'error': 'Internal server error during caption generation',
                'error_code': 'CAPTION_ERROR'
            }
    
    def _validate_image(self, image_data: str) -> Dict[str, Any]:
        """Validate base64 image data"""
        try:
            if not image_data:
                return {'valid': False, 'error': 'Empty image data'}
                
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Validate image format
            image = Image.open(io.BytesIO(image_bytes))
            
            # Check image size constraints
            max_size = 10 * 1024 * 1024  # 10MB
            if len(image_bytes) > max_size:
                return {'valid': False, 'error': 'Image too large (max 10MB)'}
                
            # Check image dimensions
            if image.size[0] > 4096 or image.size[1] > 4096:
                return {'valid': False, 'error': 'Image dimensions too large (max 4096x4096)'}
            
            return {'valid': True, 'format': image.format, 'size': image.size}
            
        except Exception as e:
            return {'valid': False, 'error': f'Invalid image format: {str(e)}'}
    
    def _generate_fallback_caption(self) -> Dict[str, Any]:
        """Generate fallback caption when ML backend unavailable"""
        fallback_captions = [
            "I can see an image that may contain objects, people, or scenes that could be helpful for navigation.",
            "There appears to be visual content in this image that I'm analyzing for accessibility purposes.",
            "This image contains visual elements that may be relevant for understanding your environment."
        ]
        
        import random
        caption = random.choice(fallback_captions)
        
        return {
            'success': True,
            'caption': caption,
            'confidence': 0.5,
            'processing_time': 0.1,
            'timestamp': datetime.now().isoformat(),
            'fallback': True,
            'model_info': {'type': 'fallback', 'version': '1.0'}
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            'service': 'vision',
            'status': 'healthy' if self._initialized else 'initializing',
            'ml_backend_available': self.ml_backend is not None,
            'config': {
                'ml_enabled': config.ML_BACKEND_ENABLED,
                'gpu_enabled': config.GPU_ENABLED
            },
            'timestamp': datetime.now().isoformat()
        }
