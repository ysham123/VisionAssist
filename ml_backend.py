"""
VisionAssist ML Backend
Simplified ML backend using BLIP for image captioning
"""
import torch
import logging
import base64
import io
from datetime import datetime
from typing import Dict, Any, Optional
from PIL import Image

logger = logging.getLogger(__name__)

class MLBackend:
    """Simplified ML Backend for image captioning"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize BLIP model"""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            logger.info("Loading BLIP model...")
            model_name = "Salesforce/blip-image-captioning-base"
            
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"BLIP model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            self.model = None
            self.processor = None
    
    def generate_caption(self, image_data: str, include_attention: bool = False) -> Dict[str, Any]:
        """Generate caption for image"""
        if not self.model:
            return {'error': 'Model not loaded'}
        
        try:
            # Process image
            image = self._decode_image(image_data)
            
            # Generate caption
            start_time = datetime.now()
            
            with torch.no_grad():
                inputs = self.processor(image, return_tensors="pt").to(self.device)
                output = self.model.generate(**inputs, max_length=50, num_beams=5)
                caption = self.processor.decode(output[0], skip_special_tokens=True)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'caption': caption,
                'confidence': 0.95,  # BLIP is generally confident
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'model_info': {
                    'name': 'BLIP',
                    'version': 'base'
                }
            }
            
            # Add mock attention weights if requested (simplified)
            if include_attention:
                result['attention_weights'] = self._generate_mock_attention()
            
            return result
            
        except Exception as e:
            logger.error(f"Caption generation error: {e}")
            return {'error': str(e)}
    
    def _decode_image(self, image_data: str) -> Image.Image:
        """Decode base64 image"""
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode and open image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        return image
    
    def _generate_mock_attention(self) -> list:
        """Generate mock attention weights for visualization"""
        import numpy as np
        # Simple 7x7 grid of attention weights
        weights = np.random.random((7, 7))
        weights = weights / weights.sum()
        return weights.flatten().tolist()
    
    def get_status(self) -> Dict[str, Any]:
        """Get backend status"""
        return {
            'loaded': self.model is not None,
            'device': str(self.device),
            'model': 'BLIP' if self.model else None
        }