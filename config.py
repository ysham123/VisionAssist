"""
VisionAssist Configuration
Simplified configuration for local development
"""
import os
from pathlib import Path

class Config:
    """Simple configuration class"""
    
    # Server
    HOST = os.getenv('HOST', '127.0.0.1')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'true').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production-' + os.urandom(16).hex())
    
    # CORS
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    
    # File Upload
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    
    # ML Backend
    ML_BACKEND_ENABLED = os.getenv('ML_BACKEND_ENABLED', 'true').lower() == 'true'
    MODEL_CACHE_DIR = Path('./models')
    MODEL_CACHE_DIR.mkdir(exist_ok=True)
    USE_GPU = torch.cuda.is_available() if 'torch' in locals() else False
    
    # API
    API_VERSION = '1.0.0'
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'false').lower() == 'true'
    
    @classmethod
    def validate(cls):
        """Basic validation"""
        issues = []
        if not cls.SECRET_KEY or 'dev-key' in cls.SECRET_KEY:
            issues.append('Using development SECRET_KEY')
        if '*' in cls.CORS_ORIGINS:
            issues.append('CORS allows all origins')
        return {'valid': len(issues) == 0, 'issues': issues}

# Single config instance
config = Config()