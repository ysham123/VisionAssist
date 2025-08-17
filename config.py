"""
VisionAssist Configuration Management
Centralized configuration with environment variable support
"""
import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class Config:
    """Application configuration"""
    # Server settings
    HOST: str = os.getenv('HOST', 'localhost')
    PORT: int = int(os.getenv('PORT', 5000))
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # ML Backend settings
    ML_BACKEND_ENABLED: bool = os.getenv('ML_BACKEND_ENABLED', 'True').lower() == 'true'
    MODEL_CACHE_DIR: str = os.getenv('MODEL_CACHE_DIR', './models')
    GPU_ENABLED: bool = os.getenv('GPU_ENABLED', 'True').lower() == 'true'
    
    # API settings
    API_VERSION: str = os.getenv('API_VERSION', 'v1')
    CORS_ORIGINS: str = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5000,http://127.0.0.1:49173')
    
    # Security
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    MAX_CONTENT_LENGTH: int = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    
    # Database (for future use)
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'sqlite:///visionassist.db')
    REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    @property
    def cors_origins_list(self) -> list:
        """Convert CORS origins string to list"""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(',')]

# Global config instance
config = Config()

def validate_config() -> Dict[str, Any]:
    """Validate configuration and return status"""
    issues = []
    
    if config.SECRET_KEY == 'dev-key-change-in-production':
        issues.append("SECRET_KEY is using default value - change for production")
    
    if config.DEBUG and os.getenv('ENVIRONMENT') == 'production':
        issues.append("DEBUG mode enabled in production environment")
    
    if '*' in config.CORS_ORIGINS:
        issues.append("CORS allows all origins - security risk")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'config': {
            'host': config.HOST,
            'port': config.PORT,
            'debug': config.DEBUG,
            'ml_backend_enabled': config.ML_BACKEND_ENABLED,
            'cors_origins': config.cors_origins_list
        }
    }
