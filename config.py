"""
VisionAssist Configuration Management
Centralized configuration with environment variable support and validation
"""
import os
import secrets
from typing import Dict, Any, List, Optional
from pydantic import HttpUrl, validator
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Server settings
    HOST: str = '0.0.0.0'
    PORT: int = 5000
    DEBUG: bool = True
    ENVIRONMENT: str = 'development'
    
    # Security (simplified for local testing)
    SECRET_KEY: str = 'dev-key-for-testing-only'
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours for testing
    CORS_ORIGINS: List[str] = ['*']  # Allow all origins for local testing
    
    # File uploads
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS: List[str] = ['png', 'jpg', 'jpeg', 'gif']
    
    # ML Backend (optimized for local testing)
    ML_BACKEND_ENABLED: bool = True
    MODEL_CACHE_DIR: str = './models'
    GPU_ENABLED: bool = False
    USE_MOCK_RESPONSES: bool = True  # Enable mock responses for testing
    
    # Database (not required for local testing)
    DATABASE_URL: str = 'sqlite:///./test.db'
    
    # API
    API_PREFIX: str = '/api/v1'
    API_VERSION: str = '1.0.0'
    
    # Logging
    LOG_LEVEL: str = 'INFO'
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = True
    
    @validator('SECRET_KEY', pre=True)
    def validate_secret_key(cls, v: str) -> str:
        if not v:
            if os.getenv('ENVIRONMENT') == 'production':
                raise ValueError('SECRET_KEY must be set in production')
            return 'dev-key-for-testing-only-1234567890123456'  # 32+ chars for local dev
        return v
    
    @validator('CORS_ORIGINS', pre=True)
    def validate_cors_origins(cls, v: str) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v or []
    
    @validator('MODEL_CACHE_DIR')
    def validate_model_cache_dir(cls, v: str) -> str:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(path, os.W_OK):
            raise ValueError(f'Cannot write to model cache directory: {v}')
        return str(path.absolute())

# Initialize settings
config = Settings()

def validate_config() -> Dict[str, Any]:
    """Validate configuration and return status"""
    issues = []
    
    if config.DEBUG and config.ENVIRONMENT == 'production':
        issues.append('Debug mode enabled in production environment')
    
    if not config.SECRET_KEY or config.SECRET_KEY == 'dev-key-change-in-production':
        issues.append('Using default or empty SECRET_KEY in production')
    
    if '*' in config.CORS_ORIGINS:
        issues.append('CORS allows all origins - security risk')
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'environment': config.ENVIRONMENT,
        'debug': config.DEBUG,
        'cors_origins': config.CORS_ORIGINS
    }
    
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
