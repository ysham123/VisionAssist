"""
Configuration management for VisionAssist project.
Centralizes all configuration parameters and paths.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

class Config:
    """Configuration class for VisionAssist project."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration with default values and optional config file."""
        
        # Base paths
        self.PROJECT_ROOT = Path(__file__).parent.absolute()
        self.DATA_DIR = self.PROJECT_ROOT / "processed_data"
        self.COCO_DIR = self.PROJECT_ROOT / "COCO"
        self.ANNOTATIONS_DIR = self.PROJECT_ROOT / "annotations"
        
        # COCO dataset paths
        self.COCO_IMAGE_DIR = self.COCO_DIR / "train2014" / "train2014"
        self.COCO_ANNOTATION_FILE = self.COCO_DIR / "annotations_trainval2014" / "annotations" / "captions_train2014.json"
        
        # VizWiz dataset paths (alternative)
        self.VIZWIZ_IMAGE_DIR = Path("../Data/train/")
        self.VIZWIZ_ANNOTATION_FILE = self.ANNOTATIONS_DIR / "train.json"
        
        # Model configuration
        self.MODEL_CONFIG = {
            'vocab_size': None,  # Will be set dynamically
            'embed_dim': 512,
            'hidden_dim': 512,
            'encoder_type': 'mobilenet',  # 'mobilenet' or 'resnet50'
            'decoder_type': 'lstm',  # 'lstm' or 'transformer'
            'num_layers': 1,
            'dropout': 0.5,
            'freeze_encoder': True,
            'use_pretrained_embeddings': False,
            'embedding_path': None
        }
        
        # Training configuration
        self.TRAINING_CONFIG = {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 40,
            'scheduler_type': 'warmup',  # 'warmup', 'step', 'cosine'
            'gradient_clip_value': 5.0,
            'teacher_forcing_ratio': 1.0,
            'accumulation_steps': 1,
            'mixed_precision': False,
            'early_stopping_patience': 5,
            'save_best_only': True
        }
        
        # Data processing configuration
        self.DATA_CONFIG = {
            'max_caption_length': 30,
            'min_word_freq': 2,
            'image_size': 224,
            'test_size': 0.1,
            'val_size': 0.1,
            'use_coco': True,  # Set to False for VizWiz
            'num_workers': 4,
            'pin_memory': True
        }
        
        # Special tokens
        self.SPECIAL_TOKENS = {
            'pad_token': '<PAD>',
            'unk_token': '<UNK>',
            'start_token': '<START>',
            'end_token': '<END>',
            'pad_idx': 0,
            'unk_idx': 1,
            'start_idx': 2,
            'end_idx': 3
        }
        
        # Image transforms configuration
        self.TRANSFORM_CONFIG = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'resize_size': 256,
            'crop_size': 224,
            'random_crop': True,
            'random_flip': True
        }
        
        # Evaluation configuration
        self.EVAL_CONFIG = {
            'beam_size': 3,
            'max_generation_length': 20,
            'repetition_penalty': 1.2,
            'compute_bleu': True,
            'compute_meteor': True,
            'compute_cider': True,
            'bleu_weights': [(1.0,), (0.5, 0.5), (0.33, 0.33, 0.33), (0.25, 0.25, 0.25, 0.25)]
        }
        
        # Logging configuration
        self.LOGGING_CONFIG = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_file': self.PROJECT_ROOT / 'logs' / 'visionassist.log',
            'max_bytes': 10485760,  # 10MB
            'backup_count': 5
        }
        
        # Device configuration
        self.DEVICE_CONFIG = {
            'use_cuda': True,
            'cuda_device': 0,
            'mixed_precision': False
        }
        
        # File paths
        self.FILE_PATHS = {
            'model_checkpoint': self.PROJECT_ROOT / 'best_model.pth',
            'vocab_file': self.DATA_DIR / 'word_to_idx.pkl',
            'train_filenames': self.DATA_DIR / 'train_filenames.npy',
            'val_filenames': self.DATA_DIR / 'val_filenames.npy',
            'test_filenames': self.DATA_DIR / 'test_filenames.npy',
            'train_captions': self.DATA_DIR / 'train_captions.pkl',
            'val_captions': self.DATA_DIR / 'val_captions.pkl',
            'test_captions': self.DATA_DIR / 'test_captions.pkl',
            'train_features': self.DATA_DIR / 'train_features.npy',
            'val_features': self.DATA_DIR / 'val_features.npy',
            'test_features': self.DATA_DIR / 'test_features.npy'
        }
        
        # Load custom configuration if provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        
        # Create necessary directories
        self._create_directories()
    
    def load_config(self, config_file: str):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
            
            # Update configurations
            for section, values in custom_config.items():
                if hasattr(self, section):
                    getattr(self, section).update(values)
                    
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def save_config(self, config_file: str):
        """Save current configuration to JSON file."""
        config_dict = {
            'MODEL_CONFIG': self.MODEL_CONFIG,
            'TRAINING_CONFIG': self.TRAINING_CONFIG,
            'DATA_CONFIG': self.DATA_CONFIG,
            'TRANSFORM_CONFIG': self.TRANSFORM_CONFIG,
            'EVAL_CONFIG': self.EVAL_CONFIG,
            'LOGGING_CONFIG': {k: str(v) for k, v in self.LOGGING_CONFIG.items()},
            'DEVICE_CONFIG': self.DEVICE_CONFIG
        }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
        except Exception as e:
            print(f"Error saving config file {config_file}: {e}")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            self.PROJECT_ROOT / 'logs',
            self.PROJECT_ROOT / 'checkpoints'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_image_dir(self) -> Path:
        """Get the appropriate image directory based on dataset choice."""
        if self.DATA_CONFIG['use_coco']:
            return self.COCO_IMAGE_DIR
        else:
            return self.VIZWIZ_IMAGE_DIR
    
    def get_annotation_file(self) -> Path:
        """Get the appropriate annotation file based on dataset choice."""
        if self.DATA_CONFIG['use_coco']:
            return self.COCO_ANNOTATION_FILE
        else:
            return self.VIZWIZ_ANNOTATION_FILE
    
    def validate_paths(self) -> bool:
        """Validate that all required paths exist."""
        required_paths = []
        
        if self.DATA_CONFIG['use_coco']:
            required_paths.extend([self.COCO_IMAGE_DIR, self.COCO_ANNOTATION_FILE])
        else:
            required_paths.extend([self.VIZWIZ_IMAGE_DIR, self.VIZWIZ_ANNOTATION_FILE])
        
        missing_paths = [path for path in required_paths if not path.exists()]
        
        if missing_paths:
            print(f"Warning: Missing required paths: {missing_paths}")
            return False
        
        return True
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"VisionAssist Config - Dataset: {'COCO' if self.DATA_CONFIG['use_coco'] else 'VizWiz'}"

# Global configuration instance
config = Config()

# Environment variable overrides
if os.getenv('VISIONASSIST_CONFIG'):
    config.load_config(os.getenv('VISIONASSIST_CONFIG'))

# Convenience functions
def get_config() -> Config:
    """Get the global configuration instance."""
    return config

def update_config(**kwargs):
    """Update configuration parameters."""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown configuration parameter: {key}")
