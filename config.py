#!/usr/bin/env python3
"""
Configuration settings for Enhanced PyTorch Face Recognition System
"""

import os
from datetime import datetime

class Config:
    """Configuration class containing all system settings"""
    
    # Model Configuration
    RETINAFACE_NETWORK = 'mobile0.25'  # or 'resnet50'
    ARCFACE_NETWORK = 'r100'  # or 'r50', 'r34', 'r18'
    
    # Model Weights Paths
    WEIGHTS_DIR = 'weights'
    RETINAFACE_WEIGHTS = os.path.join(WEIGHTS_DIR, 'mobilenet0.25_Final.pth')
    ARCFACE_WEIGHTS = os.path.join(WEIGHTS_DIR, 'backbone.pth')
    
    # Face Detection Thresholds
    FACE_DETECTION_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    
    # Face Recognition Thresholds
    RECOGNITION_THRESHOLD = 0.6
    SIMILARITY_METRIC = 'cosine'  # 'cosine' or 'euclidean'
    
    # Embedding Configuration
    EMBEDDING_SIZE = 512
    REFERENCE_EMBEDDINGS_PATH = 'reference_embeddings'
    EMBEDDING_FILE = 'person_embeddings.pkl'
    METADATA_FILE = 'embedding_metadata.json'
    
    # Gallery Processing Configuration
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    BATCH_SIZE = 32
    MAX_IMAGE_SIZE = (1920, 1080)  # Resize large images for memory efficiency
    MIN_FACE_SIZE = 30  # Minimum face size in pixels
    
    # Output Configuration
    OUTPUT_DIR = 'output'
    UNKNOWN_FOLDER = 'Unknown'
    REPORT_FILENAME = 'processing_report.txt'
    METADATA_FILENAME = 'face_metadata.json'
    
    # Memory and Performance
    MAX_MEMORY_USAGE_GB = 8  # Maximum memory usage for batch processing
    NUM_WORKERS = 4  # Number of worker threads for parallel processing
    ENABLE_GPU_ACCELERATION = True
    
    # Image Processing
    FACE_CROP_MARGIN = 0.2  # Margin around detected face (20%)
    IMAGE_QUALITY_THRESHOLD = 50  # Minimum image quality score
    MAX_FACES_PER_IMAGE = 50  # Maximum faces to process per image
    
    # Logging Configuration
    LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
    LOG_FILE = f'logs/face_recognition_{datetime.now().strftime("%Y%m%d")}.log'
    ENABLE_PROGRESS_BAR = True
    
    # Database Configuration
    DATABASE_DIR = 'database'
    MIN_IMAGES_PER_IDENTITY = 1
    MAX_IMAGES_PER_IDENTITY = 100
    
    # Advanced Settings
    ENABLE_FACE_ALIGNMENT = True
    ENABLE_QUALITY_FILTERING = True
    ENABLE_DUPLICATE_DETECTION = True
    DUPLICATE_THRESHOLD = 0.95
    
    # Performance Monitoring
    ENABLE_PERFORMANCE_MONITORING = True
    PERFORMANCE_LOG_INTERVAL = 100  # Log performance every N images
    
    def __init__(self):
        """Initialize configuration and create necessary directories"""
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.WEIGHTS_DIR,
            self.REFERENCE_EMBEDDINGS_PATH,
            self.OUTPUT_DIR,
            self.DATABASE_DIR,
            'logs',
            'utils',
            'gallery_processor'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_embedding_file_path(self):
        """Get full path to embedding file"""
        return os.path.join(self.REFERENCE_EMBEDDINGS_PATH, self.EMBEDDING_FILE)
    
    def get_metadata_file_path(self):
        """Get full path to metadata file"""
        return os.path.join(self.REFERENCE_EMBEDDINGS_PATH, self.METADATA_FILE)
    
    def get_log_file_path(self):
        """Get full path to log file"""
        os.makedirs('logs', exist_ok=True)
        return self.LOG_FILE
    
    def update_from_args(self, args):
        """Update configuration from command line arguments"""
        if hasattr(args, 'confidence_threshold'):
            self.RECOGNITION_THRESHOLD = args.confidence_threshold
        if hasattr(args, 'output_path'):
            self.OUTPUT_DIR = args.output_path
    
    def validate_config(self):
        """Validate configuration settings"""
        errors = []
        
        # Check if weight files exist
        if not os.path.exists(self.RETINAFACE_WEIGHTS):
            errors.append(f"RetinaFace weights not found: {self.RETINAFACE_WEIGHTS}")
        
        if not os.path.exists(self.ARCFACE_WEIGHTS):
            errors.append(f"ArcFace weights not found: {self.ARCFACE_WEIGHTS}")
        
        # Validate thresholds
        if not 0 <= self.RECOGNITION_THRESHOLD <= 1:
            errors.append("Recognition threshold must be between 0 and 1")
        
        if not 0 <= self.FACE_DETECTION_THRESHOLD <= 1:
            errors.append("Face detection threshold must be between 0 and 1")
        
        # Validate batch size
        if self.BATCH_SIZE <= 0:
            errors.append("Batch size must be positive")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
        
        return True
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
    
    def save_config(self, filepath):
        """Save configuration to file"""
        import json
        
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def load_config(self, filepath):
        """Load configuration from file"""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

# Global configuration instance
config = Config()