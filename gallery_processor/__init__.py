#!/usr/bin/env python3
"""
Gallery Processor Package
Enhanced PyTorch Face Recognition System - Gallery Processing Module

This package contains modules for batch processing photo galleries:
- EmbeddingManager: Handles persistent storage and retrieval of face embeddings
- BatchProcessor: Processes large collections of photos efficiently
- PhotoOrganizer: Organizes processed photos into person-specific folders
"""

__version__ = "1.0.0"
__author__ = "Enhanced PyTorch Face Recognition System"

from .embedding_manager import EmbeddingManager
from .batch_processor import BatchProcessor
from .photo_organizer import PhotoOrganizer

__all__ = [
    'EmbeddingManager',
    'BatchProcessor',
    'PhotoOrganizer'
]