#!/usr/bin/env python3
"""
Image utility functions for enhanced face recognition system
Handles image preprocessing, validation, and quality assessment
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional, List
import logging
from PIL import Image, ExifTags
import hashlib

class ImageUtils:
    """Utility class for image processing operations"""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def is_valid_image_file(filepath: str) -> bool:
        """Check if file is a valid image file"""
        try:
            if not os.path.exists(filepath):
                return False
            
            # Check file extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            _, ext = os.path.splitext(filepath.lower())
            if ext not in valid_extensions:
                return False
            
            # Try to open with OpenCV
            image = cv2.imread(filepath)
            if image is None:
                return False
            
            # Check if image has valid dimensions
            if len(image.shape) < 2 or image.shape[0] < 10 or image.shape[1] < 10:
                return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def load_image_safe(filepath: str) -> Optional[np.ndarray]:
        """Safely load an image file with error handling"""
        try:
            if not ImageUtils.is_valid_image_file(filepath):
                return None
            
            image = cv2.imread(filepath)
            if image is None:
                return None
            
            # Handle EXIF orientation
            image = ImageUtils.correct_image_orientation(image, filepath)
            
            return image
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Error loading image {filepath}: {str(e)}")
            return None
    
    @staticmethod
    def correct_image_orientation(image: np.ndarray, filepath: str) -> np.ndarray:
        """Correct image orientation based on EXIF data"""
        try:
            # Open with PIL to read EXIF
            pil_image = Image.open(filepath)
            
            # Get EXIF orientation
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            
            if hasattr(pil_image, '_getexif'):
                exif = pil_image._getexif()
                if exif is not None and orientation in exif:
                    orientation_value = exif[orientation]
                    
                    # Apply rotation based on orientation
                    if orientation_value == 3:
                        image = cv2.rotate(image, cv2.ROTATE_180)
                    elif orientation_value == 6:
                        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    elif orientation_value == 8:
                        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        except Exception:
            # If EXIF processing fails, return original image
            pass
        
        return image
    
    @staticmethod
    def resize_image_maintain_aspect(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h)
        
        # Only resize if image is larger than max dimensions
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image
    
    @staticmethod
    def assess_image_quality(image: np.ndarray) -> float:
        """Assess image quality using various metrics (0-100 score)"""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(100, laplacian_var / 100 * 100)
            
            # Calculate brightness
            brightness = np.mean(gray)
            brightness_score = 100 - abs(brightness - 128) / 128 * 100
            
            # Calculate contrast
            contrast = np.std(gray)
            contrast_score = min(100, contrast / 64 * 100)
            
            # Combine scores
            quality_score = (sharpness_score * 0.5 + brightness_score * 0.3 + contrast_score * 0.2)
            
            return max(0, min(100, quality_score))
            
        except Exception:
            return 50.0  # Default medium quality score
    
    @staticmethod
    def detect_blur(image: np.ndarray, threshold: float = 100.0) -> Tuple[bool, float]:
        """Detect if image is blurry using Laplacian variance"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate Laplacian variance
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            is_blurry = variance < threshold
            return is_blurry, variance
            
        except Exception:
            return False, threshold
    
    @staticmethod
    def enhance_face_image(face_image: np.ndarray) -> np.ndarray:
        """Enhance face image quality for better recognition"""
        try:
            # Convert to RGB if needed
            if len(face_image.shape) == 3:
                # Apply histogram equalization to improve contrast
                lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
                lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                enhanced = cv2.equalizeHist(face_image)
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            # Apply slight Gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            return enhanced
            
        except Exception:
            return face_image
    
    @staticmethod
    def extract_face_with_margin(image: np.ndarray, bbox: List[int], 
                               margin_ratio: float = 0.2) -> Optional[np.ndarray]:
        """Extract face region with specified margin"""
        try:
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            
            # Calculate margin
            face_w = x2 - x1
            face_h = y2 - y1
            margin_x = int(face_w * margin_ratio)
            margin_y = int(face_h * margin_ratio)
            
            # Apply margin with bounds checking
            x1_margin = max(0, x1 - margin_x)
            y1_margin = max(0, y1 - margin_y)
            x2_margin = min(w, x2 + margin_x)
            y2_margin = min(h, y2 + margin_y)
            
            # Extract face region
            face_image = image[y1_margin:y2_margin, x1_margin:x2_margin]
            
            if face_image.size == 0:
                return None
            
            return face_image
            
        except Exception:
            return None
    
    @staticmethod
    def calculate_face_size(bbox: List[int]) -> int:
        """Calculate face size (area) from bounding box"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return width * height
    
    @staticmethod
    def is_face_size_valid(bbox: List[int], min_size: int = 30) -> bool:
        """Check if face size meets minimum requirements"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return width >= min_size and height >= min_size
    
    @staticmethod
    def calculate_image_hash(image_path: str) -> str:
        """Calculate MD5 hash of image file for duplicate detection"""
        try:
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception:
            return ""
    
    @staticmethod
    def find_duplicate_images(image_paths: List[str]) -> List[List[str]]:
        """Find duplicate images based on file hash"""
        hash_to_paths = {}
        
        for path in image_paths:
            file_hash = ImageUtils.calculate_image_hash(path)
            if file_hash:
                if file_hash not in hash_to_paths:
                    hash_to_paths[file_hash] = []
                hash_to_paths[file_hash].append(path)
        
        # Return groups of duplicates (groups with more than 1 image)
        duplicates = [paths for paths in hash_to_paths.values() if len(paths) > 1]
        return duplicates
    
    @staticmethod
    def create_thumbnail(image: np.ndarray, size: Tuple[int, int] = (150, 150)) -> np.ndarray:
        """Create thumbnail of specified size"""
        try:
            thumbnail = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            return thumbnail
        except Exception:
            return image
    
    @staticmethod
    def save_image_with_metadata(image: np.ndarray, output_path: str, 
                               metadata: dict = None) -> bool:
        """Save image with optional metadata"""
        try:
            # Save image
            success = cv2.imwrite(output_path, image)
            
            if success and metadata:
                # Save metadata as JSON sidecar file
                metadata_path = os.path.splitext(output_path)[0] + '_metadata.json'
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            return success
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error saving image {output_path}: {str(e)}")
            return False
    
    @staticmethod
    def batch_resize_images(image_paths: List[str], output_dir: str, 
                          max_size: Tuple[int, int] = (1920, 1080)) -> List[str]:
        """Batch resize images to specified maximum dimensions"""
        resized_paths = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        for image_path in image_paths:
            try:
                image = ImageUtils.load_image_safe(image_path)
                if image is None:
                    continue
                
                # Resize if needed
                resized_image = ImageUtils.resize_image_maintain_aspect(
                    image, max_size[0], max_size[1]
                )
                
                # Save resized image
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_dir, filename)
                
                if cv2.imwrite(output_path, resized_image):
                    resized_paths.append(output_path)
                
            except Exception as e:
                logging.getLogger(__name__).warning(f"Error resizing {image_path}: {str(e)}")
        
        return resized_paths
    
    @staticmethod
    def get_image_info(image_path: str) -> dict:
        """Get comprehensive information about an image file"""
        info = {
            'filepath': image_path,
            'filename': os.path.basename(image_path),
            'file_size_bytes': 0,
            'dimensions': None,
            'channels': 0,
            'format': None,
            'quality_score': 0,
            'is_valid': False,
            'error': None
        }
        
        try:
            if not os.path.exists(image_path):
                info['error'] = "File not found"
                return info
            
            # File size
            info['file_size_bytes'] = os.path.getsize(image_path)
            
            # Load image
            image = ImageUtils.load_image_safe(image_path)
            if image is None:
                info['error'] = "Could not load image"
                return info
            
            # Image properties
            info['dimensions'] = (image.shape[1], image.shape[0])  # (width, height)
            info['channels'] = image.shape[2] if len(image.shape) > 2 else 1
            info['format'] = os.path.splitext(image_path)[1].lower()
            info['quality_score'] = ImageUtils.assess_image_quality(image)
            info['is_valid'] = True
            
        except Exception as e:
            info['error'] = str(e)
        
        return info