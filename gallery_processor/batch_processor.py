#!/usr/bin/env python3
"""
Batch Processor for processing large photo galleries
Handles efficient processing of multiple images with face detection and recognition
"""

import os
import cv2
import numpy as np
import torch
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple, Optional
import time
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed

class BatchProcessor:
    """Handles batch processing of photo galleries for face recognition"""
    
    def __init__(self, retinaface_model, arcface_model, device, config):
        self.retinaface = retinaface_model
        self.arcface = arcface_model
        self.device = device
        self.config = config
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Performance monitoring
        self.start_time = None
        self.processed_images = 0
        self.total_faces_detected = 0
        self.processing_times = []
        
        # Memory management
        self.max_memory_usage = config.MAX_MEMORY_USAGE_GB * 1024 * 1024 * 1024  # Convert to bytes
        
    def get_image_files(self, gallery_path: str) -> List[str]:
        """Get all valid image files from gallery path"""
        image_files = []
        
        # Supported formats
        supported_formats = tuple(self.config.SUPPORTED_IMAGE_FORMATS)
        
        try:
            if os.path.isfile(gallery_path):
                # Single file
                if gallery_path.lower().endswith(supported_formats):
                    image_files.append(gallery_path)
            else:
                # Directory - walk through all subdirectories
                for root, dirs, files in os.walk(gallery_path):
                    for file in files:
                        if file.lower().endswith(supported_formats):
                            image_files.append(os.path.join(root, file))
            
            self.logger.info(f"Found {len(image_files)} image files in {gallery_path}")
            
        except Exception as e:
            self.logger.error(f"Error scanning gallery path {gallery_path}: {str(e)}")
        
        return sorted(image_files)
    
    def process_single_image(self, image_path: str, embedding_manager, 
                           confidence_threshold: float) -> Dict:
        """Process a single image and return face detection/recognition results"""
        result = {
            'image_path': image_path,
            'filename': os.path.basename(image_path),
            'faces': [],
            'error': None,
            'processing_time': 0,
            'image_size': None
        }
        
        start_time = time.time()
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                result['error'] = "Could not load image"
                return result
            
            # Store image size
            result['image_size'] = image.shape[:2]  # (height, width)
            
            # Resize if image is too large (for memory efficiency)
            if (image.shape[0] > self.config.MAX_IMAGE_SIZE[1] or 
                image.shape[1] > self.config.MAX_IMAGE_SIZE[0]):
                
                # Calculate scaling factor
                scale_factor = min(
                    self.config.MAX_IMAGE_SIZE[0] / image.shape[1],
                    self.config.MAX_IMAGE_SIZE[1] / image.shape[0]
                )
                
                new_width = int(image.shape[1] * scale_factor)
                new_height = int(image.shape[0] * scale_factor)
                
                image = cv2.resize(image, (new_width, new_height))
                self.logger.debug(f"Resized image {image_path} by factor {scale_factor:.2f}")
            
            # Detect faces
            faces, landmarks = self.retinaface.detect(image)
            
            if len(faces) == 0:
                result['processing_time'] = time.time() - start_time
                return result
            
            # Limit number of faces processed per image
            if len(faces) > self.config.MAX_FACES_PER_IMAGE:
                # Sort by confidence and take top faces
                faces = sorted(faces, key=lambda x: x[4], reverse=True)[:self.config.MAX_FACES_PER_IMAGE]
                self.logger.debug(f"Limited faces to {self.config.MAX_FACES_PER_IMAGE} in {image_path}")
            
            # Process each detected face
            for i, face in enumerate(faces):
                try:
                    x1, y1, x2, y2, face_confidence = face.astype(np.int32)
                    
                    # Skip low confidence detections
                    if face_confidence < self.config.FACE_DETECTION_THRESHOLD:
                        continue
                    
                    # Skip very small faces
                    face_width = x2 - x1
                    face_height = y2 - y1
                    if min(face_width, face_height) < self.config.MIN_FACE_SIZE:
                        continue
                    
                    # Extract face region with margin
                    margin = int(max(face_width, face_height) * self.config.FACE_CROP_MARGIN)
                    x1_margin = max(0, x1 - margin)
                    y1_margin = max(0, y1 - margin)
                    x2_margin = min(image.shape[1], x2 + margin)
                    y2_margin = min(image.shape[0], y2 + margin)
                    
                    face_img = image[y1_margin:y2_margin, x1_margin:x2_margin]
                    
                    if face_img.size == 0:
                        continue
                    
                    # Extract embedding
                    embedding = embedding_manager.extract_face_embedding(face_img)
                    if embedding is None:
                        continue
                    
                    # Find matching identity
                    identity, similarity = embedding_manager.find_best_match(
                        embedding, confidence_threshold
                    )
                    
                    # Store face information
                    face_info = {
                        'face_id': i,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'bbox_with_margin': [int(x1_margin), int(y1_margin), 
                                           int(x2_margin), int(y2_margin)],
                        'detection_confidence': float(face_confidence),
                        'identity': identity,
                        'recognition_confidence': float(similarity),
                        'embedding': embedding.tolist()  # Convert to list for JSON serialization
                    }
                    
                    result['faces'].append(face_info)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing face {i} in {image_path}: {str(e)}")
                    continue
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def check_memory_usage(self):
        """Check current memory usage and warn if approaching limit"""
        process = psutil.Process()
        memory_usage = process.memory_info().rss  # bytes
        
        if memory_usage > self.max_memory_usage:
            self.logger.warning(f"Memory usage ({memory_usage / 1024**3:.1f} GB) "
                              f"exceeds limit ({self.max_memory_usage / 1024**3:.1f} GB)")
            return False
        
        return True
    
    def process_gallery(self, gallery_path: str, embedding_manager, 
                       confidence_threshold: float = 0.6) -> List[Dict]:
        """Process entire gallery of images"""
        self.logger.info(f"Starting gallery processing: {gallery_path}")
        self.start_time = time.time()
        
        # Get all image files
        image_files = self.get_image_files(gallery_path)
        if not image_files:
            self.logger.warning("No image files found in gallery")
            return []
        
        results = []
        
        # Process images with progress bar
        if self.config.ENABLE_PROGRESS_BAR:
            pbar = tqdm(total=len(image_files), desc="Processing images", 
                       unit="img", dynamic_ncols=True)
        
        try:
            # Process images in batches to manage memory
            batch_size = self.config.BATCH_SIZE
            
            for i in range(0, len(image_files), batch_size):
                batch_files = image_files[i:i + batch_size]
                
                # Check memory usage before processing batch
                if not self.check_memory_usage():
                    self.logger.warning("High memory usage detected, reducing batch size")
                    batch_size = max(1, batch_size // 2)
                    batch_files = image_files[i:i + batch_size]
                
                # Process batch
                if self.config.NUM_WORKERS > 1:
                    # Multi-threaded processing
                    batch_results = self._process_batch_parallel(
                        batch_files, embedding_manager, confidence_threshold
                    )
                else:
                    # Single-threaded processing
                    batch_results = self._process_batch_sequential(
                        batch_files, embedding_manager, confidence_threshold
                    )
                
                results.extend(batch_results)
                
                # Update progress
                if self.config.ENABLE_PROGRESS_BAR:
                    pbar.update(len(batch_files))
                
                # Update statistics
                self.processed_images += len(batch_files)
                for result in batch_results:
                    self.total_faces_detected += len(result['faces'])
                    self.processing_times.append(result['processing_time'])
                
                # Log progress periodically
                if (self.processed_images % self.config.PERFORMANCE_LOG_INTERVAL == 0 and
                    self.config.ENABLE_PERFORMANCE_MONITORING):
                    self._log_progress_stats()
        
        finally:
            if self.config.ENABLE_PROGRESS_BAR:
                pbar.close()
        
        # Log final statistics
        self._log_final_stats(len(image_files))
        
        return results
    
    def _process_batch_sequential(self, image_files: List[str], embedding_manager, 
                                 confidence_threshold: float) -> List[Dict]:
        """Process a batch of images sequentially"""
        results = []
        
        for image_path in image_files:
            result = self.process_single_image(image_path, embedding_manager, confidence_threshold)
            results.append(result)
        
        return results
    
    def _process_batch_parallel(self, image_files: List[str], embedding_manager, 
                              confidence_threshold: float) -> List[Dict]:
        """Process a batch of images in parallel using ThreadPoolExecutor"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.NUM_WORKERS) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_single_image, image_path, 
                               embedding_manager, confidence_threshold): image_path
                for image_path in image_files
            }
            
            # Collect results
            for future in as_completed(future_to_path):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    image_path = future_to_path[future]
                    self.logger.error(f"Error in parallel processing {image_path}: {str(e)}")
                    # Create error result
                    error_result = {
                        'image_path': image_path,
                        'filename': os.path.basename(image_path),
                        'faces': [],
                        'error': str(e),
                        'processing_time': 0,
                        'image_size': None
                    }
                    results.append(error_result)
        
        return results
    
    def _log_progress_stats(self):
        """Log progress statistics"""
        if self.start_time and self.processing_times:
            elapsed_time = time.time() - self.start_time
            avg_time_per_image = np.mean(self.processing_times[-self.config.PERFORMANCE_LOG_INTERVAL:])
            images_per_second = 1.0 / avg_time_per_image if avg_time_per_image > 0 else 0
            
            self.logger.info(f"Progress: {self.processed_images} images processed, "
                           f"{self.total_faces_detected} faces detected, "
                           f"{images_per_second:.1f} images/sec, "
                           f"elapsed: {elapsed_time:.1f}s")
    
    def _log_final_stats(self, total_images: int):
        """Log final processing statistics"""
        if self.start_time and self.processing_times:
            total_time = time.time() - self.start_time
            avg_time_per_image = np.mean(self.processing_times)
            total_faces = sum(1 for result in self.processing_times if result > 0)
            
            self.logger.info(f"Gallery processing completed!")
            self.logger.info(f"Total images: {total_images}")
            self.logger.info(f"Successfully processed: {self.processed_images}")
            self.logger.info(f"Total faces detected: {self.total_faces_detected}")
            self.logger.info(f"Average time per image: {avg_time_per_image:.2f}s")
            self.logger.info(f"Total processing time: {total_time:.1f}s")
            self.logger.info(f"Images per second: {self.processed_images / total_time:.1f}")
    
    def get_processing_summary(self) -> Dict:
        """Get summary of processing statistics"""
        if not self.start_time:
            return {}
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'total_processing_time': total_time,
            'images_processed': self.processed_images,
            'total_faces_detected': self.total_faces_detected,
            'average_time_per_image': np.mean(self.processing_times) if self.processing_times else 0,
            'images_per_second': self.processed_images / total_time if total_time > 0 else 0,
            'faces_per_image': self.total_faces_detected / self.processed_images if self.processed_images > 0 else 0
        }