#!/usr/bin/env python3
"""
Embedding Manager for persistent storage and retrieval of face embeddings
Handles reference embeddings for gallery photo matching
"""

import os
import pickle
import json
import numpy as np
import cv2
import torch
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

class EmbeddingManager:
    """Manages face embeddings for reference database and similarity matching"""
    
    def __init__(self, storage_path: str, arcface_model, device, config=None):
        self.storage_path = storage_path
        self.arcface = arcface_model
        self.device = device
        self.config = config
        
        # Embedding storage
        self.reference_embeddings = {}  # {identity_name: [embeddings]}
        self.embedding_metadata = {}    # {identity_name: metadata}
        
        # File paths
        self.embeddings_file = os.path.join(storage_path, 'person_embeddings.pkl')
        self.metadata_file = os.path.join(storage_path, 'embedding_metadata.json')
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load existing embeddings if available
        self.load_embeddings()
    
    def extract_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding from a face image"""
        try:
            if face_image is None or face_image.size == 0:
                return None
            
            # Resize face image if too small
            if min(face_image.shape[:2]) < 112:
                face_image = cv2.resize(face_image, (112, 112))
            
            # Get embedding using ArcFace model
            embedding = self.arcface.get_embedding(face_image)
            
            if embedding is not None:
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting embedding: {str(e)}")
            return None
    
    def register_identity_from_folder(self, identity_name: str, folder_path: str) -> bool:
        """Register identity embeddings from a folder of images"""
        try:
            if not os.path.exists(folder_path):
                self.logger.error(f"Folder not found: {folder_path}")
                return False
            
            # Get all image files
            image_files = []
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            
            for filename in os.listdir(folder_path):
                if any(filename.lower().endswith(fmt) for fmt in supported_formats):
                    image_files.append(os.path.join(folder_path, filename))
            
            if not image_files:
                self.logger.warning(f"No valid image files found in {folder_path}")
                return False
            
            # Extract embeddings from all images
            embeddings = []
            processed_files = []
            
            for image_path in image_files:
                try:
                    # Load and process image
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Extract embedding
                    embedding = self.extract_face_embedding(image)
                    if embedding is not None:
                        embeddings.append(embedding)
                        processed_files.append(os.path.basename(image_path))
                    
                except Exception as e:
                    self.logger.warning(f"Error processing {image_path}: {str(e)}")
                    continue
            
            if not embeddings:
                self.logger.error(f"No valid embeddings extracted for {identity_name}")
                return False
            
            # Store embeddings and metadata
            self.reference_embeddings[identity_name] = embeddings
            self.embedding_metadata[identity_name] = {
                'num_embeddings': len(embeddings),
                'source_folder': folder_path,
                'processed_files': processed_files,
                'created_date': datetime.now().isoformat(),
                'embedding_dimension': len(embeddings[0])
            }
            
            # Save to disk
            self.save_embeddings()
            
            self.logger.info(f"Registered {len(embeddings)} embeddings for {identity_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering identity {identity_name}: {str(e)}")
            return False
    
    def add_reference_embedding(self, identity_name: str, embedding: np.ndarray, 
                              source_info: dict = None) -> bool:
        """Add a single reference embedding for an identity"""
        try:
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            # Initialize if new identity
            if identity_name not in self.reference_embeddings:
                self.reference_embeddings[identity_name] = []
                self.embedding_metadata[identity_name] = {
                    'num_embeddings': 0,
                    'created_date': datetime.now().isoformat(),
                    'embedding_dimension': len(embedding),
                    'sources': []
                }
            
            # Add embedding
            self.reference_embeddings[identity_name].append(embedding)
            
            # Update metadata
            self.embedding_metadata[identity_name]['num_embeddings'] += 1
            self.embedding_metadata[identity_name]['last_updated'] = datetime.now().isoformat()
            
            if source_info:
                self.embedding_metadata[identity_name]['sources'].append(source_info)
            
            # Save to disk
            self.save_embeddings()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding embedding for {identity_name}: {str(e)}")
            return False
    
    def find_best_match(self, query_embedding: np.ndarray, 
                       threshold: float = 0.6) -> Tuple[str, float]:
        """Find the best matching identity for a query embedding"""
        if not self.reference_embeddings:
            return "Unknown", 0.0
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        best_identity = "Unknown"
        best_similarity = 0.0
        
        try:
            for identity_name, embeddings in self.reference_embeddings.items():
                # Calculate similarity with all embeddings of this identity
                similarities = []
                
                for ref_embedding in embeddings:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        ref_embedding.reshape(1, -1)
                    )[0, 0]
                    similarities.append(similarity)
                
                # Use maximum similarity for this identity
                max_similarity = max(similarities)
                
                # Update best match if better
                if max_similarity > best_similarity and max_similarity >= threshold:
                    best_similarity = max_similarity
                    best_identity = identity_name
            
            return best_identity, best_similarity
            
        except Exception as e:
            self.logger.error(f"Error finding best match: {str(e)}")
            return "Unknown", 0.0
    
    def find_all_matches(self, query_embedding: np.ndarray, 
                        threshold: float = 0.6) -> List[Tuple[str, float]]:
        """Find all matching identities above threshold"""
        matches = []
        
        if not self.reference_embeddings:
            return matches
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        try:
            for identity_name, embeddings in self.reference_embeddings.items():
                # Calculate best similarity for this identity
                max_similarity = 0.0
                
                for ref_embedding in embeddings:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        ref_embedding.reshape(1, -1)
                    )[0, 0]
                    max_similarity = max(max_similarity, similarity)
                
                # Add to matches if above threshold
                if max_similarity >= threshold:
                    matches.append((identity_name, max_similarity))
            
            # Sort by similarity (descending)
            matches.sort(key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error finding matches: {str(e)}")
        
        return matches
    
    def save_embeddings(self):
        """Save embeddings and metadata to disk"""
        try:
            # Save embeddings as pickle
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.reference_embeddings, f)
            
            # Save metadata as JSON
            with open(self.metadata_file, 'w') as f:
                json.dump(self.embedding_metadata, f, indent=2)
            
            self.logger.debug("Embeddings and metadata saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {str(e)}")
    
    def load_embeddings(self):
        """Load embeddings and metadata from disk"""
        try:
            # Load embeddings
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'rb') as f:
                    self.reference_embeddings = pickle.load(f)
                self.logger.info(f"Loaded embeddings for {len(self.reference_embeddings)} identities")
            
            # Load metadata
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    self.embedding_metadata = json.load(f)
                self.logger.debug("Embedding metadata loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {str(e)}")
            self.reference_embeddings = {}
            self.embedding_metadata = {}
    
    def load_reference_embeddings(self):
        """Load reference embeddings (compatibility method)"""
        self.load_embeddings()
    
    def load_reference_embeddings_from_database(self, database_path: str):
        """Load reference embeddings from database folder structure"""
        try:
            if not os.path.exists(database_path):
                self.logger.error(f"Database path not found: {database_path}")
                return
            
            # Get all identity folders
            identity_folders = [d for d in os.listdir(database_path) 
                              if os.path.isdir(os.path.join(database_path, d))]
            
            if not identity_folders:
                self.logger.warning(f"No identity folders found in {database_path}")
                return
            
            # Process each identity folder
            for identity_name in identity_folders:
                identity_path = os.path.join(database_path, identity_name)
                
                # Skip if already registered
                if identity_name in self.reference_embeddings:
                    self.logger.info(f"Identity {identity_name} already registered, skipping...")
                    continue
                
                # Register identity from folder
                success = self.register_identity_from_folder(identity_name, identity_path)
                if success:
                    self.logger.info(f"Registered identity: {identity_name}")
                else:
                    self.logger.warning(f"Failed to register identity: {identity_name}")
            
            self.logger.info(f"Database loading complete. Total identities: {len(self.reference_embeddings)}")
            
        except Exception as e:
            self.logger.error(f"Error loading database: {str(e)}")
    
    def remove_identity(self, identity_name: str) -> bool:
        """Remove an identity from the reference database"""
        try:
            if identity_name in self.reference_embeddings:
                del self.reference_embeddings[identity_name]
                
            if identity_name in self.embedding_metadata:
                del self.embedding_metadata[identity_name]
            
            self.save_embeddings()
            self.logger.info(f"Removed identity: {identity_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing identity {identity_name}: {str(e)}")
            return False
    
    def get_identity_stats(self, identity_name: str) -> Optional[dict]:
        """Get statistics for a specific identity"""
        if identity_name not in self.embedding_metadata:
            return None
        
        metadata = self.embedding_metadata[identity_name].copy()
        
        # Add runtime statistics
        if identity_name in self.reference_embeddings:
            embeddings = self.reference_embeddings[identity_name]
            metadata['current_embedding_count'] = len(embeddings)
            
            # Calculate embedding statistics
            if embeddings:
                embeddings_array = np.array(embeddings)
                metadata['embedding_stats'] = {
                    'mean_norm': float(np.mean([np.linalg.norm(emb) for emb in embeddings])),
                    'std_norm': float(np.std([np.linalg.norm(emb) for emb in embeddings])),
                    'dimension': embeddings_array.shape[1]
                }
        
        return metadata
    
    def get_all_identities(self) -> List[str]:
        """Get list of all registered identities"""
        return list(self.reference_embeddings.keys())
    
    def get_database_summary(self) -> dict:
        """Get summary of the entire reference database"""
        summary = {
            'total_identities': len(self.reference_embeddings),
            'total_embeddings': sum(len(embeddings) for embeddings in self.reference_embeddings.values()),
            'identities': {},
            'created_date': min([meta.get('created_date', '') for meta in self.embedding_metadata.values()] + [''])
        }
        
        for identity_name in self.reference_embeddings:
            summary['identities'][identity_name] = {
                'embedding_count': len(self.reference_embeddings[identity_name]),
                'metadata': self.embedding_metadata.get(identity_name, {})
            }
        
        return summary
    
    def update_identity_metadata(self, identity_name: str, metadata_updates: dict):
        """Update metadata for an identity"""
        if identity_name not in self.embedding_metadata:
            self.embedding_metadata[identity_name] = {}
        
        self.embedding_metadata[identity_name].update(metadata_updates)
        self.embedding_metadata[identity_name]['last_updated'] = datetime.now().isoformat()
        
        self.save_embeddings()
    
    def cleanup_invalid_embeddings(self):
        """Remove invalid or corrupted embeddings"""
        removed_count = 0
        
        for identity_name in list(self.reference_embeddings.keys()):
            embeddings = self.reference_embeddings[identity_name]
            valid_embeddings = []
            
            for embedding in embeddings:
                try:
                    # Check if embedding is valid
                    if isinstance(embedding, np.ndarray) and embedding.size > 0:
                        # Check for NaN or infinite values
                        if not (np.isnan(embedding).any() or np.isinf(embedding).any()):
                            valid_embeddings.append(embedding)
                        else:
                            removed_count += 1
                    else:
                        removed_count += 1
                except:
                    removed_count += 1
            
            # Update embeddings list
            if valid_embeddings:
                self.reference_embeddings[identity_name] = valid_embeddings
                # Update metadata
                if identity_name in self.embedding_metadata:
                    self.embedding_metadata[identity_name]['num_embeddings'] = len(valid_embeddings)
            else:
                # Remove identity if no valid embeddings
                self.remove_identity(identity_name)
        
        if removed_count > 0:
            self.save_embeddings()
            self.logger.info(f"Cleaned up {removed_count} invalid embeddings")
        
        return removed_count