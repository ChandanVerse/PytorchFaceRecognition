#!/usr/bin/env python3
"""
Similarity utility functions for face recognition system
Handles various similarity metrics and optimization techniques
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cdist
import faiss
import pickle

class SimilarityUtils:
    """Utility class for similarity calculations and optimizations"""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def cosine_similarity_single(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            if not embeddings:
                return np.array([])
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings)
            
            if metric == 'cosine':
                # Calculate cosine similarity matrix
                similarities = cosine_similarity(embeddings_array)
                return similarities
            elif metric == 'euclidean':
                # Calculate Euclidean distances and convert to similarities
                distances = euclidean_distances(embeddings_array)
                # Convert distances to similarities (0-1 scale)
                max_distance = np.max(distances)
                if max_distance > 0:
                    similarities = 1.0 - (distances / max_distance)
                else:
                    similarities = np.ones_like(distances)
                return similarities
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
        except Exception as e:
            logging.getLogger(__name__).error(f"Error calculating similarity matrix: {str(e)}")
            return np.array([])
    
    @staticmethod
    def find_top_k_similar(query_embedding: np.ndarray, 
                          reference_embeddings: List[np.ndarray],
                          k: int = 5, metric: str = 'cosine') -> List[Tuple[int, float]]:
        """Find top-k most similar embeddings to query"""
        try:
            if not reference_embeddings or k <= 0:
                return []
            
            similarities = []
            
            for i, ref_embedding in enumerate(reference_embeddings):
                if metric == 'cosine':
                    similarity = SimilarityUtils.cosine_similarity_single(
                        query_embedding, ref_embedding
                    )
                elif metric == 'euclidean':
                    distance = SimilarityUtils.euclidean_distance_single(
                        query_embedding, ref_embedding
                    )
                    similarity = SimilarityUtils.euclidean_to_similarity(distance)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                
                similarities.append((i, similarity))
            
            # Sort by similarity (descending) and return top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error finding top-k similar: {str(e)}")
            return []
    
    @staticmethod
    def cluster_embeddings_threshold(embeddings: List[np.ndarray], 
                                   threshold: float = 0.8, 
                                   metric: str = 'cosine') -> List[List[int]]:
        """Cluster embeddings based on similarity threshold"""
        try:
            if not embeddings:
                return []
            
            n = len(embeddings)
            similarity_matrix = SimilarityUtils.calculate_similarity_matrix(embeddings, metric)
            
            # Find clusters using threshold
            visited = [False] * n
            clusters = []
            
            for i in range(n):
                if visited[i]:
                    continue
                
                # Start new cluster
                cluster = [i]
                visited[i] = True
                
                # Add similar embeddings to cluster
                for j in range(i + 1, n):
                    if not visited[j] and similarity_matrix[i, j] >= threshold:
                        cluster.append(j)
                        visited[j] = True
                
                clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error clustering embeddings: {str(e)}")
            return []
    
    @staticmethod
    def calculate_average_embedding(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
        """Calculate average embedding from a list of embeddings"""
        try:
            if not embeddings:
                return None
            
            # Convert to numpy array and calculate mean
            embeddings_array = np.array(embeddings)
            avg_embedding = np.mean(embeddings_array, axis=0)
            
            # Normalize the average embedding
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            
            return avg_embedding
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error calculating average embedding: {str(e)}")
            return None
    
    @staticmethod
    def remove_outlier_embeddings(embeddings: List[np.ndarray], 
                                threshold: float = 0.5) -> List[int]:
        """Remove outlier embeddings based on average similarity to others"""
        try:
            if len(embeddings) <= 2:
                return list(range(len(embeddings)))
            
            similarity_matrix = SimilarityUtils.calculate_similarity_matrix(embeddings, 'cosine')
            
            # Calculate average similarity for each embedding
            avg_similarities = []
            for i in range(len(embeddings)):
                # Exclude self-similarity
                similarities = [similarity_matrix[i, j] for j in range(len(embeddings)) if i != j]
                avg_similarity = np.mean(similarities)
                avg_similarities.append(avg_similarity)
            
            # Filter out embeddings with low average similarity
            valid_indices = [i for i, avg_sim in enumerate(avg_similarities) 
                           if avg_sim >= threshold]
            
            return valid_indices
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error removing outliers: {str(e)}")
            return list(range(len(embeddings)))
    
    @staticmethod
    def optimize_threshold(positive_pairs: List[Tuple[np.ndarray, np.ndarray]],
                          negative_pairs: List[Tuple[np.ndarray, np.ndarray]],
                          metric: str = 'cosine') -> Tuple[float, Dict]:
        """Optimize similarity threshold based on positive and negative pairs"""
        try:
            # Calculate similarities for positive pairs
            positive_similarities = []
            for emb1, emb2 in positive_pairs:
                if metric == 'cosine':
                    sim = SimilarityUtils.cosine_similarity_single(emb1, emb2)
                else:
                    dist = SimilarityUtils.euclidean_distance_single(emb1, emb2)
                    sim = SimilarityUtils.euclidean_to_similarity(dist)
                positive_similarities.append(sim)
            
            # Calculate similarities for negative pairs
            negative_similarities = []
            for emb1, emb2 in negative_pairs:
                if metric == 'cosine':
                    sim = SimilarityUtils.cosine_similarity_single(emb1, emb2)
                else:
                    dist = SimilarityUtils.euclidean_distance_single(emb1, emb2)
                    sim = SimilarityUtils.euclidean_to_similarity(dist)
                negative_similarities.append(sim)
            
            # Find optimal threshold that maximizes accuracy
            all_similarities = positive_similarities + negative_similarities
            all_labels = [1] * len(positive_similarities) + [0] * len(negative_similarities)
            
            best_threshold = 0.5
            best_accuracy = 0.0
            best_metrics = {}
            
            # Test different thresholds
            thresholds = np.linspace(0.1, 0.9, 81)
            
            for threshold in thresholds:
                predictions = [1 if sim >= threshold else 0 for sim in all_similarities]
                
                # Calculate metrics
                tp = sum(1 for pred, label in zip(predictions, all_labels) if pred == 1 and label == 1)
                tn = sum(1 for pred, label in zip(predictions, all_labels) if pred == 0 and label == 0)
                fp = sum(1 for pred, label in zip(predictions, all_labels) if pred == 1 and label == 0)
                fn = sum(1 for pred, label in zip(predictions, all_labels) if pred == 0 and label == 1)
                
                accuracy = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
                    best_metrics = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'true_positives': tp,
                        'true_negatives': tn,
                        'false_positives': fp,
                        'false_negatives': fn
                    }
            
            return best_threshold, best_metrics
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error optimizing threshold: {str(e)}")
            return 0.6, {}

class FAISSIndex:
    """FAISS-based index for fast similarity search in large embedding databases"""
    
    def __init__(self, dimension: int, index_type: str = 'flat'):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.embeddings = []
        self.metadata = []
        self.logger = logging.getLogger(__name__)
        
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index"""
        try:
            if self.index_type == 'flat':
                # L2 distance (Euclidean)
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == 'cosine':
                # Inner product (for normalized vectors, equivalent to cosine similarity)
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == 'ivf':
                # IVF index for large datasets
                nlist = 100  # number of clusters
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
                
        except Exception as e:
            self.logger.error(f"Error creating FAISS index: {str(e)}")
            raise
    
    def add_embeddings(self, embeddings: List[np.ndarray], metadata: List[dict] = None):
        """Add embeddings to the index"""
        try:
            if not embeddings:
                return
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Normalize for cosine similarity
            if self.index_type == 'cosine':
                faiss.normalize_L2(embeddings_array)
            
            # Train index if needed (for IVF)
            if self.index_type == 'ivf' and not self.index.is_trained:
                self.index.train(embeddings_array)
            
            # Add to index
            self.index.add(embeddings_array)
            
            # Store embeddings and metadata
            self.embeddings.extend(embeddings)
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{}] * len(embeddings))
            
            self.logger.info(f"Added {len(embeddings)} embeddings to FAISS index")
            
        except Exception as e:
            self.logger.error(f"Error adding embeddings to index: {str(e)}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float, dict]]:
        """Search for k most similar embeddings"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Prepare query
            query = query_embedding.astype('float32').reshape(1, -1)
            
            # Normalize for cosine similarity
            if self.index_type == 'cosine':
                faiss.normalize_L2(query)
            
            # Search
            scores, indices = self.index.search(query, min(k, self.index.ntotal))
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid index
                    metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                    
                    # Convert score based on index type
                    if self.index_type == 'cosine':
                        similarity = float(score)  # Already similarity for IP
                    else:
                        # Convert L2 distance to similarity
                        similarity = SimilarityUtils.euclidean_to_similarity(float(score))
                    
                    results.append((int(idx), similarity, metadata))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching FAISS index: {str(e)}")
            return []
    
    def save_index(self, filepath: str):
        """Save FAISS index to disk"""
        try:
            faiss.write_index(self.index, filepath)
            
            # Save metadata separately
            metadata_file = filepath.replace('.index', '_metadata.pkl')
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'embeddings': self.embeddings,
                    'metadata': self.metadata,
                    'dimension': self.dimension,
                    'index_type': self.index_type
                }, f)
            
            self.logger.info(f"FAISS index saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving FAISS index: {str(e)}")
    
    def load_index(self, filepath: str):
        """Load FAISS index from disk"""
        try:
            self.index = faiss.read_index(filepath)
            
            # Load metadata
            metadata_file = filepath.replace('.index', '_metadata.pkl')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data.get('embeddings', [])
                    self.metadata = data.get('metadata', [])
                    self.dimension = data.get('dimension', self.dimension)
                    self.index_type = data.get('index_type', self.index_type)
            
            self.logger.info(f"FAISS index loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading FAISS index: {str(e)}")
    
    def get_stats(self) -> dict:
        """Get index statistics"""
        return {
            'total_embeddings': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }
            # Normalize embeddings
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    @staticmethod
    def euclidean_distance_single(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate Euclidean distance between two embeddings"""
        try:
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(distance)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error calculating Euclidean distance: {str(e)}")
            return float('inf')
    
    @staticmethod
    def euclidean_to_similarity(distance: float, max_distance: float = 2.0) -> float:
        """Convert Euclidean distance to similarity score (0-1)"""
        # Normalize distance to similarity (closer = higher similarity)
        similarity = 1.0 - min(distance / max_distance, 1.0)
        return max(0.0, similarity)
    
    @staticmethod
    def calculate_similarity_matrix(embeddings: List[np.ndarray], metric: str = 'cosine') -> np.ndarray:
        """Calculate pairwise similarity matrix for a list of embeddings"""
        try