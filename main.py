#!/usr/bin/env python3
"""
Enhanced PyTorch Face Recognition System
Supports: Registration, Real-time Recognition, and Gallery Processing

Usage:
    python main.py -m 0 -dp database                    # Registration mode
    python main.py -m 1 -dp database                    # Real-time recognition
    python main.py -m 2 --gallery_path ./photos --reference_path ./database --output_path ./output --confidence_threshold 0.6
"""

import argparse
import os
import sys
import cv2
import torch
import numpy as np
import logging
from datetime import datetime

# Import existing modules (assumed to be in the original project)
from models.retinaface import RetinaFace
from models.arcface import ArcFace
from utils.data_pipe import de_preprocess, get_train_dataset, get_val_dataset
from utils.buffer_update import update_buffer

# Import new gallery processing modules
from gallery_processor.embedding_manager import EmbeddingManager
from gallery_processor.batch_processor import BatchProcessor
from gallery_processor.photo_organizer import PhotoOrganizer
from config import Config

class FaceRecognitionSystem:
    """Enhanced Face Recognition System with Gallery Processing"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize models
        self.retinaface = None
        self.arcface = None
        self.embedding_manager = None
        self.batch_processor = None
        self.photo_organizer = None
        
        self.load_models()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.LOG_LEVEL.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_models(self):
        """Load RetinaFace and ArcFace models"""
        try:
            self.logger.info("Loading RetinaFace model...")
            self.retinaface = RetinaFace(
                network=self.config.RETINAFACE_NETWORK,
                device=self.device
            )
            
            self.logger.info("Loading ArcFace model...")
            self.arcface = ArcFace(
                network=self.config.ARCFACE_NETWORK,
                device=self.device
            )
            
            # Initialize gallery processing components
            self.embedding_manager = EmbeddingManager(
                self.config.REFERENCE_EMBEDDINGS_PATH,
                self.arcface,
                self.device
            )
            
            self.batch_processor = BatchProcessor(
                self.retinaface,
                self.arcface,
                self.device,
                self.config
            )
            
            self.photo_organizer = PhotoOrganizer(self.config)
            
            self.logger.info("All models loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            sys.exit(1)
    
    def registration_mode(self, database_path):
        """Original registration mode for known identities"""
        self.logger.info("Starting registration mode...")
        
        if not os.path.exists(database_path):
            os.makedirs(database_path)
            
        # Check existing identities
        identities = [d for d in os.listdir(database_path) 
                     if os.path.isdir(os.path.join(database_path, d))]
        
        if identities:
            print("\nExisting identities:")
            for i, identity in enumerate(identities):
                print(f"{i+1}. {identity}")
            print(f"{len(identities)+1}. Add new identity")
            
            choice = input("\nSelect option: ")
            try:
                choice = int(choice)
                if choice <= len(identities):
                    selected_identity = identities[choice-1]
                else:
                    selected_identity = input("Enter new identity name: ").strip()
            except:
                selected_identity = input("Enter identity name: ").strip()
        else:
            selected_identity = input("Enter identity name: ").strip()
        
        identity_path = os.path.join(database_path, selected_identity)
        os.makedirs(identity_path, exist_ok=True)
        
        # Capture and save face images
        cap = cv2.VideoCapture(0)
        count = 0
        
        print(f"\nRegistering identity: {selected_identity}")
        print("Press SPACE to capture, ESC to exit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect faces
            faces, landmarks = self.retinaface.detect(frame)
            
            # Draw bounding boxes
            for face in faces:
                x1, y1, x2, y2, conf = face.astype(np.int32)
                if conf > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{conf:.2f}', (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow('Registration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32 and len(faces) > 0:  # SPACE
                # Save the face image
                face = faces[0]  # Take first detected face
                x1, y1, x2, y2, conf = face.astype(np.int32)
                
                if conf > 0.5:
                    # Crop and save face
                    face_img = frame[y1:y2, x1:x2]
                    filename = f"img_{count:04d}.jpg"
                    filepath = os.path.join(identity_path, filename)
                    cv2.imwrite(filepath, face_img)
                    
                    print(f"Saved: {filename} (confidence: {conf:.2f})")
                    count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Generate embeddings for registered faces
        self.logger.info(f"Generating embeddings for {selected_identity}...")
        self.embedding_manager.register_identity_from_folder(selected_identity, identity_path)
        
        self.logger.info(f"Registration completed! Saved {count} images for {selected_identity}")
    
    def recognition_mode(self, database_path):
        """Original real-time recognition mode"""
        self.logger.info("Starting real-time recognition mode...")
        
        # Load reference embeddings
        self.embedding_manager.load_reference_embeddings()
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces, landmarks = self.retinaface.detect(frame)
            
            for face in faces:
                x1, y1, x2, y2, conf = face.astype(np.int32)
                
                if conf > self.config.FACE_DETECTION_THRESHOLD:
                    # Extract face region
                    face_img = frame[y1:y2, x1:x2]
                    
                    # Get embedding
                    embedding = self.arcface.get_embedding(face_img)
                    
                    # Find best match
                    identity, similarity = self.embedding_manager.find_best_match(
                        embedding, self.config.RECOGNITION_THRESHOLD
                    )
                    
                    # Draw results
                    color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{identity} ({similarity:.2f})" if identity != "Unknown" else "Unknown"
                    cv2.putText(frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def gallery_processing_mode(self, gallery_path, reference_path, output_path, confidence_threshold):
        """New gallery processing mode for batch photo organization"""
        self.logger.info("Starting gallery processing mode...")
        
        # Validate paths
        if not os.path.exists(gallery_path):
            self.logger.error(f"Gallery path does not exist: {gallery_path}")
            return
        
        if not os.path.exists(reference_path):
            self.logger.error(f"Reference path does not exist: {reference_path}")
            return
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Load reference embeddings from database
        self.logger.info("Loading reference embeddings...")
        self.embedding_manager.load_reference_embeddings_from_database(reference_path)
        
        # Process gallery images
        self.logger.info("Processing gallery images...")
        results = self.batch_processor.process_gallery(
            gallery_path, 
            self.embedding_manager,
            confidence_threshold
        )
        
        # Organize photos
        self.logger.info("Organizing photos...")
        summary = self.photo_organizer.organize_photos(results, gallery_path, output_path)
        
        # Generate report
        self.photo_organizer.generate_report(summary, output_path)
        
        self.logger.info("Gallery processing completed!")
        print(f"\nProcessing Summary:")
        print(f"Total images processed: {summary['total_processed']}")
        print(f"Images with faces: {summary['images_with_faces']}")
        print(f"Total faces detected: {summary['total_faces']}")
        print(f"Recognized faces: {summary['recognized_faces']}")
        print(f"Unknown faces: {summary['unknown_faces']}")
        print(f"Results saved to: {output_path}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced PyTorch Face Recognition System')
    parser.add_argument('-m', '--mode', type=int, required=True, 
                       choices=[0, 1, 2], 
                       help='Mode: 0=Registration, 1=Recognition, 2=Gallery Processing')
    parser.add_argument('-dp', '--database_path', default='database', 
                       help='Database path (for modes 0 and 1)')
    parser.add_argument('--gallery_path', 
                       help='Path to gallery images (for mode 2)')
    parser.add_argument('--reference_path', 
                       help='Path to reference database (for mode 2)')
    parser.add_argument('--output_path', default='output', 
                       help='Output path for organized photos (for mode 2)')
    parser.add_argument('--confidence_threshold', type=float, default=0.6, 
                       help='Confidence threshold for face matching (for mode 2)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Initialize system
    system = FaceRecognitionSystem(config)
    
    # Run appropriate mode
    if args.mode == 0:
        system.registration_mode(args.database_path)
    elif args.mode == 1:
        system.recognition_mode(args.database_path)
    elif args.mode == 2:
        if not args.gallery_path or not args.reference_path:
            print("Error: --gallery_path and --reference_path are required for gallery processing mode")
            sys.exit(1)
        system.gallery_processing_mode(
            args.gallery_path, 
            args.reference_path, 
            args.output_path, 
            args.confidence_threshold
        )

if __name__ == '__main__':
    main()