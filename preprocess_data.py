import os
import cv2
import numpy as np
from core.detection import RetinaDetector
import shutil
from pathlib import Path
from collections import defaultdict
import re

class CelebDatasetPreprocessor:
    def __init__(self, dataset_path, output_path="database"):
        self.dataset_path = dataset_path
        self.output_path = output_path
        
    def extract_name_from_filename(self, filename):
        """
        Extract celebrity name from filename
        Examples: AJ_Cook_0001.jpg -> AJ_Cook
                 Aaron_Peirsol_0003.jpg -> Aaron_Peirsol
        """
        # Remove file extension
        name_part = Path(filename).stem
        
        # Split by underscore and remove the number part (last element)
        parts = name_part.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            # Join all parts except the last (which is the number)
            celebrity_name = '_'.join(parts[:-1])
            return celebrity_name
        
        return None
    
    def organize_by_celebrity(self):
        """
        Group image files by celebrity name
        """
        celebrity_groups = defaultdict(list)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(self.dataset_path).rglob(f'*{ext}'))
            image_files.extend(Path(self.dataset_path).rglob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} total images")
        
        # Group by celebrity name
        for img_path in image_files:
            celebrity_name = self.extract_name_from_filename(img_path.name)
            if celebrity_name:
                celebrity_groups[celebrity_name].append(img_path)
        
        print(f"Found {len(celebrity_groups)} different celebrities")
        
        # Sort files for each celebrity
        filtered_groups = {}
        for name, files in celebrity_groups.items():
            # Sort files to ensure consistent ordering
            sorted_files = sorted(files, key=lambda x: x.name)
            filtered_groups[name] = sorted_files
        
        print(f"Total celebrities found: {len(filtered_groups)}")
        
        return filtered_groups
    
    def process_celebrity_images(self, celebrity_groups):
        """
        Process and save celebrity images with simple resizing
        """
        os.makedirs(self.output_path, exist_ok=True)
        
        total_processed = 0
        successful_celebrities = 0
        
        for celebrity_name, image_files in celebrity_groups.items():
            print(f"\nProcessing {celebrity_name} ({len(image_files)} images)...")
            
            # Create celebrity directory
            celebrity_dir = os.path.join(self.output_path, celebrity_name)
            os.makedirs(celebrity_dir, exist_ok=True)
            
            image_count = 0
            
            for i, img_path in enumerate(image_files):
                try:
                    # Read image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"  Could not read {img_path.name}")
                        continue
                    
                    # Resize to standard size
                    img_resized = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
                    
                    # Save processed image
                    output_filename = f"{celebrity_name}_{image_count:03d}.jpg"
                    output_path = os.path.join(celebrity_dir, output_filename)
                    cv2.imwrite(output_path, img_resized)
                    
                    image_count += 1
                    total_processed += 1
                    print(f"  ✓ Saved {output_filename}")
                    
                except Exception as e:
                    print(f"  ✗ Error processing {img_path.name}: {e}")
                    continue
            
            successful_celebrities += 1
            print(f"  ✓ Successfully processed {celebrity_name}: {image_count} images")
        
        return total_processed, successful_celebrities
    
    def process_dataset(self):
        """
        Complete processing pipeline
        """
        print("=== Celebrity Face Recognition Database Creator ===\n")
        
        # Step 1: Organize images by celebrity
        celebrity_groups = self.organize_by_celebrity()
        
        if not celebrity_groups:
            print("No celebrity groups found. Check your dataset path and file naming.")
            return
        
        # Step 2: Process images
        total_processed, successful_celebrities = self.process_celebrity_images(celebrity_groups)
        
        # Step 3: Summary
        print("\n=== Processing Complete ===")
        print(f"Total images processed: {total_processed}")
        print(f"Successful celebrities: {successful_celebrities}")
        print(f"Database created at: {self.output_path}")
        
        # List created identities
        if successful_celebrities > 0:
            print(f"\nCreated identities:")
            for identity_dir in sorted(os.listdir(self.output_path)):
                if os.path.isdir(os.path.join(self.output_path, identity_dir)):
                    image_count = len([f for f in os.listdir(os.path.join(self.output_path, identity_dir)) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    print(f"  {identity_dir}: {image_count} images")

def main():
    # Configuration
    kaggle_dataset_path = "Large" 
    output_database_path = "dataset"
    
    # Create preprocessor
    preprocessor = CelebDatasetPreprocessor(
        dataset_path=kaggle_dataset_path,
        output_path=output_database_path
    )
    
    # Process the dataset
    preprocessor.process_dataset()

if __name__ == "__main__":
    main()