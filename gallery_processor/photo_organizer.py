#!/usr/bin/env python3
"""
Photo Organizer for organizing and delivering gallery processing results
Handles copying/organizing photos into person-specific folders and generating reports
"""

import os
import shutil
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict, Counter
import cv2

class PhotoOrganizer:
    """Organizes processed gallery photos by recognized persons"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def organize_photos(self, processing_results: List[Dict], 
                       source_gallery_path: str, output_path: str) -> Dict:
        """Organize photos into person-specific folders based on recognition results"""
        
        self.logger.info("Starting photo organization...")
        
        # Initialize summary statistics
        summary = {
            'total_processed': len(processing_results),
            'images_with_faces': 0,
            'images_without_faces': 0,
            'total_faces': 0,
            'recognized_faces': 0,
            'unknown_faces': 0,
            'person_counts': defaultdict(int),
            'error_count': 0,
            'organized_photos': defaultdict(list),
            'processing_errors': []
        }
        
        # Create output directories
        self._create_output_directories(output_path)
        
        # Process each image result
        for result in processing_results:
            try:
                self._process_image_result(result, source_gallery_path, output_path, summary)
            except Exception as e:
                self.logger.error(f"Error organizing {result.get('filename', 'unknown')}: {str(e)}")
                summary['error_count'] += 1
                summary['processing_errors'].append({
                    'filename': result.get('filename', 'unknown'),
                    'error': str(e)
                })
        
        # Generate metadata files
        self._save_metadata(processing_results, output_path, summary)
        
        self.logger.info(f"Photo organization completed. "
                        f"Processed {summary['total_processed']} images, "
                        f"found {summary['total_faces']} faces, "
                        f"recognized {summary['recognized_faces']} faces")
        
        return summary
    
    def _create_output_directories(self, output_path: str):
        """Create necessary output directories"""
        try:
            # Main output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Unknown faces directory
            unknown_dir = os.path.join(output_path, self.config.UNKNOWN_FOLDER)
            os.makedirs(unknown_dir, exist_ok=True)
            
            # Metadata directory
            metadata_dir = os.path.join(output_path, 'metadata')
            os.makedirs(metadata_dir, exist_ok=True)
            
            self.logger.debug(f"Created output directories in {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating output directories: {str(e)}")
            raise
    
    def _process_image_result(self, result: Dict, source_gallery_path: str, 
                            output_path: str, summary: Dict):
        """Process a single image result and organize accordingly"""
        
        filename = result['filename']
        image_path = result['image_path']
        faces = result['faces']
        
        # Handle processing errors
        if result.get('error'):
            summary['error_count'] += 1
            summary['processing_errors'].append({
                'filename': filename,
                'error': result['error']
            })
            return
        
        # Update face statistics
        if faces:
            summary['images_with_faces'] += 1
            summary['total_faces'] += len(faces)
            
            # Group faces by identity
            identity_groups = defaultdict(list)
            for face in faces:
                identity = face['identity']
                identity_groups[identity].append(face)
                
                if identity == "Unknown":
                    summary['unknown_faces'] += 1
                else:
                    summary['recognized_faces'] += 1
                    summary['person_counts'][identity] += 1
            
            # Copy image to appropriate folders
            self._copy_image_to_folders(image_path, filename, identity_groups, 
                                      output_path, summary)
        else:
            summary['images_without_faces'] += 1
    
    def _copy_image_to_folders(self, source_path: str, filename: str, 
                             identity_groups: Dict, output_path: str, summary: Dict):
        """Copy image to person-specific folders"""
        
        try:
            for identity, faces in identity_groups.items():
                # Create person-specific directory
                if identity == "Unknown":
                    person_dir = os.path.join(output_path, self.config.UNKNOWN_FOLDER)
                else:
                    person_dir = os.path.join(output_path, self._sanitize_folder_name(identity))
                
                os.makedirs(person_dir, exist_ok=True)
                
                # Create unique filename to avoid conflicts
                base_name, ext = os.path.splitext(filename)
                counter = 1
                target_filename = filename
                target_path = os.path.join(person_dir, target_filename)
                
                while os.path.exists(target_path):
                    target_filename = f"{base_name}_{counter:03d}{ext}"
                    target_path = os.path.join(person_dir, target_filename)
                    counter += 1
                
                # Copy the image
                shutil.copy2(source_path, target_path)
                
                # Store in summary
                summary['organized_photos'][identity].append({
                    'original_filename': filename,
                    'target_filename': target_filename,
                    'target_path': target_path,
                    'faces': faces
                })
                
                self.logger.debug(f"Copied {filename} to {identity} folder as {target_filename}")
        
        except Exception as e:
            self.logger.error(f"Error copying image {filename}: {str(e)}")
            raise
    
    def _sanitize_folder_name(self, name: str) -> str:
        """Sanitize folder name by removing invalid characters"""
        # Remove or replace invalid characters for folder names
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        
        # Remove leading/trailing spaces and dots
        name = name.strip(' .')
        
        # Ensure non-empty name
        if not name:
            name = "Unknown"
        
        # Limit length
        if len(name) > 50:
            name = name[:50]
        
        return name
    
    def _save_metadata(self, processing_results: List[Dict], output_path: str, summary: Dict):
        """Save detailed metadata about the processing results"""
        
        try:
            metadata_dir = os.path.join(output_path, 'metadata')
            
            # Save detailed processing results
            results_file = os.path.join(metadata_dir, 'detailed_results.json')
            with open(results_file, 'w') as f:
                json.dump(processing_results, f, indent=2, default=str)
            
            # Save summary statistics
            summary_file = os.path.join(metadata_dir, 'processing_summary.json')
            # Convert defaultdict to regular dict for JSON serialization
            json_summary = dict(summary)
            json_summary['person_counts'] = dict(summary['person_counts'])
            json_summary['organized_photos'] = dict(summary['organized_photos'])
            
            with open(summary_file, 'w') as f:
                json.dump(json_summary, f, indent=2, default=str)
            
            # Save face coordinates for each image (useful for verification)
            face_coords_file = os.path.join(metadata_dir, 'face_coordinates.json')
            face_coordinates = {}
            
            for result in processing_results:
                if result['faces']:
                    face_coordinates[result['filename']] = {
                        'image_size': result['image_size'],
                        'faces': [
                            {
                                'identity': face['identity'],
                                'bbox': face['bbox'],
                                'detection_confidence': face['detection_confidence'],
                                'recognition_confidence': face['recognition_confidence']
                            }
                            for face in result['faces']
                        ]
                    }
            
            with open(face_coords_file, 'w') as f:
                json.dump(face_coordinates, f, indent=2)
            
            self.logger.info(f"Metadata saved to {metadata_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}")
    
    def generate_report(self, summary: Dict, output_path: str):
        """Generate a comprehensive text report of the processing results"""
        
        try:
            report_file = os.path.join(output_path, self.config.REPORT_FILENAME)
            
            with open(report_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("PHOTO GALLERY PROCESSING REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Overall Statistics
                f.write("OVERALL STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Images Processed: {summary['total_processed']}\n")
                f.write(f"Images with Faces: {summary['images_with_faces']}\n")
                f.write(f"Images without Faces: {summary['images_without_faces']}\n")
                f.write(f"Total Faces Detected: {summary['total_faces']}\n")
                f.write(f"Recognized Faces: {summary['recognized_faces']}\n")
                f.write(f"Unknown Faces: {summary['unknown_faces']}\n")
                f.write(f"Processing Errors: {summary['error_count']}\n\n")
                
                # Recognition Rate
                if summary['total_faces'] > 0:
                    recognition_rate = (summary['recognized_faces'] / summary['total_faces']) * 100
                    f.write(f"Recognition Rate: {recognition_rate:.1f}%\n\n")
                
                # Person-wise Statistics
                if summary['person_counts']:
                    f.write("RECOGNIZED PERSONS\n")
                    f.write("-" * 40 + "\n")
                    
                    # Sort by face count (descending)
                    sorted_persons = sorted(summary['person_counts'].items(), 
                                          key=lambda x: x[1], reverse=True)
                    
                    for person, count in sorted_persons:
                        f.write(f"{person}: {count} face(s)\n")
                        
                        # List photos for this person
                        if person in summary['organized_photos']:
                            photos = summary['organized_photos'][person]
                            f.write(f"  Photos ({len(photos)}): ")
                            photo_names = [p['target_filename'] for p in photos[:5]]  # Show first 5
                            f.write(", ".join(photo_names))
                            if len(photos) > 5:
                                f.write(f" ... and {len(photos) - 5} more")
                            f.write("\n")
                        f.write("\n")
                
                # Unknown Faces
                if summary['unknown_faces'] > 0:
                    f.write("UNKNOWN FACES\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Total Unknown Faces: {summary['unknown_faces']}\n")
                    
                    if "Unknown" in summary['organized_photos']:
                        unknown_photos = summary['organized_photos']["Unknown"]
                        f.write(f"Images with Unknown Faces ({len(unknown_photos)}): ")
                        photo_names = [p['target_filename'] for p in unknown_photos[:10]]
                        f.write(", ".join(photo_names))
                        if len(unknown_photos) > 10:
                            f.write(f" ... and {len(unknown_photos) - 10} more")
                        f.write("\n\n")
                
                # Processing Errors
                if summary['processing_errors']:
                    f.write("PROCESSING ERRORS\n")
                    f.write("-" * 40 + "\n")
                    for error in summary['processing_errors']:
                        f.write(f"{error['filename']}: {error['error']}\n")
                    f.write("\n")
                
                # Folder Structure
                f.write("OUTPUT FOLDER STRUCTURE\n")
                f.write("-" * 40 + "\n")
                f.write(f"Main Output Directory: {output_path}\n")
                f.write("├── [Person Name]/\n")
                f.write("│   └── [Photos containing this person]\n")
                f.write(f"├── {self.config.UNKNOWN_FOLDER}/\n")
                f.write("│   └── [Photos with unrecognized faces]\n")
                f.write("├── metadata/\n")
                f.write("│   ├── detailed_results.json\n")
                f.write("│   ├── processing_summary.json\n")
                f.write("│   └── face_coordinates.json\n")
                f.write(f"└── {self.config.REPORT_FILENAME}\n\n")
                
                # Usage Instructions
                f.write("USAGE NOTES\n")
                f.write("-" * 40 + "\n")
                f.write("1. Check person-specific folders for photos containing each individual\n")
                f.write("2. Review the Unknown folder for unrecognized faces\n")
                f.write("3. Use metadata/face_coordinates.json to see exact face locations\n")
                f.write("4. Processing errors are listed above and in metadata/\n")
                f.write("5. Original images are preserved; only copies are organized\n\n")
                
            self.logger.info(f"Report generated: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
    
    def create_visual_report(self, processing_results: List[Dict], 
                           output_path: str, max_images: int = 50):
        """Create a visual report with sample images and face annotations"""
        
        try:
            visual_report_dir = os.path.join(output_path, 'visual_report')
            os.makedirs(visual_report_dir, exist_ok=True)
            
            # Select sample images with faces
            sample_results = [r for r in processing_results if r['faces'] and not r.get('error')]
            if len(sample_results) > max_images:
                # Take evenly distributed samples
                step = len(sample_results) // max_images
                sample_results = sample_results[::step][:max_images]
            
            self.logger.info(f"Creating visual report with {len(sample_results)} sample images")
            
            # Create annotated images
            for i, result in enumerate(sample_results):
                try:
                    # Load original image
                    image = cv2.imread(result['image_path'])
                    if image is None:
                        continue
                    
                    # Draw face annotations
                    for face in result['faces']:
                        bbox = face['bbox']
                        identity = face['identity']
                        confidence = face['recognition_confidence']
                        
                        # Choose color based on identity
                        if identity == "Unknown":
                            color = (0, 0, 255)  # Red for unknown
                        else:
                            color = (0, 255, 0)  # Green for recognized
                        
                        # Draw bounding box
                        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        
                        # Draw label
                        label = f"{identity} ({confidence:.2f})"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # Background rectangle for text
                        cv2.rectangle(image, 
                                    (bbox[0], bbox[1] - label_size[1] - 10),
                                    (bbox[0] + label_size[0], bbox[1]), 
                                    color, -1)
                        
                        # Text
                        cv2.putText(image, label, 
                                  (bbox[0], bbox[1] - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Save annotated image
                    output_filename = f"sample_{i:03d}_{result['filename']}"
                    output_filepath = os.path.join(visual_report_dir, output_filename)
                    cv2.imwrite(output_filepath, image)
                    
                except Exception as e:
                    self.logger.warning(f"Error creating visual for {result['filename']}: {str(e)}")
                    continue
            
            # Create HTML index file
            self._create_html_index(sample_results, visual_report_dir)
            
            self.logger.info(f"Visual report created in {visual_report_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating visual report: {str(e)}")
    
    def _create_html_index(self, sample_results: List[Dict], visual_report_dir: str):
        """Create HTML index file for visual report"""
        
        try:
            html_file = os.path.join(visual_report_dir, 'index.html')
            
            with open(html_file, 'w') as f:
                f.write("<!DOCTYPE html>\n")
                f.write("<html>\n<head>\n")
                f.write("<title>Face Recognition Visual Report</title>\n")
                f.write("<style>\n")
                f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
                f.write("h1 { color: #333; }\n")
                f.write(".image-container { margin: 20px 0; padding: 10px; border: 1px solid #ddd; }\n")
                f.write(".image-info { margin-bottom: 10px; font-weight: bold; }\n")
                f.write("img { max-width: 800px; height: auto; }\n")
                f.write(".face-info { margin: 5px 0; padding: 5px; background-color: #f5f5f5; }\n")
                f.write("</style>\n")
                f.write("</head>\n<body>\n")
                
                f.write("<h1>Face Recognition Visual Report</h1>\n")
                f.write(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
                f.write(f"<p>Sample Images: {len(sample_results)}</p>\n")
                
                for i, result in enumerate(sample_results):
                    f.write("<div class='image-container'>\n")
                    f.write(f"<div class='image-info'>Image: {result['filename']}</div>\n")
                    f.write(f"<div class='image-info'>Faces Detected: {len(result['faces'])}</div>\n")
                    
                    # List face information
                    for j, face in enumerate(result['faces']):
                        identity = face['identity']
                        confidence = face['recognition_confidence']
                        f.write(f"<div class='face-info'>Face {j+1}: {identity} "
                               f"(Confidence: {confidence:.2f})</div>\n")
                    
                    # Image
                    image_filename = f"sample_{i:03d}_{result['filename']}"
                    f.write(f"<img src='{image_filename}' alt='{result['filename']}'>\n")
                    f.write("</div>\n")
                
                f.write("</body>\n</html>\n")
            
        except Exception as e:
            self.logger.error(f"Error creating HTML index: {str(e)}")
    
    def cleanup_empty_folders(self, output_path: str):
        """Remove empty folders from output directory"""
        try:
            removed_folders = []
            
            for root, dirs, files in os.walk(output_path, topdown=False):
                for dirname in dirs:
                    dirpath = os.path.join(root, dirname)
                    try:
                        # Skip metadata and visual report folders
                        if dirname in ['metadata', 'visual_report']:
                            continue
                        
                        # Check if directory is empty
                        if not os.listdir(dirpath):
                            os.rmdir(dirpath)
                            removed_folders.append(dirpath)
                    except OSError:
                        pass  # Directory not empty or permission error
            
            if removed_folders:
                self.logger.info(f"Removed {len(removed_folders)} empty folders")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up empty folders: {str(e)}")
    
    def get_organization_summary(self, summary: Dict) -> str:
        """Get a brief text summary of organization results"""
        recognized_persons = len(summary['person_counts'])
        total_recognized_faces = summary['recognized_faces']
        unknown_faces = summary['unknown_faces']
        
        summary_text = (f"Organized {summary['total_processed']} images. "
                       f"Found {summary['total_faces']} faces total. "
                       f"Recognized {total_recognized_faces} faces from {recognized_persons} persons. "
                       f"{unknown_faces} faces remain unknown.")
        
        return summary_text