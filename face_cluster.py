#!/usr/bin/env python3
import os
import shutil
import numpy as np
from deepface import DeepFace
from sklearn.cluster import DBSCAN
from pathlib import Path
import argparse
import logging
from PIL import Image
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_faces(image_path):
    """Extract face embeddings from an image using deepface."""
    try:
        # Get face embeddings using VGG-Face
        result = DeepFace.represent(img_path=str(image_path), model_name="VGG-Face", enforce_detection=False)
        
        # Convert to numpy arrays
        encodings = []
        for face_data in result:
            # Extract the embedding vector
            embedding = np.array(face_data["embedding"])
            encodings.append(embedding)
        
        return encodings
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return []

def cluster_faces(image_dir, output_dir, tolerance=0.6):
    """Cluster images based on faces and organize into directories."""
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(list(Path(image_dir).glob(ext)))
    
    if not image_paths:
        logger.error(f"No images found in {image_dir}")
        return False
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Extract face encodings
    encodings = []
    image_paths_with_faces = []
    
    for i, img_path in enumerate(tqdm(image_paths, desc="Processing images")):
        logger.debug(f"Processing image {i+1}/{len(image_paths)}: {img_path.name}")
        face_encodings = extract_faces(img_path)
        
        if face_encodings:
            for encoding in face_encodings:
                encodings.append(encoding)
                image_paths_with_faces.append(img_path)
    
    if not encodings:
        logger.error("No faces found in any images")
        return False
    
    logger.info(f"Found {len(encodings)} faces in {len(set(image_paths_with_faces))} images")
    
    # Cluster the faces
    clustering = DBSCAN(eps=tolerance, min_samples=1, metric="euclidean").fit(encodings)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a mapping from image paths to clusters
    image_clusters = {}
    for i, label in enumerate(clustering.labels_):
        img_path = image_paths_with_faces[i]
        if img_path not in image_clusters:
            image_clusters[img_path] = set()
        image_clusters[img_path].add(label)
    
    # Organize images into cluster directories
    for img_path, clusters in image_clusters.items():
        # For images with multiple faces, place in each cluster
        for cluster in clusters:
            cluster_dir = os.path.join(output_dir, f"cluster_{cluster}")
            os.makedirs(cluster_dir, exist_ok=True)
            
            # Copy image to cluster directory
            dest_path = os.path.join(cluster_dir, img_path.name)
            if not os.path.exists(dest_path):
                shutil.copy2(img_path, dest_path)
                logger.debug(f"Copied {img_path.name} to {cluster_dir}")
    
    # Handle images with no faces detected
    no_faces_dir = os.path.join(output_dir, "no_faces")
    os.makedirs(no_faces_dir, exist_ok=True)
    
    for img_path in image_paths:
        if img_path not in image_paths_with_faces:
            dest_path = os.path.join(no_faces_dir, img_path.name)
            if not os.path.exists(dest_path):
                shutil.copy2(img_path, dest_path)
                logger.debug(f"Copied {img_path.name} to {no_faces_dir} (no faces detected)")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Cluster images based on faces")
    parser.add_argument("--input", "-i", required=True, help="Directory containing input images")
    parser.add_argument("--output", "-o", required=True, help="Directory to output clustered images")
    parser.add_argument("--tolerance", "-t", type=float, default=0.6, 
                      help="Tolerance for face matching (lower is stricter, default: 0.6)")
    
    args = parser.parse_args()
    
    success = cluster_faces(args.input, args.output, args.tolerance)
    
    if success:
        logger.info(f"Successfully clustered images from {args.input} to {args.output}")
    else:
        logger.error(f"Failed to cluster images from {args.input}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())