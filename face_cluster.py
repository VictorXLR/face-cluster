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
from collections import Counter, defaultdict
from sklearn.cluster import AgglomerativeClustering
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FACE_DETECTION_MODEL = "VGG-Face" 

def extract_faces(image_path):
    """Extract face embeddings from an image using deepface."""
    try:
        # Get face embeddings using VGG-Face
        result = DeepFace.represent(img_path=str(image_path), model_name=FACE_DETECTION_MODEL, enforce_detection=False)
        
        # Convert to numpy arrays
        encodings = [np.array(face_data["embedding"]) for face_data in result]
        return encodings
    except FileNotFoundError:
        logger.error(f"File not found: {image_path}")
    except ValueError as ve:
        logger.error(f"Value error for {image_path}: {ve}")
    except Exception as e:
        logger.error(f"Unexpected error for {image_path}: {e}\n{traceback.format_exc()}")
    return []

def is_blank_or_black(image_path):
    """Check if an image is blank (all white) or black (all black)."""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True
        # Check if all pixels are 0 (black) or 255 (white)
        if np.all(img == 0) or np.all(img == 255):
            return True
        # Optionally, check for very low variance (almost blank)
        if np.var(img) < 1e-3:
            return True
        return False
    except Exception as e:
        logger.error(f"Error reading {image_path}: {e}")
        return True

def is_blurry(image_path, threshold=30.0):
    """Check if an image is blurry using Laplacian variance."""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Image not found or unreadable: {image_path}")
            return True
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        return laplacian_var < threshold
    except Exception as e:
        logger.error(f"Error checking blur for {image_path}: {e}\n{traceback.format_exc()}")
        return True

def process_bad_images(input_dir):
    """Delete blank, black, blurry, or faceless images."""
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(list(Path(input_dir).glob(ext)))
    deleted = 0
    blurry_dir = Path(input_dir) / "blurry"
    blurry_dir.mkdir(exist_ok=True)
    for img_path in tqdm(image_paths, desc="Cleaning images"):
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None or np.all(img == 0) or np.all(img == 255) or np.var(img) < 1e-3:
                os.remove(img_path)
                logger.info(f"Deleted blank/black image: {img_path}")
                deleted += 1
                continue
            if is_blurry(img_path):
                shutil.move(str(img_path), blurry_dir / img_path.name)
                logger.info(f"Moved blurry image: {img_path}")
                continue
            faces = extract_faces(img_path)
            if not faces:
                os.remove(img_path)
                logger.info(f"Deleted faceless image: {img_path}")
                deleted += 1
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}\n{traceback.format_exc()}")
    logger.info(f"Deleted {deleted} bad images from {input_dir}")

def cluster_faces(image_dir, output_dir, tolerance=0.6, min_cluster_size=3, linkage='ward', metric='euclidean'):
    """Cluster images based on face appearance frequency and organize into directories."""
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(list(Path(image_dir).glob(ext)))

    if not image_paths:
        logger.error(f"No images found in {image_dir}")
        return False
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Extract face encodings
    encodings, image_paths_with_faces = extract_encodings(image_paths)
    if not encodings:
        logger.error("No faces found in any images")
        return False
    
    logger.info(f"Found {len(encodings)} faces in {len(set(image_paths_with_faces))} images")
    
    # Cluster the faces using AgglomerativeClustering for better control
    labels = perform_clustering(encodings, tolerance, linkage, metric)
    
    # Organize images into directories based on clustering
    organize_clusters(labels, image_paths_with_faces, output_dir, min_cluster_size)
    
    # Handle images with no faces detected
    handle_no_faces(image_paths, image_paths_with_faces, output_dir)

    return True

def extract_encodings(image_paths):
    encodings = []
    image_paths_with_faces = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        logger.debug(f"Processing image: {img_path.name}")
        face_encodings = extract_faces(img_path)
        
        if face_encodings:
            encodings.extend(face_encodings)
            image_paths_with_faces.append(img_path)
    
    return encodings, image_paths_with_faces

def perform_clustering(encodings, tolerance, linkage, metric):
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=tolerance, metric=metric, linkage=linkage)
    return clustering.fit_predict(encodings)

def organize_clusters(labels, image_paths_with_faces, output_dir, min_cluster_size):
    # Count appearances of each cluster
    cluster_counts = Counter(labels)
    
    # Merge small clusters into a single 'other' cluster
    merged_labels = [-1 if cluster_counts[label] < min_cluster_size else label for label in labels]
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    for i, label in enumerate(merged_labels):
        cluster_dir = os.path.join(output_dir, "other" if label == -1 else f"cluster_{label}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Copy image to cluster directory
        dest_path = os.path.join(cluster_dir, image_paths_with_faces[i].name)
        if not os.path.exists(dest_path):
            shutil.copy2(image_paths_with_faces[i], dest_path)
            logger.debug(f"Copied {image_paths_with_faces[i].name} to {cluster_dir}")

def handle_no_faces(image_paths, image_paths_with_faces, output_dir):
    # Handle images with no faces detected
    no_faces_dir = os.path.join(output_dir, "no_faces")
    os.makedirs(no_faces_dir, exist_ok=True)
    
    for img_path in image_paths:
        if img_path not in image_paths_with_faces:
            dest_path = os.path.join(no_faces_dir, img_path.name)
            if not os.path.exists(dest_path):
                shutil.copy2(img_path, dest_path)
                logger.debug(f"Copied {img_path.name} to {no_faces_dir} (no faces detected)")

def main():
    parser = argparse.ArgumentParser(description="Cluster images based on faces")
    parser.add_argument("--input", "-i", required=True, help="Directory containing input images")
    parser.add_argument("--output", "-o", required=True, help="Directory to output clustered images")
    parser.add_argument("--tolerance", "-t", type=float, default=0.6, 
                      help="Tolerance for face matching (lower is stricter, default: 0.6)")
    parser.add_argument("--linkage", "-l", type=str, default="ward", choices=["ward", "average", "complete"], help="Linkage method for clustering (default: ward)")
    parser.add_argument("--metric", "-m", type=str, default="euclidean", choices=["euclidean", "cosine"], help="Distance metric for clustering (default: euclidean)")

    args = parser.parse_args()
    
    # Process bad images before clustering
    process_bad_images(args.input)
    
    success = cluster_faces(args.input, args.output, args.tolerance, linkage=args.linkage, metric=args.metric)
    
    if success:
        logger.info(f"Successfully clustered images from {args.input} to {args.output}")
        return 0
    else:
        logger.error(f"Failed to cluster images from {args.input}")
        return 1

if __name__ == "__main__":
    exit(main())