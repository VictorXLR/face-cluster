#!/usr/bin/env python3
import unittest
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from face_cluster import extract_faces, cluster_faces

class TestFaceClustering(unittest.TestCase):
    
    def setUp(self):
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Remove test directories
        shutil.rmtree(self.test_dir, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)
    
    @patch('face_cluster.face_recognition')
    def test_extract_faces(self, mock_face_recognition):
        # Mock face_recognition behavior
        mock_image = MagicMock()
        mock_encodings = [MagicMock()]
        mock_face_recognition.load_image_file.return_value = mock_image
        mock_face_recognition.face_encodings.return_value = mock_encodings
        
        # Create a dummy test image
        test_image = os.path.join(self.test_dir, "test.jpg")
        with open(test_image, "w") as f:
            f.write("dummy image data")
        
        # Test the function
        result = extract_faces(test_image)
        
        # Assertions
        mock_face_recognition.load_image_file.assert_called_once_with(test_image)
        mock_face_recognition.face_encodings.assert_called_once_with(mock_image)
        self.assertEqual(result, mock_encodings)
    
    @patch('face_cluster.face_recognition')
    @patch('face_cluster.DBSCAN')
    def test_cluster_faces(self, mock_dbscan, mock_face_recognition):
        # Create mock images
        for i in range(3):
            img_path = os.path.join(self.test_dir, f"image{i}.jpg")
            with open(img_path, "w") as f:
                f.write("dummy image data")
        
        # Mock face recognition
        mock_encoding = MagicMock()
        mock_face_recognition.load_image_file.return_value = MagicMock()
        mock_face_recognition.face_encodings.return_value = [mock_encoding]
        
        # Mock DBSCAN clustering
        mock_dbscan_instance = MagicMock()
        mock_dbscan_instance.labels_ = [0, 0, 1]  # Two images in cluster 0, one in cluster 1
        mock_dbscan.return_value.fit.return_value = mock_dbscan_instance
        
        # Run clustering
        result = cluster_faces(self.test_dir, self.output_dir)
        
        # Verify results
        self.assertTrue(result)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "cluster_0")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "cluster_1")))
    
    @patch('face_cluster.extract_faces')
    def test_no_faces(self, mock_extract_faces):
        # Create mock images
        for i in range(2):
            img_path = os.path.join(self.test_dir, f"image{i}.jpg")
            with open(img_path, "w") as f:
                f.write("dummy image data")
        
        # Mock no faces found
        mock_extract_faces.return_value = []
        
        # Run clustering
        result = cluster_faces(self.test_dir, self.output_dir)
        
        # Verify results
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()