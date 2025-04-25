# Face Clustering Tool

This tool clusters images based on faces detected in them and organizes them into directories.

## Requirements

- Python 3.6+
- deepface
- scikit-learn
- numpy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python face_cluster.py --input /path/to/images --output /path/to/output
```

### Options

- `--input`, `-i`: Directory containing input images (required)
- `--output`, `-o`: Directory to output clustered images (required)
- `--tolerance`, `-t`: Tolerance for face matching (lower is stricter, default: 0.6)

## How it works

1. The script scans all images in the input directory
2. For each image, faces are detected and encoded using facial recognition
3. The face encodings are clustered using AgglomerativeClustering algorithm
4. Images are organized into directories based on the clusters
5. Images with no detected faces are placed in a "no_faces" directory

## Testing

Run the unit tests with:

```bash
python -m unittest test_face_cluster.py
```
