"""
Script to download and prepare sample data for Text-to-CAD model training.
"""

import os
import json
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil

def download_file(url: str, destination: Path) -> None:
    """Download a file from URL with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def prepare_sample_data():
    """Prepare sample data for training."""
    # Create necessary directories
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    # Sample data URLs (replace with actual URLs)
    sample_data_url = "https://example.com/sample_cad_data.zip"  # Replace with actual URL
    
    # Download sample data
    print("Downloading sample data...")
    zip_path = raw_dir / "sample_data.zip"
    download_file(sample_data_url, zip_path)
    
    # Extract data
    print("Extracting data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)
    
    # Process data
    print("Processing data...")
    # Add your data processing logic here
    # This could include:
    # - Converting CAD files to a common format
    # - Extracting features
    # - Creating text descriptions
    # - Splitting into train/val/test sets
    
    # Create sample metadata
    metadata = {
        "total_samples": 0,
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0,
        "categories": [],
        "file_formats": []
    }
    
    # Save metadata
    with open(processed_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Clean up
    print("Cleaning up...")
    zip_path.unlink()
    
    print("Data preparation complete!")

if __name__ == "__main__":
    prepare_sample_data() 