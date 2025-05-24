"""
Script to generate sample data for testing the Text-to-CAD model.
"""

import json
import numpy as np
from pathlib import Path
import random

def generate_cad_features() -> list:
    """Generate random CAD features for testing."""
    # Generate random features (replace with actual CAD feature extraction)
    return np.random.rand(128).tolist()

def generate_sample_description() -> str:
    """Generate a sample text description."""
    descriptions = [
        "A simple cube with rounded edges",
        "A cylindrical container with a lid",
        "A rectangular prism with chamfered corners",
        "A sphere with a cylindrical hole through the center",
        "A hexagonal nut with standard thread pitch",
        "A rectangular bracket with mounting holes",
        "A cone with a flat top surface",
        "A torus with a circular cross-section",
        "A pyramid with a square base",
        "A cylinder with a helical groove"
    ]
    return random.choice(descriptions)

def generate_sample_data(num_samples: int = 100):
    """Generate sample data for testing."""
    # Create directories
    data_dir = Path("data")
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    
    # Create sample data
    for split_dir, num in [(train_dir, int(num_samples * 0.7)),
                          (val_dir, int(num_samples * 0.15)),
                          (test_dir, int(num_samples * 0.15))]:
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate samples
        for i in range(num):
            sample = {
                "id": f"sample_{i:04d}",
                "description": generate_sample_description(),
                "cad_features": generate_cad_features()
            }
            
            # Save sample
            with open(split_dir / f"sample_{i:04d}.json", 'w') as f:
                json.dump(sample, f, indent=2)
        
        # Create metadata
        metadata = {
            "total_samples": num,
            "categories": ["cube", "cylinder", "prism", "sphere", "nut", "bracket", "cone", "torus", "pyramid"],
            "file_formats": ["json"],
            "feature_dimension": 128
        }
        
        with open(split_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Generated {num_samples} samples:")
    print(f"- Training: {int(num_samples * 0.7)} samples")
    print(f"- Validation: {int(num_samples * 0.15)} samples")
    print(f"- Testing: {int(num_samples * 0.15)} samples")

if __name__ == "__main__":
    generate_sample_data() 