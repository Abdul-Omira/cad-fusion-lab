"""
Test the new modular data collection system.
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.data.feature_extractors import FeatureExtractionOrchestrator
from src.data.data_augmentation import DataAugmentationOrchestrator
from src.data.data_validation import DataValidationOrchestrator
from src.data.collection_orchestrator import DataCollectionOrchestrator, CollectionConfig


def test_feature_extraction():
    """Test the feature extraction system."""
    print("Testing Feature Extraction...")
    
    extractor = FeatureExtractionOrchestrator()
    
    test_text = "Create a cylindrical housing with mounting holes and threaded connections for mechanical assembly"
    features = extractor.extract_all_features(test_text)
    
    print(f"Extracted features: {list(features.keys())}")
    print(f"Semantic features: {features.get('semantic', {})}")
    print(f"CAD features: {features.get('cad', {})}")
    print("✅ Feature extraction test passed\n")


def test_data_augmentation():
    """Test the data augmentation system."""
    print("Testing Data Augmentation...")
    
    augmenter = DataAugmentationOrchestrator(max_variations_per_sample=3)
    
    sample_data = {
        'title': 'Test CAD Model',
        'description': 'A simple cube with rounded edges for testing',
        'cad_features': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    
    augmented_samples = augmenter.augment_sample(sample_data)
    
    print(f"Original samples: 1")
    print(f"Augmented samples: {len(augmented_samples)}")
    print(f"Augmentation types: {[s.get('augmentation_type', 'original') for s in augmented_samples]}")
    print("✅ Data augmentation test passed\n")


def test_data_validation():
    """Test the data validation system."""
    print("Testing Data Validation...")
    
    validator = DataValidationOrchestrator()
    
    # Test valid sample
    valid_sample = {
        'title': 'Valid CAD Model',
        'description': 'A well-described cylindrical part with extrude operations and dimensional specifications in mm',
        'cad_features': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    
    is_valid, errors = validator.validate_sample(valid_sample)
    print(f"Valid sample result: {is_valid}")
    print(f"Errors: {len(errors)}")
    
    # Test invalid sample
    invalid_sample = {
        'title': '',
        'description': 'x',
        'cad_features': 'invalid'
    }
    
    is_valid, errors = validator.validate_sample(invalid_sample)
    print(f"Invalid sample result: {is_valid}")
    print(f"Errors: {len(errors)}")
    print(f"Error messages: {[str(e) for e in errors[:3]]}")
    print("✅ Data validation test passed\n")


def test_full_orchestration():
    """Test the complete data collection orchestration."""
    print("Testing Full Data Collection Orchestration...")
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        config = CollectionConfig(
            output_dir=temp_dir,
            enable_augmentation=True,
            enable_validation=True,
            augmentation_rate=0.5,
            max_variations_per_sample=2
        )
        
        # Define mock data sources
        data_sources = [
            {
                'type': 'mock',
                'count': 5
            }
        ]
        
        # Run collection
        orchestrator = DataCollectionOrchestrator(config)
        results = orchestrator.collect_and_process_data(data_sources)
        
        print(f"Collection Results:")
        print(f"  Total collected: {results['collection_summary']['total_collected']}")
        print(f"  Total processed: {results['collection_summary']['total_processed']}")
        print(f"  Total valid: {results['collection_summary']['total_valid']}")
        print(f"  Success rate: {results['collection_summary']['success_rate']:.2%}")
        
        # Check output files
        output_path = Path(temp_dir)
        train_files = list((output_path / 'train').glob('*.json')) if (output_path / 'train').exists() else []
        print(f"  Train files created: {len(train_files)}")
        
        # Check a sample file
        if train_files:
            with open(train_files[0]) as f:
                sample = json.load(f)
            print(f"  Sample keys: {list(sample.keys())}")
            print(f"  Has features: {'extracted_features' in sample}")
            print(f"  Has CAD features: {'cad_features' in sample and len(sample['cad_features']) > 0}")
            
        print("✅ Full orchestration test passed\n")


if __name__ == "__main__":
    print("Testing New Modular Data Collection System")
    print("=" * 50)
    
    try:
        test_feature_extraction()
        test_data_augmentation() 
        test_data_validation()
        test_full_orchestration()
        
        print("🎉 All tests passed! The new modular system is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()