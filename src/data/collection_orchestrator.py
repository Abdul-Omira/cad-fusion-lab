"""
Modern Data Collection Orchestrator

This module provides a clean, modular data collection system that replaces
the monolithic CADDataCollector class. It coordinates feature extraction,
augmentation, and validation using the new modular architecture.

Refactored from the monolithic data_collection.py script.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
from dataclasses import dataclass

from .feature_extractors import FeatureExtractionOrchestrator
from .data_augmentation import DataAugmentationOrchestrator  
from .data_validation import DataValidationOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class CollectionConfig:
    """Configuration for data collection process."""
    output_dir: str = "data/raw"
    max_workers: int = 4
    enable_augmentation: bool = True
    enable_validation: bool = True
    max_samples_per_source: int = 1000
    validation_threshold: float = 0.8  # Minimum validation score to keep sample
    
    # Feature extraction settings
    extract_semantic_features: bool = True
    extract_technical_features: bool = True
    extract_cad_features: bool = True
    extract_geometric_features: bool = True
    
    # Augmentation settings
    augmentation_rate: float = 0.3
    max_variations_per_sample: int = 3
    
    # Validation settings
    stop_on_validation_error: bool = False
    include_warnings: bool = True


class DataCollectionOrchestrator:
    """Main orchestrator for the data collection process."""
    
    def __init__(self, config: CollectionConfig = None):
        self.config = config or CollectionConfig()
        
        # Initialize orchestrators
        self.feature_extractor = FeatureExtractionOrchestrator()
        self.data_augmenter = DataAugmentationOrchestrator(
            max_variations_per_sample=self.config.max_variations_per_sample
        )
        self.data_validator = DataValidationOrchestrator()
        
        # Configure orchestrators
        self._configure_components()
        
        # Setup output directories
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_collected': 0,
            'total_processed': 0,
            'total_valid': 0,
            'total_augmented': 0,
            'errors': [],
            'start_time': None,
            'end_time': None
        }
        
    def _configure_components(self):
        """Configure the orchestrator components based on config."""
        # Configure data validator
        self.data_validator.configure_validation(
            stop_on_first_error=self.config.stop_on_validation_error,
            include_warnings=self.config.include_warnings
        )
        
        # Configure augmentation rates
        self.data_augmenter.configure_augmenter('text', augmentation_rate=self.config.augmentation_rate)
        self.data_augmenter.configure_augmenter('features', augmentation_rate=self.config.augmentation_rate)
        self.data_augmenter.configure_augmenter('geometric', augmentation_rate=self.config.augmentation_rate)
        self.data_augmenter.configure_augmenter('cad_specific', augmentation_rate=self.config.augmentation_rate)
        
    def collect_and_process_data(self, data_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main entry point for data collection and processing.
        
        Args:
            data_sources: List of data source configurations
            
        Returns:
            Dictionary with collection statistics and results
        """
        logger.info("Starting data collection and processing...")
        self.stats['start_time'] = datetime.now()
        
        try:
            # Step 1: Collect raw data from sources
            raw_data = self._collect_from_sources(data_sources)
            self.stats['total_collected'] = len(raw_data)
            logger.info(f"Collected {len(raw_data)} raw samples")
            
            # Step 2: Process each sample (extract features, validate, augment)
            processed_data = self._process_samples(raw_data)
            self.stats['total_processed'] = len(processed_data)
            logger.info(f"Processed {len(processed_data)} samples")
            
            # Step 3: Save processed data
            self._save_processed_data(processed_data)
            
            # Step 4: Generate final statistics
            final_stats = self._generate_final_statistics(processed_data)
            
            self.stats['end_time'] = datetime.now()
            logger.info("Data collection and processing completed successfully")
            
            return final_stats
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            self.stats['errors'].append(str(e))
            self.stats['end_time'] = datetime.now()
            raise
            
    def _collect_from_sources(self, data_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Collect raw data from various sources.
        
        This is a simplified version - in practice, you'd implement
        web scraping, API calls, file reading, etc.
        """
        raw_data = []
        
        for source_config in data_sources:
            try:
                source_type = source_config.get('type', 'unknown')
                logger.info(f"Collecting from source: {source_type}")
                
                if source_type == 'mock':
                    # Generate mock data for demonstration
                    mock_samples = self._generate_mock_data(source_config.get('count', 10))
                    raw_data.extend(mock_samples)
                elif source_type == 'file':
                    # Load from file
                    file_samples = self._load_from_file(source_config['path'])
                    raw_data.extend(file_samples)
                else:
                    logger.warning(f"Unknown source type: {source_type}")
                    
            except Exception as e:
                logger.error(f"Failed to collect from source {source_config}: {e}")
                self.stats['errors'].append(f"Source collection error: {str(e)}")
                
        return raw_data
        
    def _generate_mock_data(self, count: int) -> List[Dict[str, Any]]:
        """Generate mock CAD data for testing."""
        mock_descriptions = [
            "A rectangular bracket with mounting holes for mechanical assembly",
            "Cylindrical housing with threaded connections and internal channels",
            "Complex gear mechanism with precise tooth geometry and bearing seats",
            "Automotive engine component with cooling fins and bolt patterns",
            "Aerospace structural element with weight optimization features",
            "Industrial valve body with flow control mechanisms",
            "Electronic enclosure with heat dissipation and cable management",
            "Medical device component with biocompatible surface finish",
            "Robotic joint assembly with servo motor mounting points",
            "Precision machined part with tight tolerances and surface requirements"
        ]
        
        mock_data = []
        for i in range(count):
            sample = {
                'id': f'mock_{i:04d}',
                'title': f"CAD Model {i+1}",
                'description': mock_descriptions[i % len(mock_descriptions)],
                'source': 'mock_generator',
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'category': 'mechanical',
                    'complexity': 'medium',
                    'application': 'general'
                }
            }
            mock_data.append(sample)
            
        return mock_data
        
    def _load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'samples' in data:
                return data['samples']
            else:
                logger.warning(f"Unexpected file format in {file_path}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to load from file {file_path}: {e}")
            return []
            
    def _process_samples(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process raw samples through the feature extraction, validation, and augmentation pipeline."""
        processed_samples = []
        
        for i, sample in enumerate(raw_data):
            try:
                logger.debug(f"Processing sample {i+1}/{len(raw_data)}")
                
                # Step 1: Extract features
                enhanced_sample = self._extract_features(sample)
                
                # Step 2: Validate sample
                is_valid, validation_errors = self._validate_sample(enhanced_sample)
                
                if is_valid:
                    self.stats['total_valid'] += 1
                    
                    # Step 3: Apply data augmentation if enabled
                    if self.config.enable_augmentation:
                        augmented_samples = self._augment_sample(enhanced_sample)
                        processed_samples.extend(augmented_samples)
                        self.stats['total_augmented'] += len(augmented_samples) - 1  # Exclude original
                    else:
                        processed_samples.append(enhanced_sample)
                else:
                    logger.warning(f"Sample {i} failed validation: {[str(e) for e in validation_errors]}")
                    self.stats['errors'].append(f"Sample {i} validation failed")
                    
            except Exception as e:
                logger.error(f"Failed to process sample {i}: {e}")
                self.stats['errors'].append(f"Sample {i} processing error: {str(e)}")
                
        return processed_samples
        
    def _extract_features(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from a sample."""
        enhanced_sample = sample.copy()
        
        try:
            text = sample.get('description', '')
            metadata = sample.get('metadata', {})
            
            # Extract all features
            all_features = self.feature_extractor.extract_all_features(text, metadata)
            enhanced_sample['extracted_features'] = all_features
            
            # Generate a simple feature vector for compatibility
            # In practice, this would be more sophisticated
            feature_vector = self._generate_feature_vector(all_features)
            enhanced_sample['cad_features'] = feature_vector
            
        except Exception as e:
            logger.warning(f"Feature extraction failed for sample: {e}")
            enhanced_sample['extracted_features'] = {}
            enhanced_sample['cad_features'] = [0.0] * 128  # Default feature vector
            
        return enhanced_sample
        
    def _generate_feature_vector(self, features: Dict[str, Any]) -> List[float]:
        """Generate a numerical feature vector from extracted features."""
        # This is a simplified feature vector generation
        # In practice, this would use more sophisticated encoding
        
        vector = [0.0] * 128  # Fixed-size vector
        
        try:
            # Encode semantic features
            semantic = features.get('semantic', {})
            vector[0] = semantic.get('semantic_density', 0.0)
            vector[1] = semantic.get('technical_coverage', 0.0)
            vector[2] = semantic.get('concept_complexity', 0.0)
            vector[3] = semantic.get('domain_specificity', 0.0)
            
            # Encode technical features
            technical = features.get('technical', {})
            vector[4] = 1.0 if technical.get('has_dimensions', False) else 0.0
            vector[5] = 1.0 if technical.get('has_materials', False) else 0.0
            vector[6] = 1.0 if technical.get('has_tolerances', False) else 0.0
            vector[7] = technical.get('precision_level', 0.0)
            
            # Encode CAD features
            cad_features = features.get('cad', {})
            operations = cad_features.get('operations', {})
            vector[8] = len(operations.get('basic', [])) / 4.0  # Normalize by max basic ops
            vector[9] = len(operations.get('modification', [])) / 4.0
            vector[10] = len(operations.get('pattern', [])) / 4.0
            vector[11] = cad_features.get('complexity', 0.0)
            
            # Add some random variation to fill the rest of the vector
            import random
            for i in range(12, 128):
                vector[i] = random.gauss(0, 0.1)  # Small random values
                
        except Exception as e:
            logger.warning(f"Feature vector generation failed: {e}")
            
        return vector
        
    def _validate_sample(self, sample: Dict[str, Any]) -> tuple:
        """Validate a sample using the validation orchestrator."""
        if not self.config.enable_validation:
            return True, []
            
        try:
            return self.data_validator.validate_sample(sample)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False, [f"Validation error: {str(e)}"]
            
    def _augment_sample(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Augment a sample using the augmentation orchestrator."""
        try:
            return self.data_augmenter.augment_sample(sample)
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}")
            return [sample]  # Return original if augmentation fails
            
    def _save_processed_data(self, processed_data: List[Dict[str, Any]]):
        """Save processed data to files."""
        try:
            # Split data into train/val/test
            total_samples = len(processed_data)
            train_end = int(total_samples * 0.8)
            val_end = int(total_samples * 0.9)
            
            train_data = processed_data[:train_end]
            val_data = processed_data[train_end:val_end]
            test_data = processed_data[val_end:]
            
            # Save splits
            splits = {
                'train': train_data,
                'val': val_data,
                'test': test_data
            }
            
            for split_name, split_data in splits.items():
                split_dir = self.output_dir / split_name
                split_dir.mkdir(exist_ok=True)
                
                # Save individual samples
                for i, sample in enumerate(split_data):
                    sample_file = split_dir / f"sample_{i:06d}.json"
                    with open(sample_file, 'w') as f:
                        json.dump(sample, f, indent=2)
                        
                # Save metadata
                metadata = {
                    'total_samples': len(split_data),
                    'split': split_name,
                    'timestamp': datetime.now().isoformat(),
                    'config': self.config.__dict__
                }
                
                metadata_file = split_dir / 'metadata.json'
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
            logger.info(f"Saved {len(processed_data)} processed samples to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            raise
            
    def _generate_final_statistics(self, processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate final collection statistics."""
        duration = None
        if self.stats['start_time'] and self.stats['end_time']:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
        final_stats = {
            'collection_summary': {
                'total_collected': self.stats['total_collected'],
                'total_processed': self.stats['total_processed'],
                'total_valid': self.stats['total_valid'],
                'total_augmented': self.stats['total_augmented'],
                'success_rate': self.stats['total_valid'] / max(self.stats['total_collected'], 1),
                'augmentation_ratio': self.stats['total_augmented'] / max(self.stats['total_valid'], 1),
                'duration_seconds': duration,
                'errors_count': len(self.stats['errors'])
            },
            'data_splits': {
                'train': len([d for d in processed_data[:int(len(processed_data) * 0.8)]]),
                'val': len([d for d in processed_data[int(len(processed_data) * 0.8):int(len(processed_data) * 0.9)]]),
                'test': len([d for d in processed_data[int(len(processed_data) * 0.9):]])
            },
            'config_used': self.config.__dict__,
            'errors': self.stats['errors'][:10],  # Show only first 10 errors
            'timestamp': datetime.now().isoformat()
        }
        
        # Save statistics
        stats_file = self.output_dir / 'collection_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2)
            
        return final_stats


# Example usage function
def run_data_collection_example():
    """Example of how to use the new data collection system."""
    
    # Configure collection
    config = CollectionConfig(
        output_dir="data/processed",
        enable_augmentation=True,
        enable_validation=True,
        augmentation_rate=0.4,
        max_variations_per_sample=3
    )
    
    # Define data sources
    data_sources = [
        {
            'type': 'mock',
            'count': 50
        }
    ]
    
    # Run collection
    orchestrator = DataCollectionOrchestrator(config)
    results = orchestrator.collect_and_process_data(data_sources)
    
    print(f"Collection completed!")
    print(f"Collected: {results['collection_summary']['total_collected']} samples")
    print(f"Processed: {results['collection_summary']['total_processed']} samples")
    print(f"Success rate: {results['collection_summary']['success_rate']:.2%}")
    
    return results


if __name__ == "__main__":
    run_data_collection_example()