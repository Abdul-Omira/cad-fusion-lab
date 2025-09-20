"""
Data Augmentation Module for CAD Data Collection

This module provides various data augmentation techniques:
- Text augmentation (synonym replacement, paraphrasing)
- Feature augmentation (noise injection, transformations)
- Cross-lingual augmentation (translation-based)
- Multimodal augmentation (cross-modal consistency)

Refactored from the monolithic data_collection.py script.
"""

import random
import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


class BaseAugmenter(ABC):
    """Abstract base class for all data augmenters."""
    
    def __init__(self, augmentation_rate: float = 0.3):
        self.name = self.__class__.__name__
        self.augmentation_rate = augmentation_rate
        
    @abstractmethod
    def augment(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate augmented versions of the input data."""
        pass
        
    def _should_augment(self) -> bool:
        """Randomly determine if augmentation should be applied."""
        return random.random() < self.augmentation_rate


class TextAugmenter(BaseAugmenter):
    """Augment text descriptions with various techniques."""
    
    def __init__(self, augmentation_rate: float = 0.3, max_variations: int = 3):
        super().__init__(augmentation_rate)
        self.max_variations = max_variations
        self.synonyms = {
            'create': ['generate', 'produce', 'make', 'build', 'construct'],
            'design': ['plan', 'draft', 'blueprint', 'conceive', 'develop'],
            'model': ['representation', 'prototype', 'design', 'structure'],
            'part': ['component', 'element', 'piece', 'section'],
            'assembly': ['construction', 'build', 'structure', 'unit'],
            'feature': ['characteristic', 'attribute', 'element', 'property'],
            'dimension': ['measurement', 'size', 'extent', 'proportion'],
            'material': ['substance', 'matter', 'compound', 'medium']
        }
        
    def augment(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate text variations of the input data."""
        if not self._should_augment():
            return [data]
            
        variations = [data.copy()]  # Include original
        text = data.get('description', '')
        
        if not text:
            return variations
            
        # Generate different types of variations
        augmentation_methods = [
            self._synonym_replacement,
            self._sentence_restructure,
            self._technical_paraphrase,
            self._style_variation
        ]
        
        for method in augmentation_methods[:self.max_variations]:
            try:
                augmented_text = method(text)
                if augmented_text != text:
                    augmented_data = data.copy()
                    augmented_data['description'] = augmented_text
                    augmented_data['augmentation_type'] = method.__name__
                    variations.append(augmented_data)
            except Exception as e:
                logger.warning(f"Text augmentation failed with {method.__name__}: {e}")
                
        return variations
        
    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms."""
        words = text.split()
        augmented_words = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            if word_lower in self.synonyms and random.random() < 0.3:
                synonym = random.choice(self.synonyms[word_lower])
                # Preserve original capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                augmented_words.append(synonym)
            else:
                augmented_words.append(word)
                
        return ' '.join(augmented_words)
        
    def _sentence_restructure(self, text: str) -> str:
        """Restructure sentences while preserving meaning."""
        sentences = text.split('. ')
        if len(sentences) < 2:
            return text
            
        # Simple restructuring: reverse sentence order occasionally
        if random.random() < 0.5 and len(sentences) == 2:
            return f"{sentences[1]}. {sentences[0]}"
            
        return text
        
    def _technical_paraphrase(self, text: str) -> str:
        """Create technical paraphrases."""
        # Simple technical transformations
        replacements = {
            'a cylinder': 'a cylindrical shape',
            'a cube': 'a cubic form',
            'a sphere': 'a spherical object',
            'with holes': 'featuring apertures',
            'rounded edges': 'filleted corners',
            'sharp corners': 'angular vertices'
        }
        
        result = text
        for original, replacement in replacements.items():
            if original in result.lower():
                result = result.replace(original, replacement)
                
        return result
        
    def _style_variation(self, text: str) -> str:
        """Create style variations (formal/informal)."""
        # Simple style transformations
        formal_replacements = {
            "it's": "it is",
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not"
        }
        
        result = text
        for informal, formal in formal_replacements.items():
            result = result.replace(informal, formal)
            
        return result


class FeatureAugmenter(BaseAugmenter):
    """Augment feature vectors and numerical data."""
    
    def __init__(self, augmentation_rate: float = 0.3, noise_level: float = 0.1):
        super().__init__(augmentation_rate)
        self.noise_level = noise_level
        
    def augment(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate feature variations of the input data."""
        if not self._should_augment():
            return [data]
            
        variations = [data.copy()]
        
        # Augment numerical features
        if 'cad_features' in data:
            try:
                augmented_features = self._add_gaussian_noise(data['cad_features'])
                augmented_data = data.copy()
                augmented_data['cad_features'] = augmented_features
                augmented_data['augmentation_type'] = 'feature_noise'
                variations.append(augmented_data)
            except Exception as e:
                logger.warning(f"Feature augmentation failed: {e}")
                
        return variations
        
    def _add_gaussian_noise(self, features: List[float]) -> List[float]:
        """Add Gaussian noise to feature vector."""
        features_array = np.array(features)
        noise = np.random.normal(0, self.noise_level, features_array.shape)
        augmented_features = features_array + noise
        
        # Ensure features remain in reasonable bounds
        augmented_features = np.clip(augmented_features, -10, 10)
        
        return augmented_features.tolist()


class GeometricAugmenter(BaseAugmenter):
    """Augment geometric descriptions and properties."""
    
    def __init__(self, augmentation_rate: float = 0.3):
        super().__init__(augmentation_rate)
        self.geometric_variations = {
            'scale': ['small', 'medium', 'large', 'tiny', 'huge'],
            'shape_modifiers': ['smooth', 'rough', 'textured', 'polished'],
            'orientation': ['horizontal', 'vertical', 'angled', 'tilted'],
            'precision': ['precise', 'approximate', 'exact', 'rough']
        }
        
    def augment(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate geometric variations of the input data."""
        if not self._should_augment():
            return [data]
            
        variations = [data.copy()]
        description = data.get('description', '')
        
        if description:
            augmented_description = self._add_geometric_modifiers(description)
            if augmented_description != description:
                augmented_data = data.copy()
                augmented_data['description'] = augmented_description
                augmented_data['augmentation_type'] = 'geometric_variation'
                variations.append(augmented_data)
                
        return variations
        
    def _add_geometric_modifiers(self, description: str) -> str:
        """Add geometric modifiers to the description."""
        # Add shape modifiers
        if 'surface' in description.lower() and random.random() < 0.5:
            modifier = random.choice(self.geometric_variations['shape_modifiers'])
            description = description.replace('surface', f'{modifier} surface')
            
        # Add scale information
        if any(shape in description.lower() for shape in ['cube', 'sphere', 'cylinder']) and random.random() < 0.3:
            scale = random.choice(self.geometric_variations['scale'])
            description = f"A {scale} " + description.lower()
            
        return description


class CADSpecificAugmenter(BaseAugmenter):
    """Augment CAD-specific terminology and operations."""
    
    def __init__(self, augmentation_rate: float = 0.3):
        super().__init__(augmentation_rate)
        self.operation_synonyms = {
            'extrude': ['extend', 'protrude', 'project'],
            'revolve': ['rotate', 'spin', 'turn'],
            'sweep': ['trace', 'follow', 'guide'],
            'fillet': ['round', 'curve', 'smooth'],
            'chamfer': ['bevel', 'angle', 'cut'],
            'pattern': ['repeat', 'array', 'duplicate']
        }
        
    def augment(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate CAD-specific variations."""
        if not self._should_augment():
            return [data]
            
        variations = [data.copy()]
        description = data.get('description', '')
        
        if description:
            augmented_description = self._replace_cad_operations(description)
            if augmented_description != description:
                augmented_data = data.copy()
                augmented_data['description'] = augmented_description
                augmented_data['augmentation_type'] = 'cad_terminology'
                variations.append(augmented_data)
                
        return variations
        
    def _replace_cad_operations(self, description: str) -> str:
        """Replace CAD operations with synonyms."""
        result = description
        for operation, synonyms in self.operation_synonyms.items():
            if operation in result.lower() and random.random() < 0.4:
                synonym = random.choice(synonyms)
                result = result.replace(operation, synonym)
                
        return result


class DataAugmentationOrchestrator:
    """Orchestrates multiple data augmenters for comprehensive augmentation."""
    
    def __init__(self, max_variations_per_sample: int = 5):
        self.max_variations = max_variations_per_sample
        self.augmenters = {
            'text': TextAugmenter(augmentation_rate=0.4, max_variations=2),
            'features': FeatureAugmenter(augmentation_rate=0.3),
            'geometric': GeometricAugmenter(augmentation_rate=0.3),
            'cad_specific': CADSpecificAugmenter(augmentation_rate=0.3)
        }
        
    def augment_sample(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply all augmentation techniques to a single sample."""
        all_variations = [data.copy()]  # Start with original
        
        # Apply each augmenter
        for augmenter_name, augmenter in self.augmenters.items():
            try:
                # Apply augmentation to original data
                variations = augmenter.augment(data)
                
                # Add new variations (skip the first one which is the original)
                for variation in variations[1:]:
                    if len(all_variations) < self.max_variations:
                        all_variations.append(variation)
                    else:
                        break
                        
            except Exception as e:
                logger.error(f"Augmentation failed with {augmenter_name}: {e}")
                
        return all_variations
        
    def augment_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply augmentation to a batch of samples."""
        augmented_batch = []
        
        for sample in batch:
            variations = self.augment_sample(sample)
            augmented_batch.extend(variations)
            
        return augmented_batch
        
    def configure_augmenter(self, augmenter_name: str, **kwargs):
        """Configure parameters for a specific augmenter."""
        if augmenter_name in self.augmenters:
            for key, value in kwargs.items():
                if hasattr(self.augmenters[augmenter_name], key):
                    setattr(self.augmenters[augmenter_name], key, value)
                    
    def add_augmenter(self, name: str, augmenter: BaseAugmenter):
        """Add a custom augmenter."""
        self.augmenters[name] = augmenter
        
    def remove_augmenter(self, name: str):
        """Remove an augmenter."""
        if name in self.augmenters:
            del self.augmenters[name]
            
    def get_augmentation_stats(self, original_count: int, augmented_count: int) -> Dict[str, Any]:
        """Get statistics about the augmentation process."""
        return {
            'original_samples': original_count,
            'augmented_samples': augmented_count,
            'augmentation_ratio': augmented_count / original_count if original_count > 0 else 0,
            'augmenters_used': list(self.augmenters.keys()),
            'max_variations_per_sample': self.max_variations
        }