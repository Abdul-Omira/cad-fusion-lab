"""
Feature Extraction Module for CAD Data Collection

This module provides specialized feature extractors for different aspects of CAD data:
- Semantic features from text descriptions
- Technical features (materials, dimensions, etc.)
- Geometric features (shapes, topology)
- CAD-specific features (operations, parameters)
- Image features (visual characteristics)

Refactored from the monolithic data_collection.py script.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Set
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


class BaseFeatureExtractor(ABC):
    """Abstract base class for all feature extractors."""
    
    def __init__(self):
        self.name = self.__class__.__name__
        
    @abstractmethod
    def extract_features(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract features from input text and metadata."""
        pass
        
    def _safe_extract(self, extraction_func, *args, **kwargs) -> Any:
        """Safely execute an extraction function with error handling."""
        try:
            return extraction_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Feature extraction failed in {self.name}: {e}")
            return {}


class SemanticFeatureExtractor(BaseFeatureExtractor):
    """Extract semantic features from text descriptions."""
    
    def __init__(self):
        super().__init__()
        # Initialize semantic analysis tools
        self.technical_terms = {
            'materials', 'properties', 'specifications', 'requirements',
            'performance', 'applications', 'standards', 'certifications'
        }
        
    def extract_features(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract semantic features from text."""
        return {
            'semantic_density': self._calculate_semantic_density(text),
            'technical_coverage': self._calculate_technical_coverage(text),
            'concept_complexity': self._calculate_concept_complexity(text),
            'domain_specificity': self._calculate_domain_specificity(text)
        }
        
    def _calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic density of the text."""
        words = text.lower().split()
        if not words:
            return 0.0
        
        # Count technical terms
        technical_count = sum(1 for word in words if any(term in word for term in self.technical_terms))
        return min(technical_count / len(words), 1.0)
    
    def _calculate_technical_coverage(self, text: str) -> float:
        """Calculate coverage of technical terms."""
        text_lower = text.lower()
        covered_terms = sum(1 for term in self.technical_terms if term in text_lower)
        return covered_terms / len(self.technical_terms) if self.technical_terms else 0.0
        
    def _calculate_concept_complexity(self, text: str) -> float:
        """Calculate concept complexity based on sentence structure."""
        sentences = re.split(r'[.!?]+', text)
        if not sentences:
            return 0.0
            
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        # Normalize to 0-1 scale (assuming max reasonable sentence length of 50 words)
        return min(avg_sentence_length / 50.0, 1.0)
        
    def _calculate_domain_specificity(self, text: str) -> float:
        """Calculate how domain-specific the text is."""
        cad_terms = {
            'extrude', 'revolve', 'sweep', 'loft', 'fillet', 'chamfer',
            'sketch', 'constraint', 'dimension', 'assembly', 'part', 'feature'
        }
        
        text_lower = text.lower()
        cad_term_count = sum(1 for term in cad_terms if term in text_lower)
        return min(cad_term_count / 5.0, 1.0)  # Normalize assuming 5+ terms indicates high specificity


class TechnicalFeatureExtractor(BaseFeatureExtractor):
    """Extract technical specifications and properties."""
    
    def __init__(self):
        super().__init__()
        self.dimension_patterns = [
            r'(\d+\.?\d*)\s*(mm|cm|m|in|ft)',  # Dimensions with units
            r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)',   # Dimension ratios
            r'diameter\s*:?\s*(\d+\.?\d*)',     # Diameter specifications
        ]
        
    def extract_features(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract technical features from text."""
        return {
            'has_dimensions': self._extract_dimensions(text),
            'has_materials': self._extract_materials(text),
            'has_tolerances': self._extract_tolerances(text),
            'has_specifications': self._extract_specifications(text),
            'precision_level': self._calculate_precision_level(text)
        }
        
    def _extract_dimensions(self, text: str) -> bool:
        """Check if text contains dimensional information."""
        for pattern in self.dimension_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
        
    def _extract_materials(self, text: str) -> bool:
        """Check if text mentions materials."""
        materials = {'steel', 'aluminum', 'plastic', 'carbon', 'titanium', 'ceramic'}
        text_lower = text.lower()
        return any(material in text_lower for material in materials)
        
    def _extract_tolerances(self, text: str) -> bool:
        """Check if text mentions tolerances."""
        tolerance_patterns = [r'[+\-±]\s*\d+\.?\d*', r'tolerance', r'±']
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in tolerance_patterns)
        
    def _extract_specifications(self, text: str) -> bool:
        """Check if text contains specifications."""
        spec_terms = {'specification', 'standard', 'requirement', 'property'}
        text_lower = text.lower()
        return any(term in text_lower for term in spec_terms)
        
    def _calculate_precision_level(self, text: str) -> float:
        """Calculate precision level based on technical detail."""
        precision_indicators = [
            'precision', 'accurate', 'exact', 'tight tolerance',
            'high quality', 'specification', 'standard'
        ]
        text_lower = text.lower()
        precision_score = sum(1 for indicator in precision_indicators if indicator in text_lower)
        return min(precision_score / 3.0, 1.0)  # Normalize to 0-1


class CADFeatureExtractor(BaseFeatureExtractor):
    """Extract CAD-specific features and operations."""
    
    def __init__(self):
        super().__init__()
        self.cad_operations = {
            'basic': {'extrude', 'revolve', 'sweep', 'loft'},
            'modification': {'fillet', 'chamfer', 'shell', 'draft'},
            'pattern': {'linear', 'circular', 'mirror', 'array'},
            'assembly': {'mate', 'constraint', 'joint', 'connection'}
        }
        
    def extract_features(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract CAD-specific features."""
        return {
            'operations': self._extract_operations(text),
            'complexity': self._calculate_complexity(text),
            'feature_types': self._extract_feature_types(text),
            'modeling_approach': self._identify_modeling_approach(text)
        }
        
    def _extract_operations(self, text: str) -> Dict[str, List[str]]:
        """Extract CAD operations mentioned in text."""
        text_lower = text.lower()
        found_ops = {}
        
        for category, operations in self.cad_operations.items():
            found = [op for op in operations if op in text_lower]
            if found:
                found_ops[category] = found
                
        return found_ops
        
    def _calculate_complexity(self, text: str) -> float:
        """Calculate CAD model complexity based on operations."""
        text_lower = text.lower()
        total_ops = sum(1 for ops in self.cad_operations.values() 
                       for op in ops if op in text_lower)
        
        # Normalize complexity (assuming 10+ operations indicates high complexity)
        return min(total_ops / 10.0, 1.0)
        
    def _extract_feature_types(self, text: str) -> List[str]:
        """Extract types of CAD features mentioned."""
        feature_types = {
            'hole', 'slot', 'boss', 'rib', 'web', 'flange',
            'groove', 'thread', 'pocket', 'pad', 'cut'
        }
        
        text_lower = text.lower()
        return [feature for feature in feature_types if feature in text_lower]
        
    def _identify_modeling_approach(self, text: str) -> str:
        """Identify the modeling approach used."""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['parametric', 'feature-based', 'history']):
            return 'parametric'
        elif any(term in text_lower for term in ['surface', 'nurbs', 'bezier']):
            return 'surface'
        elif any(term in text_lower for term in ['mesh', 'polygon', 'triangulated']):
            return 'mesh'
        else:
            return 'unknown'


class GeometricFeatureExtractor(BaseFeatureExtractor):
    """Extract geometric properties and characteristics."""
    
    def __init__(self):
        super().__init__()
        self.geometric_shapes = {
            'primitive': {'cube', 'sphere', 'cylinder', 'cone', 'torus'},
            'complex': {'helical', 'spline', 'curved', 'organic'},
            'architectural': {'beam', 'column', 'slab', 'wall'},
            'mechanical': {'gear', 'bearing', 'shaft', 'housing'}
        }
        
    def extract_features(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract geometric features."""
        return {
            'shapes': self._extract_shapes(text),
            'geometry_type': self._classify_geometry_type(text),
            'symmetry': self._detect_symmetry(text),
            'topology': self._analyze_topology(text)
        }
        
    def _extract_shapes(self, text: str) -> Dict[str, List[str]]:
        """Extract geometric shapes mentioned in text."""
        text_lower = text.lower()
        found_shapes = {}
        
        for category, shapes in self.geometric_shapes.items():
            found = [shape for shape in shapes if shape in text_lower]
            if found:
                found_shapes[category] = found
                
        return found_shapes
        
    def _classify_geometry_type(self, text: str) -> str:
        """Classify the type of geometry."""
        text_lower = text.lower()
        
        if any(shape in text_lower for shapes in self.geometric_shapes.values() for shape in shapes):
            if any(shape in text_lower for shape in self.geometric_shapes['primitive']):
                return 'primitive'
            elif any(shape in text_lower for shape in self.geometric_shapes['complex']):
                return 'complex'
            else:
                return 'composite'
        return 'unclassified'
        
    def _detect_symmetry(self, text: str) -> Dict[str, bool]:
        """Detect symmetry characteristics."""
        text_lower = text.lower()
        return {
            'symmetric': 'symmetric' in text_lower or 'symmetry' in text_lower,
            'axial': 'axial' in text_lower or 'rotational' in text_lower,
            'radial': 'radial' in text_lower or 'circular' in text_lower,
            'bilateral': 'bilateral' in text_lower or 'mirror' in text_lower
        }
        
    def _analyze_topology(self, text: str) -> Dict[str, Any]:
        """Analyze topological characteristics."""
        text_lower = text.lower()
        return {
            'closed': 'closed' in text_lower or 'sealed' in text_lower,
            'open': 'open' in text_lower or 'hollow' in text_lower,
            'connected': 'connected' in text_lower or 'joined' in text_lower,
            'manifold': 'manifold' in text_lower or 'watertight' in text_lower
        }


class FeatureExtractionOrchestrator:
    """Orchestrates multiple feature extractors for comprehensive analysis."""
    
    def __init__(self):
        self.extractors = {
            'semantic': SemanticFeatureExtractor(),
            'technical': TechnicalFeatureExtractor(),
            'cad': CADFeatureExtractor(),
            'geometric': GeometricFeatureExtractor()
        }
        
    def extract_all_features(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract features using all available extractors."""
        all_features = {}
        
        for name, extractor in self.extractors.items():
            try:
                features = extractor.extract_features(text, metadata)
                all_features[name] = features
            except Exception as e:
                logger.error(f"Feature extraction failed for {name}: {e}")
                all_features[name] = {}
                
        return all_features
        
    def add_extractor(self, name: str, extractor: BaseFeatureExtractor):
        """Add a custom feature extractor."""
        self.extractors[name] = extractor
        
    def remove_extractor(self, name: str):
        """Remove a feature extractor."""
        if name in self.extractors:
            del self.extractors[name]