"""
Data Validation Module for CAD Data Collection

This module provides comprehensive validation for collected CAD data:
- General data validation (completeness, format, quality)
- CAD-specific validation (technical accuracy, feasibility)
- Image validation (quality, relevance, format)
- Domain-specific validation (industry standards, requirements)

Refactored from the monolithic data_collection.py script.
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional, Set
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError:
    """Represents a validation error with severity and context."""
    
    def __init__(self, message: str, severity: str = "error", field: str = None):
        self.message = message
        self.severity = severity  # "error", "warning", "info"
        self.field = field
        self.timestamp = None  # Could add timestamp if needed
        
    def __str__(self):
        field_info = f" (field: {self.field})" if self.field else ""
        return f"[{self.severity.upper()}] {self.message}{field_info}"


class BaseValidator(ABC):
    """Abstract base class for all validators."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.validation_rules = {}
        
    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """Validate data and return success status and errors."""
        pass
        
    def _create_error(self, message: str, severity: str = "error", field: str = None) -> ValidationError:
        """Create a validation error."""
        return ValidationError(message, severity, field)


class BasicDataValidator(BaseValidator):
    """Validates basic data structure and required fields."""
    
    def __init__(self):
        super().__init__()
        self.required_fields = ['title', 'description']
        self.optional_fields = ['cad_features', 'metadata', 'images']
        self.min_description_length = 10
        self.max_description_length = 5000
        
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """Validate basic data requirements."""
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in data:
                errors.append(self._create_error(f"Missing required field: {field}", field=field))
            elif not data[field] or (isinstance(data[field], str) and not data[field].strip()):
                errors.append(self._create_error(f"Empty required field: {field}", field=field))
                
        # Validate description length
        if 'description' in data and isinstance(data['description'], str):
            desc_length = len(data['description'].strip())
            if desc_length < self.min_description_length:
                errors.append(self._create_error(
                    f"Description too short ({desc_length} chars, minimum {self.min_description_length})",
                    severity="warning",
                    field="description"
                ))
            elif desc_length > self.max_description_length:
                errors.append(self._create_error(
                    f"Description too long ({desc_length} chars, maximum {self.max_description_length})",
                    field="description"
                ))
                
        # Validate data types
        if 'title' in data and not isinstance(data['title'], str):
            errors.append(self._create_error("Title must be a string", field="title"))
            
        if 'description' in data and not isinstance(data['description'], str):
            errors.append(self._create_error("Description must be a string", field="description"))
            
        # Validate feature vector if present
        if 'cad_features' in data:
            feature_errors = self._validate_features(data['cad_features'])
            errors.extend(feature_errors)
            
        return len(errors) == 0, errors
        
    def _validate_features(self, features: Any) -> List[ValidationError]:
        """Validate CAD feature vector."""
        errors = []
        
        if not isinstance(features, (list, tuple, np.ndarray)):
            errors.append(self._create_error(
                "CAD features must be a list, tuple, or array",
                field="cad_features"
            ))
            return errors
            
        try:
            features_array = np.array(features, dtype=float)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(features_array)):
                errors.append(self._create_error(
                    "CAD features contain NaN values",
                    severity="warning",
                    field="cad_features"
                ))
                
            if np.any(np.isinf(features_array)):
                errors.append(self._create_error(
                    "CAD features contain infinite values",
                    severity="warning",
                    field="cad_features"
                ))
                
            # Check reasonable bounds
            if np.any(np.abs(features_array) > 1000):
                errors.append(self._create_error(
                    "Some CAD feature values are extremely large",
                    severity="warning",
                    field="cad_features"
                ))
                
        except (ValueError, TypeError):
            errors.append(self._create_error(
                "CAD features cannot be converted to numeric array",
                field="cad_features"
            ))
            
        return errors


class CADSpecificValidator(BaseValidator):
    """Validates CAD-specific content and technical accuracy."""
    
    def __init__(self):
        super().__init__()
        self.cad_terms = {
            'operations': {'extrude', 'revolve', 'sweep', 'loft', 'fillet', 'chamfer'},
            'features': {'hole', 'slot', 'boss', 'rib', 'pocket', 'cut'},
            'constraints': {'dimension', 'angle', 'parallel', 'perpendicular', 'tangent'},
            'materials': {'steel', 'aluminum', 'plastic', 'carbon', 'titanium'}
        }
        self.min_cad_terms = 2  # Minimum CAD terms for technical relevance
        
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """Validate CAD-specific content."""
        errors = []
        
        description = data.get('description', '').lower()
        title = data.get('title', '').lower()
        full_text = f"{title} {description}".lower()
        
        # Check for CAD relevance
        cad_term_count = sum(
            1 for terms in self.cad_terms.values() 
            for term in terms 
            if term in full_text
        )
        
        if cad_term_count < self.min_cad_terms:
            errors.append(self._create_error(
                f"Content appears to have low CAD relevance (only {cad_term_count} CAD terms found)",
                severity="warning",
                field="description"
            ))
            
        # Check for technical feasibility
        feasibility_errors = self._check_technical_feasibility(full_text)
        errors.extend(feasibility_errors)
        
        # Validate geometric consistency
        geometry_errors = self._check_geometric_consistency(full_text)
        errors.extend(geometry_errors)
        
        return len([e for e in errors if e.severity == "error"]) == 0, errors
        
    def _check_technical_feasibility(self, text: str) -> List[ValidationError]:
        """Check for technically infeasible combinations."""
        errors = []
        
        # Check for impossible material-operation combinations
        if 'extrude' in text and 'liquid' in text:
            errors.append(self._create_error(
                "Extrusion operation with liquid material may not be feasible",
                severity="warning"
            ))
            
        # Check for contradictory operations
        if 'solid' in text and 'hollow' in text:
            # This could be valid (hollow solid), so just a warning
            errors.append(self._create_error(
                "Description contains potentially contradictory terms (solid and hollow)",
                severity="info"
            ))
            
        return errors
        
    def _check_geometric_consistency(self, text: str) -> List[ValidationError]:
        """Check for geometric consistency in descriptions."""
        errors = []
        
        # Check for dimensional consistency
        dimension_pattern = r'(\d+\.?\d*)\s*(mm|cm|m|in|ft)'
        dimensions = re.findall(dimension_pattern, text)
        
        if len(dimensions) > 1:
            try:
                # Convert all to mm for comparison
                mm_values = []
                for value, unit in dimensions:
                    val = float(value)
                    if unit == 'cm':
                        val *= 10
                    elif unit == 'm':
                        val *= 1000
                    elif unit == 'in':
                        val *= 25.4
                    elif unit == 'ft':
                        val *= 304.8
                    mm_values.append(val)
                
                # Check for unrealistic ratios
                if max(mm_values) / min(mm_values) > 10000:
                    errors.append(self._create_error(
                        "Extremely large dimensional ratios may indicate errors",
                        severity="warning"
                    ))
                    
            except ValueError:
                pass  # Skip if dimensions can't be parsed
                
        return errors


class QualityValidator(BaseValidator):
    """Validates content quality and readability."""
    
    def __init__(self):
        super().__init__()
        self.min_readability = 0.1  # Minimum readability threshold
        self.max_repetition_ratio = 0.3  # Maximum ratio of repeated content
        
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """Validate content quality."""
        errors = []
        
        description = data.get('description', '')
        title = data.get('title', '')
        
        if description:
            # Check readability
            readability = self._calculate_readability(description)
            if readability < self.min_readability:
                errors.append(self._create_error(
                    f"Low readability score ({readability:.2f})",
                    severity="warning",
                    field="description"
                ))
                
            # Check for excessive repetition
            repetition_ratio = self._calculate_repetition(description)
            if repetition_ratio > self.max_repetition_ratio:
                errors.append(self._create_error(
                    f"High repetition ratio ({repetition_ratio:.2f})",
                    severity="warning",
                    field="description"
                ))
                
            # Check language quality
            language_errors = self._check_language_quality(description)
            errors.extend(language_errors)
            
        return len([e for e in errors if e.severity == "error"]) == 0, errors
        
    def _calculate_readability(self, text: str) -> float:
        """Calculate a simple readability score."""
        if not text.strip():
            return 0.0
            
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        
        if sentences == 0 or words == 0:
            return 0.0
            
        # Simple readability metric (words per sentence, normalized)
        avg_sentence_length = words / sentences
        # Normalize to 0-1 scale (assuming 20 words per sentence is optimal)
        readability = min(1.0, 20.0 / (avg_sentence_length + 10))
        
        return readability
        
    def _calculate_repetition(self, text: str) -> float:
        """Calculate repetition ratio in text."""
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
            
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
            
        total_repetitions = sum(count - 1 for count in word_counts.values() if count > 1)
        return total_repetitions / len(words)
        
    def _check_language_quality(self, text: str) -> List[ValidationError]:
        """Check for language quality issues."""
        errors = []
        
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:\-()]', text)) / len(text)
        if special_char_ratio > 0.1:
            errors.append(self._create_error(
                "High ratio of special characters",
                severity="warning",
                field="description"
            ))
            
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', text)
        incomplete_sentences = sum(1 for s in sentences if len(s.strip().split()) < 3)
        if incomplete_sentences / max(len(sentences), 1) > 0.5:
            errors.append(self._create_error(
                "Many incomplete sentences detected",
                severity="warning",
                field="description"
            ))
            
        return errors


class ImageValidator(BaseValidator):
    """Validates image data if present."""
    
    def __init__(self):
        super().__init__()
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        self.min_resolution = (64, 64)  # Minimum width, height
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """Validate image data."""
        errors = []
        
        images = data.get('images', [])
        if not images:
            return True, errors  # No images to validate
            
        for i, image_info in enumerate(images):
            image_errors = self._validate_single_image(image_info, i)
            errors.extend(image_errors)
            
        return len([e for e in errors if e.severity == "error"]) == 0, errors
        
    def _validate_single_image(self, image_info: Dict[str, Any], index: int) -> List[ValidationError]:
        """Validate a single image."""
        errors = []
        field_prefix = f"images[{index}]"
        
        # Check required fields
        if 'url' not in image_info and 'path' not in image_info:
            errors.append(self._create_error(
                "Image must have either 'url' or 'path'",
                field=field_prefix
            ))
            return errors
            
        # Validate file format
        image_path = image_info.get('path', image_info.get('url', ''))
        if image_path:
            file_ext = Path(image_path).suffix.lower()
            if file_ext not in self.supported_formats:
                errors.append(self._create_error(
                    f"Unsupported image format: {file_ext}",
                    severity="warning",
                    field=f"{field_prefix}.format"
                ))
                
        # Validate image metadata if present
        if 'width' in image_info and 'height' in image_info:
            try:
                width = int(image_info['width'])
                height = int(image_info['height'])
                
                if width < self.min_resolution[0] or height < self.min_resolution[1]:
                    errors.append(self._create_error(
                        f"Image resolution too low: {width}x{height}",
                        severity="warning",
                        field=f"{field_prefix}.resolution"
                    ))
                    
            except (ValueError, TypeError):
                errors.append(self._create_error(
                    "Invalid image dimensions",
                    field=f"{field_prefix}.dimensions"
                ))
                
        # Validate file size if present
        if 'file_size' in image_info:
            try:
                file_size = int(image_info['file_size'])
                if file_size > self.max_file_size:
                    errors.append(self._create_error(
                        f"Image file too large: {file_size} bytes",
                        severity="warning",
                        field=f"{field_prefix}.size"
                    ))
            except (ValueError, TypeError):
                errors.append(self._create_error(
                    "Invalid file size",
                    field=f"{field_prefix}.file_size"
                ))
                
        return errors


class DataValidationOrchestrator:
    """Orchestrates multiple validators for comprehensive data validation."""
    
    def __init__(self):
        self.validators = {
            'basic': BasicDataValidator(),
            'cad_specific': CADSpecificValidator(),
            'quality': QualityValidator(),
            'images': ImageValidator()
        }
        self.validation_config = {
            'stop_on_first_error': False,
            'include_warnings': True,
            'include_info': False
        }
        
    def validate_sample(self, data: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """Validate a single data sample using all validators."""
        all_errors = []
        overall_valid = True
        
        for validator_name, validator in self.validators.items():
            try:
                is_valid, errors = validator.validate(data)
                
                if not is_valid:
                    overall_valid = False
                    
                # Filter errors based on configuration
                filtered_errors = self._filter_errors(errors)
                all_errors.extend(filtered_errors)
                
                # Stop on first error if configured
                if self.validation_config['stop_on_first_error'] and not is_valid:
                    break
                    
            except Exception as e:
                logger.error(f"Validation failed with {validator_name}: {e}")
                all_errors.append(ValidationError(
                    f"Validator {validator_name} failed: {str(e)}",
                    severity="error"
                ))
                overall_valid = False
                
        return overall_valid, all_errors
        
    def validate_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a batch of samples and return statistics."""
        results = {
            'total_samples': len(batch),
            'valid_samples': 0,
            'invalid_samples': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'error_breakdown': {},
            'sample_results': []
        }
        
        for i, sample in enumerate(batch):
            is_valid, errors = self.validate_sample(sample)
            
            sample_result = {
                'index': i,
                'valid': is_valid,
                'error_count': len([e for e in errors if e.severity == "error"]),
                'warning_count': len([e for e in errors if e.severity == "warning"]),
                'errors': [str(e) for e in errors]
            }
            
            results['sample_results'].append(sample_result)
            
            if is_valid:
                results['valid_samples'] += 1
            else:
                results['invalid_samples'] += 1
                
            # Count errors by type
            for error in errors:
                if error.severity == "error":
                    results['total_errors'] += 1
                elif error.severity == "warning":
                    results['total_warnings'] += 1
                    
                # Track error types
                error_type = error.field or "general"
                if error_type not in results['error_breakdown']:
                    results['error_breakdown'][error_type] = 0
                results['error_breakdown'][error_type] += 1
                
        return results
        
    def _filter_errors(self, errors: List[ValidationError]) -> List[ValidationError]:
        """Filter errors based on configuration."""
        filtered = []
        
        for error in errors:
            if error.severity == "error":
                filtered.append(error)
            elif error.severity == "warning" and self.validation_config['include_warnings']:
                filtered.append(error)
            elif error.severity == "info" and self.validation_config['include_info']:
                filtered.append(error)
                
        return filtered
        
    def configure_validation(self, **kwargs):
        """Configure validation parameters."""
        for key, value in kwargs.items():
            if key in self.validation_config:
                self.validation_config[key] = value
                
    def add_validator(self, name: str, validator: BaseValidator):
        """Add a custom validator."""
        self.validators[name] = validator
        
    def remove_validator(self, name: str):
        """Remove a validator."""
        if name in self.validators:
            del self.validators[name]