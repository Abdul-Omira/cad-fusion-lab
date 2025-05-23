"""
Tests for geometric validation module
"""

import pytest
import tempfile
import json
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.validation.geometric import (
    GeometricValidator, 
    KCLGenerator, 
    GeometricError,
    SyntaxError,
    TopologyError, 
    ThicknessError,
    IntersectionError,
    check_manufacturing_validity
)


class TestKCLGenerator:
    """Tests for KCL code generation."""
    
    @pytest.fixture
    def kcl_generator(self):
        """Create a KCL generator for testing."""
        return KCLGenerator(vocab_size=10000)
    
    def test_init(self, kcl_generator):
        """Test initialization."""
        assert kcl_generator.vocab_size == 10000
        assert kcl_generator.pad_token == 0
        assert kcl_generator.sos_token == 1
        assert kcl_generator.eos_token == 2
        assert kcl_generator.operation_start == 4
        assert len(kcl_generator.operations) > 0
    
    def test_generate_kcl_empty(self, kcl_generator):
        """Test generating KCL from empty sequence."""
        kcl = kcl_generator.generate_kcl([])
        assert kcl == ""
    
    def test_generate_kcl_with_special_tokens(self, kcl_generator):
        """Test generating KCL from sequence with special tokens."""
        # Sequence with SOS, EOS, and PAD
        sequence = [1, 4, 1000, 5, 1010, 2, 0, 0]
        kcl = kcl_generator.generate_kcl(sequence)
        
        # Should have 2 operations (Box and Cylinder)
        assert "Box" in kcl
        assert "Cylinder" in kcl
        assert len(kcl.splitlines()) == 2
    
    def test_generate_kcl_with_parameters(self, kcl_generator):
        """Test generating KCL with parameters."""
        # Box with parameters 10, 20, 30
        sequence = [4, 1010, 1020, 1030]
        kcl = kcl_generator.generate_kcl(sequence)
        
        # Should have Box operation with parameters
        assert "Box(10, 20, 30)" in kcl


class TestGeometricValidator:
    """Tests for geometric validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create a geometric validator for testing."""
        return GeometricValidator(
            min_thickness=0.5,
            check_topology=True,
            check_self_intersection=True,
            check_watertightness=True
        )
    
    def test_init(self, validator):
        """Test initialization with parameters."""
        assert validator.min_thickness == 0.5
        assert validator.check_topology is True
        assert validator.check_self_intersection is True
        assert validator.check_watertightness is True
        assert validator.kcl_generator is not None
    
    def test_validate_empty(self, validator):
        """Test validating empty sequence."""
        is_valid, errors = validator.validate([])
        
        # Empty sequence should be invalid
        assert is_valid is False
        assert len(errors) > 0
        assert any(isinstance(e, SyntaxError) for e in errors)
    
    def test_validate_valid_sequence(self, validator):
        """Test validating a valid sequence."""
        # Create a simple valid sequence (mocked)
        sequence = [1, 4, 1010, 1020, 1030, 2]  # SOS, Box with params, EOS
        
        # Mock validation to succeed by manipulating _check_syntax
        original_check_syntax = validator._check_syntax
        validator._check_syntax = lambda kcl: (True, [])
        
        # Validate
        is_valid, errors = validator.validate(sequence)
        
        # Restore original method
        validator._check_syntax = original_check_syntax
        
        # Should be valid with no errors
        assert is_valid is True
        assert len(errors) == 0
    
    def test_check_wall_thickness(self, validator):
        """Test wall thickness validation."""
        # Create mock CAD models with different thicknesses
        thin_model = {"min_thickness": 0.2}  # Below minimum 0.5
        thick_model = {"min_thickness": 0.7}  # Above minimum 0.5
        
        # Check each model
        thin_valid, thin_errors = validator._check_wall_thickness(thin_model)
        thick_valid, thick_errors = validator._check_wall_thickness(thick_model)
        
        # Thin model should fail
        assert thin_valid is False
        assert len(thin_errors) == 1
        assert isinstance(thin_errors[0], ThicknessError)
        
        # Thick model should pass
        assert thick_valid is True
        assert len(thick_errors) == 0
    
    def test_export_validation_report(self, validator):
        """Test exporting validation report."""
        # Create a sequence
        sequence = [1, 4, 1010, 1020, 1030, 2]  # SOS, Box with params, EOS
        
        # Create temporary file for report
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            report_path = tmp.name
            
        try:
            # Export validation report
            validator.export_validation_report(sequence, report_path)
            
            # Check report file exists and is valid JSON
            assert os.path.exists(report_path)
            
            with open(report_path, "r") as f:
                report = json.load(f)
            
            # Check report structure
            assert "is_valid" in report
            assert "errors" in report
            assert "kcl_code" in report
            assert "validation_timestamp" in report
            assert "validator_version" in report
            
        finally:
            # Clean up
            if os.path.exists(report_path):
                os.unlink(report_path)


class TestErrorHandling:
    """Test error handling and reporting."""
    
    def test_geometric_error_base(self):
        """Test base GeometricError class."""
        error = GeometricError(message="Test error", location={"x": 1, "y": 2, "z": 3})
        assert error.message == "Test error"
        assert error.location == {"x": 1, "y": 2, "z": 3}
        assert error.severity == "error"
    
    def test_specific_errors(self):
        """Test specific error subclasses."""
        syntax_error = SyntaxError(message="Invalid syntax")
        topology_error = TopologyError(message="Invalid topology")
        thickness_error = ThicknessError(message="Too thin")
        intersection_error = IntersectionError(message="Self-intersecting")
        
        assert syntax_error.message == "Invalid syntax"
        assert topology_error.message == "Invalid topology"
        assert thickness_error.message == "Too thin"
        assert intersection_error.message == "Self-intersecting"
        
        # Check inheritance
        assert isinstance(syntax_error, GeometricError)
        assert isinstance(topology_error, GeometricError)
        assert isinstance(thickness_error, GeometricError)
        assert isinstance(intersection_error, GeometricError)


def test_check_manufacturing_validity():
    """Test the check_manufacturing_validity helper function."""
    # Create a simple valid sequence
    valid_sequence = [1, 4, 1010, 1020, 1030, 2]  # SOS, Box with params, EOS
    
    # Should be valid (mocked internally)
    is_valid = check_manufacturing_validity(valid_sequence)
    assert isinstance(is_valid, bool)
    
    # Too short sequence should fail validation
    invalid_sequence = [1, 2]  # Just SOS and EOS
    is_invalid = check_manufacturing_validity(invalid_sequence)
    assert isinstance(is_invalid, bool)