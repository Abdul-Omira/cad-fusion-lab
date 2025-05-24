"""
Tests for the geometric validation system.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
from src.validation.geometric import (
    GeometricValidator,
    KCLGenerator,
    GeometricError,
    SyntaxError,
    TopologyError,
    ThicknessError,
    IntersectionError
)

@pytest.fixture
def validator():
    """Create a test validator instance."""
    return GeometricValidator(
        min_thickness=0.5,
        check_topology=True,
        check_self_intersection=True,
        check_watertightness=True,
        mesh_quality=0.1
    )

@pytest.fixture
def kcl_generator():
    """Create a test KCL generator instance."""
    return KCLGenerator(vocab_size=1000)

@pytest.fixture
def sample_cad_sequence():
    """Create a sample CAD sequence."""
    return [1, 4, 1000, 1001, 1002, 5, 1003, 1004, 1005, 2]  # SOS, Box, params, Cylinder, params, EOS

def test_validator_initialization(validator):
    """Test validator initialization."""
    assert validator.min_thickness == 0.5
    assert validator.check_topology is True
    assert validator.check_self_intersection is True
    assert validator.check_watertightness is True
    assert validator.mesh_quality == 0.1
    assert isinstance(validator.kcl_generator, KCLGenerator)

def test_kcl_generation(kcl_generator, sample_cad_sequence):
    """Test KCL code generation."""
    kcl_code = kcl_generator.generate_kcl(sample_cad_sequence)
    
    # Check basic KCL syntax
    assert isinstance(kcl_code, str)
    assert "Box" in kcl_code
    assert "Cylinder" in kcl_code
    assert kcl_code.count("(") == kcl_code.count(")")
    
    # Check parameter handling
    assert all(str(param) in kcl_code for param in [1000, 1001, 1002, 1003, 1004, 1005])

def test_validation_pipeline(validator, sample_cad_sequence):
    """Test the complete validation pipeline."""
    is_valid, errors = validator.validate(sample_cad_sequence)
    
    # Check return types
    assert isinstance(is_valid, bool)
    assert isinstance(errors, list)
    assert all(isinstance(error, GeometricError) for error in errors)
    
    # Check error handling
    if not is_valid:
        assert len(errors) > 0
        for error in errors:
            assert hasattr(error, "message")
            assert hasattr(error, "severity")
            if error.location:
                assert isinstance(error.location, dict)
                assert all(k in error.location for k in ["x", "y", "z"])

def test_wall_thickness_validation(validator):
    """Test wall thickness validation."""
    # Create a sequence that would result in thin walls
    thin_wall_sequence = [1, 4, 1000, 1001, 1002, 2]  # Simple box with thin walls
    
    is_valid, errors = validator.validate(thin_wall_sequence)
    
    # If CAD kernel is available, check for thickness errors
    if validator._execute_kcl("") is not None:  # Check if CAD kernel is available
        assert not is_valid
        assert any(isinstance(error, ThicknessError) for error in errors)
    else:
        # With mock validation, should always be valid
        assert is_valid
        assert len(errors) == 0

def test_topology_validation(validator):
    """Test topology validation."""
    # Create a sequence that would result in invalid topology
    invalid_topology_sequence = [1, 4, 1000, 1001, 1002, 5, 1003, 1004, 1005, 2]
    
    is_valid, errors = validator.validate(invalid_topology_sequence)
    
    # If CAD kernel is available, check for topology errors
    if validator._execute_kcl("") is not None:
        assert not is_valid
        assert any(isinstance(error, TopologyError) for error in errors)
    else:
        # With mock validation, should always be valid
        assert is_valid
        assert len(errors) == 0

def test_self_intersection_validation(validator):
    """Test self-intersection validation."""
    # Create a sequence that would result in self-intersections
    intersection_sequence = [1, 4, 1000, 1001, 1002, 4, 1003, 1004, 1005, 2]
    
    is_valid, errors = validator.validate(intersection_sequence)
    
    # If CAD kernel is available, check for intersection errors
    if validator._execute_kcl("") is not None:
        assert not is_valid
        assert any(isinstance(error, IntersectionError) for error in errors)
    else:
        # With mock validation, should always be valid
        assert is_valid
        assert len(errors) == 0

def test_watertightness_validation(validator):
    """Test watertightness validation."""
    # Create a sequence that would result in non-watertight geometry
    non_watertight_sequence = [1, 4, 1000, 1001, 1002, 5, 1003, 1004, 1005, 2]
    
    is_valid, errors = validator.validate(non_watertight_sequence)
    
    # If CAD kernel is available, check for watertightness errors
    if validator._execute_kcl("") is not None:
        assert not is_valid
        assert any(isinstance(error, GeometricError) for error in errors)
    else:
        # With mock validation, should always be valid
        assert is_valid
        assert len(errors) == 0

def test_validation_report_export(validator, sample_cad_sequence, tmp_path):
    """Test validation report export."""
    # Test JSON export
    json_path = tmp_path / "validation.json"
    validator.export_validation_report(sample_cad_sequence, str(json_path), format="json")
    
    assert json_path.exists()
    with open(json_path) as f:
        report = json.load(f)
        assert "is_valid" in report
        assert "errors" in report
        assert isinstance(report["errors"], list)
    
    # Test HTML export
    html_path = tmp_path / "validation.html"
    validator.export_validation_report(sample_cad_sequence, str(html_path), format="html")
    
    assert html_path.exists()
    with open(html_path) as f:
        content = f.read()
        assert "<html" in content
        assert "<body" in content
        assert "CAD Validation Report" in content

def test_error_handling(validator):
    """Test error handling in validation."""
    # Test with empty sequence
    is_valid, errors = validator.validate([])
    assert not is_valid
    assert len(errors) > 0
    assert any(isinstance(error, SyntaxError) for error in errors)
    
    # Test with invalid tokens
    invalid_sequence = [1, 9999, 2]  # Invalid operation token
    is_valid, errors = validator.validate(invalid_sequence)
    assert not is_valid
    assert len(errors) > 0

def test_mock_validation(validator):
    """Test mock validation when CAD kernel is not available."""
    # Force mock validation
    validator._execute_kcl = lambda x: None
    
    # Test validation
    is_valid, errors = validator.validate([1, 4, 1000, 1001, 1002, 2])
    assert is_valid
    assert len(errors) == 0

def test_kcl_generator_error_handling(kcl_generator):
    """Test KCL generator error handling."""
    # Test with empty sequence
    kcl_code = kcl_generator.generate_kcl([])
    assert kcl_code == ""
    
    # Test with invalid tokens
    kcl_code = kcl_generator.generate_kcl([9999])
    assert "Operation_9999" in kcl_code
    
    # Test with special tokens only
    kcl_code = kcl_generator.generate_kcl([1, 2])  # SOS and EOS only
    assert kcl_code == ""