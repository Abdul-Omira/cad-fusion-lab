"""
Tests for configuration system validation.
"""

import pytest
import yaml
import os
from pathlib import Path
import torch
from src.models.text_to_cad import TextToCADModel
from src.validation.geometric import GeometricValidator

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def config_paths():
    """Get paths to all configuration files."""
    config_dir = Path("configs")
    return {
        "base": config_dir / "base_config.yaml",
        "small": config_dir / "small_config.yaml",
        "large": config_dir / "large_config.yaml"
    }

def test_config_files_exist(config_paths):
    """Test that all configuration files exist."""
    for name, path in config_paths.items():
        assert path.exists(), f"{name} configuration file not found at {path}"

def test_config_structure(config_paths):
    """Test that all configurations have the required structure."""
    required_sections = [
        "model", "training", "generation", "validation",
        "data", "logging", "paths", "hardware", "deployment"
    ]
    
    for name, path in config_paths.items():
        config = load_config(path)
        for section in required_sections:
            assert section in config, f"Missing {section} section in {name} config"

def test_model_config_validation(config_paths):
    """Test that model configurations are valid."""
    for name, path in config_paths.items():
        config = load_config(path)
        model_config = config["model"]
        
        # Test vocabulary size
        assert model_config["vocab_size"] > 0, f"Invalid vocab_size in {name} config"
        
        # Test model dimensions
        assert model_config["d_model"] > 0, f"Invalid d_model in {name} config"
        assert model_config["nhead"] > 0, f"Invalid nhead in {name} config"
        assert model_config["num_decoder_layers"] > 0, f"Invalid num_decoder_layers in {name} config"
        
        # Test that nhead divides d_model evenly
        assert model_config["d_model"] % model_config["nhead"] == 0, \
            f"d_model must be divisible by nhead in {name} config"

def test_training_config_validation(config_paths):
    """Test that training configurations are valid."""
    for name, path in config_paths.items():
        config = load_config(path)
        training_config = config["training"]
        
        # Test batch size
        assert training_config["batch_size"] > 0, f"Invalid batch_size in {name} config"
        
        # Test learning rate
        learning_rate = float(training_config["learning_rate"])
        assert 0 < learning_rate < 1, f"Invalid learning_rate in {name} config"
        
        # Test epochs
        assert training_config["num_epochs"] > 0, f"Invalid num_epochs in {name} config"

def test_hardware_config_validation(config_paths):
    """Test that hardware configurations are valid."""
    for name, path in config_paths.items():
        config = load_config(path)
        hardware_config = config["hardware"]
        
        # Test device
        assert hardware_config["device"] in ["cpu", "cuda"], f"Invalid device in {name} config"
        
        # Test GPU count
        assert hardware_config["num_gpus"] >= 0, f"Invalid num_gpus in {name} config"
        
        # Test mixed precision
        assert isinstance(hardware_config["mixed_precision"], bool), \
            f"mixed_precision must be boolean in {name} config"

def test_model_initialization(config_paths):
    """Test that models can be initialized with each configuration."""
    for name, path in config_paths.items():
        config = load_config(path)
        model_config = config["model"]
        
        try:
            model = TextToCADModel(
                vocab_size=model_config["vocab_size"],
                text_encoder_name=model_config["text_encoder_name"],
                d_model=model_config["d_model"],
                nhead=model_config["nhead"],
                num_decoder_layers=model_config["num_decoder_layers"],
                dim_feedforward=model_config["dim_feedforward"],
                dropout=model_config["dropout"],
                max_seq_length=model_config["max_seq_length"]
            )
            assert isinstance(model, TextToCADModel), f"Failed to initialize model with {name} config"
        except Exception as e:
            pytest.fail(f"Model initialization failed with {name} config: {str(e)}")

def test_validator_initialization(config_paths):
    """Test that validators can be initialized with each configuration."""
    for name, path in config_paths.items():
        config = load_config(path)
        validation_config = config["validation"]
        
        try:
            validator = GeometricValidator(
                min_thickness=validation_config["min_thickness"],
                check_topology=validation_config["check_topology"],
                check_self_intersection=validation_config["check_self_intersection"],
                check_watertightness=validation_config["check_watertightness"],
                mesh_quality=validation_config["mesh_quality"]
            )
            assert isinstance(validator, GeometricValidator), \
                f"Failed to initialize validator with {name} config"
        except Exception as e:
            pytest.fail(f"Validator initialization failed with {name} config: {str(e)}")

def test_path_configuration(config_paths):
    """Test that all required paths exist or can be created."""
    for name, path in config_paths.items():
        config = load_config(path)
        paths_config = config["paths"]
        
        # Test that all paths are strings
        for key, value in paths_config.items():
            assert isinstance(value, str), f"Path {key} must be string in {name} config"
        
        # Test that paths can be created
        for key, value in paths_config.items():
            try:
                os.makedirs(value, exist_ok=True)
                assert os.path.exists(value), f"Failed to create path {key} in {name} config"
            except Exception as e:
                pytest.fail(f"Failed to create path {key} in {name} config: {str(e)}")

def test_config_consistency(config_paths):
    """Test consistency between different configurations."""
    configs = {name: load_config(path) for name, path in config_paths.items()}
    
    # Test that small config has smaller values than base config
    for key in ["d_model", "nhead", "num_decoder_layers", "dim_feedforward"]:
        assert configs["small"]["model"][key] <= configs["base"]["model"][key], \
            f"Small config {key} should be <= base config {key}"
    
    # Test that large config has larger values than base config
    for key in ["d_model", "nhead", "num_decoder_layers", "dim_feedforward"]:
        assert configs["large"]["model"][key] >= configs["base"]["model"][key], \
            f"Large config {key} should be >= base config {key}"

def test_deployment_config_validation(config_paths):
    """Test that deployment configurations are valid."""
    for name, path in config_paths.items():
        config = load_config(path)
        deployment_config = config["deployment"]
        
        # Test port number
        assert 1024 <= deployment_config["port"] <= 65535, \
            f"Invalid port number in {name} config"
        
        # Test timeout
        assert deployment_config["timeout"] > 0, f"Invalid timeout in {name} config"
        
        # Test workers
        assert deployment_config["workers"] > 0, f"Invalid workers in {name} config"
        
        # Test batch size
        assert deployment_config["max_batch_size"] > 0, \
            f"Invalid max_batch_size in {name} config" 