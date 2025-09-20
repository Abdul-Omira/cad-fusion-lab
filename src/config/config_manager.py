"""
Configuration Management System for CAD Fusion Lab

This module provides centralized configuration management for the entire project,
supporting multiple environments, validation, and easy extensibility.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    OFFLINE = "offline"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int = 10000
    text_encoder_name: str = "bert-base-uncased"
    d_model: int = 512
    nhead: int = 8
    num_decoder_layers: int = 24
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 512
    max_grad_norm: float = 1.0
    offline_mode: bool = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    save_every_n_epochs: int = 10
    eval_every_n_steps: int = 500
    early_stopping_patience: int = 5
    use_mixed_precision: bool = True


@dataclass
class DataConfig:
    """Data processing configuration."""
    train_data_path: str = "data/train"
    val_data_path: str = "data/val" 
    test_data_path: str = "data/test"
    output_dir: str = "data/processed"
    max_text_length: int = 512
    max_cad_length: int = 512
    enable_augmentation: bool = True
    augmentation_rate: float = 0.3
    max_variations_per_sample: int = 3
    validation_threshold: float = 0.8
    
    # Feature extraction settings
    extract_semantic_features: bool = True
    extract_technical_features: bool = True
    extract_cad_features: bool = True
    extract_geometric_features: bool = True


@dataclass
class ValidationConfig:
    """Validation configuration."""
    min_thickness: float = 0.5
    check_topology: bool = True
    check_self_intersection: bool = True
    check_watertightness: bool = True
    mesh_quality: float = 0.1
    enable_kcl_validation: bool = True
    validation_timeout: int = 30


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 300
    enable_cors: bool = True
    allowed_origins: List[str] = None
    log_level: str = "INFO"
    enable_metrics: bool = True


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    device: str = "auto"  # "cpu", "cuda", "auto"
    num_workers: int = 4
    pin_memory: bool = True
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    mixed_precision: bool = True


class ConfigManager:
    """Central configuration manager for the entire project."""
    
    def __init__(self, config_dir: str = "configs", environment: Environment = Environment.DEVELOPMENT):
        self.config_dir = Path(config_dir)
        self.environment = environment
        self._config_cache = {}
        
        # Initialize with defaults
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.validation = ValidationConfig()
        self.deployment = DeploymentConfig()
        self.hardware = HardwareConfig()
        
        # Load environment-specific configuration
        self.load_environment_config()
        
        # Apply environment variables
        self.apply_env_overrides()
        
        # Validate configuration
        try:
            self.validate_config()
        except Exception as e:
            logger.warning(f"Configuration validation failed: {e}. Using defaults.")
    
    def load_environment_config(self):
        """Load configuration for the current environment."""
        try:
            config_file = self.config_dir / f"{self.environment.value}_config.yaml"
            if not config_file.exists():
                # Fall back to base config
                config_file = self.config_dir / "base_config.yaml"
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                self._apply_config_data(config_data)
                logger.info(f"Loaded configuration from {config_file}")
            else:
                logger.warning(f"No configuration file found for {self.environment.value}, using defaults")
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Using default configuration")
    
    def _apply_config_data(self, config_data: Dict[str, Any]):
        """Apply configuration data to the respective sections."""
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name):
                section_config = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section_config, key):
                        # Get the expected type from the dataclass field
                        expected_type = type(getattr(section_config, key))
                        try:
                            # Convert the value to the expected type
                            if expected_type == bool and isinstance(value, str):
                                value = value.lower() in ('true', '1', 'yes', 'on')
                            elif expected_type in (int, float) and isinstance(value, str):
                                value = expected_type(value)
                            
                            setattr(section_config, key, value)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Failed to convert {section_name}.{key}={value} to {expected_type}: {e}")
                    else:
                        logger.warning(f"Unknown config key: {section_name}.{key}")
            else:
                logger.info(f"Skipping unknown section: {section_name}")
    
    def apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_overrides = {
            # Model configuration
            'CAD_MODEL_OFFLINE_MODE': ('model', 'offline_mode', bool),
            'CAD_MODEL_VOCAB_SIZE': ('model', 'vocab_size', int),
            'CAD_MODEL_D_MODEL': ('model', 'd_model', int),
            
            # Training configuration  
            'CAD_TRAINING_BATCH_SIZE': ('training', 'batch_size', int),
            'CAD_TRAINING_LEARNING_RATE': ('training', 'learning_rate', float),
            'CAD_TRAINING_NUM_EPOCHS': ('training', 'num_epochs', int),
            
            # Data configuration
            'CAD_DATA_TRAIN_PATH': ('data', 'train_data_path', str),
            'CAD_DATA_OUTPUT_DIR': ('data', 'output_dir', str),
            'CAD_DATA_ENABLE_AUGMENTATION': ('data', 'enable_augmentation', bool),
            
            # Deployment configuration
            'CAD_DEPLOY_HOST': ('deployment', 'host', str),
            'CAD_DEPLOY_PORT': ('deployment', 'port', int),
            'CAD_DEPLOY_WORKERS': ('deployment', 'workers', int),
            
            # Hardware configuration
            'CAD_HARDWARE_DEVICE': ('hardware', 'device', str),
            'CAD_HARDWARE_NUM_WORKERS': ('hardware', 'num_workers', int),
        }
        
        for env_var, (section, key, var_type) in env_overrides.items():
            if env_var in os.environ:
                try:
                    value = os.environ[env_var]
                    
                    # Type conversion
                    if var_type == bool:
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    elif var_type == int:
                        value = int(value)
                    elif var_type == float:
                        value = float(value)
                    
                    # Apply the override
                    section_config = getattr(self, section)
                    setattr(section_config, key, value)
                    logger.info(f"Applied environment override: {env_var}={value}")
                    
                except (ValueError, AttributeError) as e:
                    logger.error(f"Failed to apply environment override {env_var}: {e}")
    
    def validate_config(self):
        """Validate the configuration for consistency and correctness."""
        errors = []
        
        # Model validation
        if self.model.vocab_size <= 0:
            errors.append("Model vocab_size must be positive")
        if self.model.d_model <= 0:
            errors.append("Model d_model must be positive")
        if not 0 <= self.model.dropout <= 1:
            errors.append("Model dropout must be between 0 and 1")
            
        # Training validation
        if self.training.batch_size <= 0:
            errors.append("Training batch_size must be positive")
        if self.training.learning_rate <= 0:
            errors.append("Training learning_rate must be positive")
        if self.training.num_epochs <= 0:
            errors.append("Training num_epochs must be positive")
            
        # Data validation
        if not 0 <= self.data.augmentation_rate <= 1:
            errors.append("Data augmentation_rate must be between 0 and 1")
        if not 0 <= self.data.validation_threshold <= 1:
            errors.append("Data validation_threshold must be between 0 and 1")
            
        # Deployment validation
        if not 1 <= self.deployment.port <= 65535:
            errors.append("Deployment port must be between 1 and 65535")
        if self.deployment.workers <= 0:
            errors.append("Deployment workers must be positive")
            
        # Hardware validation
        if self.hardware.num_workers <= 0:
            errors.append("Hardware num_workers must be positive")
            
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)
            
        logger.info("Configuration validation passed")
    
    def save_config(self, output_path: str):
        """Save the current configuration to a file."""
        config_data = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'validation': asdict(self.validation),
            'deployment': asdict(self.deployment),
            'hardware': asdict(self.hardware),
            'environment': self.environment.value
        }
        
        output_path = Path(output_path)
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            with open(output_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        else:
            with open(output_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
        logger.info(f"Configuration saved to {output_path}")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get the complete configuration as a dictionary."""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'validation': asdict(self.validation),
            'deployment': asdict(self.deployment),
            'hardware': asdict(self.hardware),
            'environment': self.environment.value
        }
    
    def update_config(self, section: str, **kwargs):
        """Update configuration values for a specific section."""
        if not hasattr(self, section):
            raise ValueError(f"Unknown configuration section: {section}")
            
        section_config = getattr(self, section)
        for key, value in kwargs.items():
            if hasattr(section_config, key):
                setattr(section_config, key, value)
                logger.info(f"Updated {section}.{key} = {value}")
            else:
                logger.warning(f"Unknown config key: {section}.{key}")
        
        # Re-validate after updates
        self.validate_config()
    
    @classmethod
    def from_file(cls, config_file: str, environment: Environment = Environment.DEVELOPMENT):
        """Create a ConfigManager from a specific configuration file."""
        config_manager = cls(environment=environment)
        
        config_file = Path(config_file)
        if config_file.exists():
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            config_manager._apply_config_data(config_data)
            config_manager.apply_env_overrides()
            config_manager.validate_config()
            
        return config_manager


# Global configuration instance
_global_config: Optional[ConfigManager] = None

def get_config() -> ConfigManager:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        # Determine environment
        env_name = os.environ.get('CAD_ENVIRONMENT', 'development')
        try:
            environment = Environment(env_name)
        except ValueError:
            logger.warning(f"Unknown environment '{env_name}', using development")
            environment = Environment.DEVELOPMENT
            
        _global_config = ConfigManager(environment=environment)
    
    return _global_config

def set_config(config: ConfigManager):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config

def initialize_config(config_dir: str = "configs", environment: str = None):
    """Initialize the global configuration."""
    if environment:
        try:
            env = Environment(environment)
        except ValueError:
            logger.warning(f"Unknown environment '{environment}', using development")
            env = Environment.DEVELOPMENT
    else:
        env_name = os.environ.get('CAD_ENVIRONMENT', 'development')
        try:
            env = Environment(env_name)
        except ValueError:
            env = Environment.DEVELOPMENT
    
    global _global_config
    _global_config = ConfigManager(config_dir=config_dir, environment=env)
    return _global_config


# Example usage
if __name__ == "__main__":
    # Initialize configuration
    config = initialize_config()
    
    # Print current configuration
    print("Current Configuration:")
    print(f"Environment: {config.environment.value}")
    print(f"Model offline mode: {config.model.offline_mode}")
    print(f"Training batch size: {config.training.batch_size}")
    print(f"Data augmentation: {config.data.enable_augmentation}")
    
    # Update configuration
    config.update_config('training', batch_size=32, learning_rate=2e-4)
    
    # Save configuration
    config.save_config("current_config.yaml")
    
    print("Configuration management system initialized successfully!")