# Configuration Module

This module provides centralized configuration management for the CAD Fusion Lab project.

## Usage

```python
from src.config.config_manager import get_config, initialize_config, Environment

# Initialize with environment
config = initialize_config(environment='development')

# Access configuration
print(f"Model offline mode: {config.model.offline_mode}")
print(f"Training batch size: {config.training.batch_size}")

# Update configuration
config.update_config('training', batch_size=32, learning_rate=2e-4)

# Save current configuration
config.save_config("current_config.yaml")
```

## Environment Variables

The system supports environment variable overrides:

- `CAD_ENVIRONMENT`: Set deployment environment (development, testing, production, offline)
- `CAD_MODEL_OFFLINE_MODE`: Enable/disable offline mode
- `CAD_TRAINING_BATCH_SIZE`: Set training batch size
- `CAD_DEPLOY_PORT`: Set deployment port
- And many more...

## Configuration Files

Configuration files should be placed in the `configs/` directory:

- `base_config.yaml`: Default configuration
- `development_config.yaml`: Development-specific overrides
- `production_config.yaml`: Production-specific overrides
- `testing_config.yaml`: Testing-specific overrides
- `offline_config.yaml`: Offline deployment configuration