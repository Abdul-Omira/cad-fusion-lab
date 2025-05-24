"""
Logging utilities for the Text-to-CAD project.
"""

import logging
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("text_to_cad")
    logger.setLevel(logging.INFO)
    
    # Create handlers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_path / f"training_{timestamp}.log")
    console_handler = logging.StreamHandler()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 