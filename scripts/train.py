#!/usr/bin/env python
"""
Training script for Text-to-CAD model

This script:
1. Loads configuration from YAML file
2. Sets up data loaders for training and validation
3. Initializes model, optimizer, and scheduler
4. Trains the model using the specified training loop
5. Saves checkpoints and logs metrics
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import wandb
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.text_to_cad import TextToCADModel
from src.models.visual_feedback import VisualFeedbackModule
from src.data.preprocessing import create_dataloaders
from src.training.trainer import Trainer
from src.data.dataset import TextToCADDataset
from src.utils.logging import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Text-to-CAD model")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="outputs/logs",
                        help="Directory to save logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--finetune", action="store_true",
                        help="Finetune with visual feedback")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    # Configuration
    config = {
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'batch_size': 32,
        'use_wandb': True,
        'wandb_project': 'text-to-cad',
        'save_interval': 1,
        'log_interval': 100,
        'warmup_steps': 1000,
        'max_grad_norm': 1.0,
        'weight_decay': 0.01
    }

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Using GPT-2 tokenizer as base

    # Create datasets
    data_dir = Path("data/processed")
    train_dataset = TextToCADDataset(
        data_dir=data_dir / "train",
        tokenizer=tokenizer,
        max_length=512
    )
    val_dataset = TextToCADDataset(
        data_dir=data_dir / "val",
        tokenizer=tokenizer,
        max_length=512
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Initialize model
    model = TextToCADModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_layers=12,
        num_heads=12
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()