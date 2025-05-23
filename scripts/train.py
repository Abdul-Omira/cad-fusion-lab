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

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.text_to_cad import TextToCADModel
from src.models.visual_feedback import VisualFeedbackModule
from src.data.preprocessing import create_dataloaders
from src.training.trainer import Trainer


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
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = True
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    log_dir = args.log_dir or config.get("logging", {}).get("log_dir", "outputs/logs")
    logger = setup_logging(log_dir)
    logger.info(f"Starting training with config: {config}")
    
    # Disable wandb if requested
    if args.no_wandb:
        config["logging"]["use_wandb"] = False
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        train_cad_path=config["data"]["train_cad_path"],
        train_text_path=config["data"]["train_text_path"],
        val_cad_path=config["data"]["val_cad_path"],
        val_text_path=config["data"]["val_text_path"],
        batch_size=config["training"]["batch_size"]
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = TextToCADModel(
        vocab_size=config["model"]["vocab_size"],
        text_encoder_name=config["model"]["text_encoder_name"],
        d_model=config["model"]["d_model"],
        nhead=config["model"]["nhead"],
        num_decoder_layers=config["model"]["num_decoder_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        max_seq_length=config["model"]["max_seq_length"]
    )
    
    # Load from checkpoint if specified
    start_epoch = 0
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
    
    # Initialize visual feedback module if finetuning
    visual_feedback_module = None
    if args.finetune:
        logger.info("Initializing visual feedback module for fine-tuning...")
        visual_feedback_module = VisualFeedbackModule(
            clip_model_name=config["visual_feedback"]["clip_model_name"],
            clip_weight=config["visual_feedback"]["clip_weight"],
            geometry_weight=config["visual_feedback"]["geometry_weight"]
        )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        visual_feedback_module=visual_feedback_module
    )
    
    # Start training
    if args.finetune:
        logger.info("Starting fine-tuning with visual feedback...")
        trainer.finetune_with_ppo()
    else:
        logger.info("Starting training...")
        trainer.train()
    
    # Save final model
    final_checkpoint_path = output_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config
    }, final_checkpoint_path)
    logger.info(f"Saved final model to {final_checkpoint_path}")


if __name__ == "__main__":
    main()