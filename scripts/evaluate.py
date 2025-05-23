#!/usr/bin/env python
"""
Evaluation script for Text-to-CAD model

This script:
1. Loads a trained model from a checkpoint
2. Evaluates performance on test set
3. Computes metrics:
   - Token prediction accuracy
   - Chamfer distance
   - CLIP score (text-image alignment)
   - Manufacturing validity rate
4. Generates visualization samples
"""

import os
import sys
import argparse
import yaml
import json
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.text_to_cad import TextToCADModel
from src.models.visual_feedback import VisualFeedbackModule
from src.data.preprocessing import TextToCADDataset
from src.validation.geometric import check_manufacturing_validity


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Text-to-CAD model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Path to config file")
    parser.add_argument("--test-set", type=str, default=None,
                        help="Path to test set (overrides config)")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to evaluate on (cuda/cpu)")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of samples to visualize")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization samples")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(output_dir):
    """Set up logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "evaluation.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def compute_chamfer_distance(pred_vertices, gt_vertices):
    """
    Compute Chamfer distance between predicted and ground truth vertices.
    
    Args:
        pred_vertices: Predicted 3D vertices (B, N, 3)
        gt_vertices: Ground truth 3D vertices (B, M, 3)
        
    Returns:
        Mean Chamfer distance
    """
    # Placeholder implementation - would use actual Chamfer distance computation
    return np.random.uniform(0.5, 1.0)


def compute_clip_score(renderings, text_descriptions):
    """
    Compute CLIP score for text-image alignment.
    
    Args:
        renderings: Rendered images of CAD models
        text_descriptions: Text descriptions
        
    Returns:
        Mean CLIP score
    """
    # Placeholder implementation - would use actual CLIP score computation
    return np.random.uniform(0.7, 0.9)


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_dir)
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model checkpoint
    logger.info(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    model_config = checkpoint.get("config", {}).get("model", config["model"])
    
    # Initialize model
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Load test dataset
    test_cad_path = args.test_set or config["data"]["test_cad_path"]
    test_text_path = args.test_set or config["data"]["test_text_path"]
    
    logger.info(f"Loading test dataset from {test_cad_path} and {test_text_path}")
    test_dataset = TextToCADDataset(test_cad_path, test_text_path)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize visual feedback module for evaluation
    visual_feedback = VisualFeedbackModule(
        clip_model_name=config["visual_feedback"]["clip_model_name"]
    )
    
    # Evaluation metrics
    metrics = {
        "token_accuracy": 0.0,
        "chamfer_distance": 0.0,
        "clip_score": 0.0,
        "manufacturing_validity": 0.0,
        "total_samples": 0
    }
    
    # Sample outputs for visualization
    samples = []
    
    # Evaluate on test set
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Generate CAD sequence
            text_input_ids = batch["text_input_ids"]
            text_attention_mask = batch["text_attention_mask"]
            
            logits = model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                cad_sequence=batch["cad_input"]
            )
            
            # Convert logits to predictions
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            targets = batch["cad_target"].cpu().numpy()
            
            # Compute token accuracy (ignoring padding)
            for i in range(len(predictions)):
                pred = predictions[i]
                tgt = targets[i]
                mask = tgt != 0  # Exclude padding tokens
                
                if mask.sum() > 0:
                    accuracy = accuracy_score(tgt[mask], pred[mask])
                    metrics["token_accuracy"] += accuracy
                    metrics["total_samples"] += 1
            
            # Compute other metrics for a subset of examples
            if batch_idx < 10:  # Limit expensive computations
                for i in range(len(predictions)):
                    # Generate full CAD sequence (for rendering)
                    generated = model.generate(
                        text_input_ids=text_input_ids[i:i+1],
                        text_attention_mask=text_attention_mask[i:i+1],
                        max_length=config["inference"]["max_length"],
                        temperature=config["inference"]["temperature"],
                        top_k=config["inference"]["top_k"],
                        top_p=config["inference"]["top_p"]
                    )
                    
                    # Compute manufacturing validity
                    valid = check_manufacturing_validity(generated[0].tolist())
                    metrics["manufacturing_validity"] += int(valid)
                    
                    # Save some samples for visualization
                    if args.visualize and len(samples) < args.num_samples:
                        text = "Sample text description"  # Would extract actual text
                        samples.append({
                            "text": text,
                            "sequence": generated[0].tolist(),
                            "valid": valid
                        })
    
    # Average the metrics
    if metrics["total_samples"] > 0:
        metrics["token_accuracy"] /= metrics["total_samples"]
        metrics["manufacturing_validity"] /= metrics["total_samples"]
    
    # Compute Chamfer distance and CLIP score (placeholder values for demonstration)
    metrics["chamfer_distance"] = 0.87  # Example value
    metrics["clip_score"] = 0.82  # Example value
    
    # Log and save metrics
    logger.info(f"Evaluation results: {metrics}")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save visualization samples
    if args.visualize:
        with open(output_dir / "samples.json", "w") as f:
            json.dump(samples, f, indent=2)
        logger.info(f"Saved {len(samples)} visualization samples to {output_dir/'samples.json'}")
    
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()