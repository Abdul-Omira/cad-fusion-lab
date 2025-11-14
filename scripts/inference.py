#!/usr/bin/env python
"""
Inference Script for Text-to-CAD Model

A robust CLI tool for generating CAD models from text descriptions.
Supports single/batch generation, multiple formats, validation, and metrics.
"""

import os
import sys
import argparse
import yaml
import json
import torch
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.text_to_cad import TextToCADModel
from src.inference.pipeline import InferencePipeline, load_model_from_checkpoint


class InferenceRunner:
    """Manages CAD model inference with comprehensive error handling and logging."""

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "auto",
        log_level: str = "INFO"
    ):
        """
        Initialize the inference runner.

        Args:
            model_path: Path to model checkpoint
            config_path: Optional path to config file
            device: Device to run on (auto, cuda, cpu)
            log_level: Logging level
        """
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.logger.info(f"Using device: {self.device}")

        # Load configuration
        self.config = self.load_config(config_path)

        # Load model
        self.model = self.load_model(model_path)

        # Initialize pipeline
        pipeline_config = self.config.get("inference", {})
        self.pipeline = InferencePipeline(
            self.model,
            device=self.device,
            config=pipeline_config
        )

        self.logger.info("Inference runner initialized successfully")

    def setup_logging(self, log_level: str):
        """Set up logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler()]
        )

    def load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path and os.path.exists(config_path):
            self.logger.info(f"Loading config from {config_path}")
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        else:
            self.logger.info("No config file specified, using defaults")
            return {}

    def load_model(self, model_path: str) -> TextToCADModel:
        """Load model from checkpoint with fallback."""
        if not os.path.exists(model_path):
            self.logger.warning(f"Model checkpoint not found at {model_path}")
            self.logger.info("Creating a new model with default parameters for testing")
            return TextToCADModel(vocab_size=10000, offline_mode=True)

        try:
            self.logger.info(f"Loading model from {model_path}")
            model = load_model_from_checkpoint(model_path, self.device)
            self.logger.info("Model loaded successfully")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.logger.info("Creating fallback model")
            return TextToCADModel(vocab_size=10000, offline_mode=True)

    def generate_single(
        self,
        text: str,
        output_path: str,
        format_type: str = "step",
        validate: bool = False,
        compute_metrics: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a single CAD model.

        Args:
            text: Text description
            output_path: Output file path
            format_type: Output format (step, gltf, kcl)
            validate: Whether to validate geometry
            compute_metrics: Whether to compute CLIP score

        Returns:
            Dictionary with generation results
        """
        self.logger.info("=" * 70)
        self.logger.info(f"Generating CAD from: '{text}'")
        self.logger.info("=" * 70)

        start_time = datetime.now()

        try:
            # Generate CAD sequence
            self.logger.info("Generating CAD sequence...")
            cad_sequence = self.pipeline.generate(text)
            self.logger.info(f"Generated sequence with {len(cad_sequence)} tokens")

            # Create output directory if needed
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.info(f"Created output directory: {output_dir}")

            # Export to specified format
            self.logger.info(f"Exporting to {format_type.upper()} format...")
            if format_type == "step":
                file_path = self.pipeline.export_step(cad_sequence, output_path)
            elif format_type == "gltf":
                file_path = self.pipeline.export_gltf(cad_sequence, output_path)
            elif format_type == "kcl":
                kcl_code = self.pipeline.export_kcl(cad_sequence)
                with open(output_path, "w") as f:
                    f.write(kcl_code)
                file_path = output_path
            else:
                raise ValueError(f"Unsupported format: {format_type}")

            # Build result
            elapsed = (datetime.now() - start_time).total_seconds()
            result = {
                "text": text,
                "file_path": file_path,
                "format": format_type,
                "sequence_length": len(cad_sequence),
                "generation_time_seconds": elapsed,
                "success": True
            }

            # Validate if requested
            if validate:
                self.logger.info("Validating geometry...")
                is_valid, errors = self.pipeline.validator.validate(cad_sequence)
                result["valid"] = is_valid
                result["validation_errors"] = errors
                if is_valid:
                    self.logger.info("✓ Validation PASSED")
                else:
                    self.logger.warning(f"✗ Validation FAILED: {len(errors)} errors")
                    for error in errors[:3]:  # Show first 3 errors
                        self.logger.warning(f"  - {error}")

            # Compute metrics if requested
            if compute_metrics:
                self.logger.info("Computing CLIP score...")
                try:
                    clip_score = self.pipeline.compute_visual_score(cad_sequence, text)
                    result["clip_score"] = clip_score
                    self.logger.info(f"CLIP Score: {clip_score:.4f}")
                except Exception as e:
                    self.logger.warning(f"Failed to compute CLIP score: {e}")
                    result["clip_score"] = None

            # Log summary
            self.logger.info("=" * 70)
            self.logger.info("✓ Generation Complete!")
            self.logger.info(f"  Output: {file_path}")
            self.logger.info(f"  Format: {format_type.upper()}")
            self.logger.info(f"  Sequence Length: {len(cad_sequence)} tokens")
            self.logger.info(f"  Generation Time: {elapsed:.2f}s")
            if validate:
                self.logger.info(f"  Valid: {'✓ Yes' if result.get('valid') else '✗ No'}")
            if compute_metrics and result.get("clip_score"):
                self.logger.info(f"  CLIP Score: {result['clip_score']:.4f}")
            self.logger.info("=" * 70)

            return result

        except Exception as e:
            self.logger.error(f"Generation failed: {e}", exc_info=True)
            return {
                "text": text,
                "success": False,
                "error": str(e)
            }

    def generate_batch(
        self,
        descriptions: List[str],
        output_dir: str,
        format_type: str = "step",
        validate: bool = False,
        compute_metrics: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple CAD models.

        Args:
            descriptions: List of text descriptions
            output_dir: Output directory
            format_type: Output format
            validate: Whether to validate
            compute_metrics: Whether to compute metrics

        Returns:
            List of results
        """
        self.logger.info("=" * 70)
        self.logger.info(f"Batch Generation: {len(descriptions)} models")
        self.logger.info("=" * 70)

        os.makedirs(output_dir, exist_ok=True)
        results = []

        for idx, text in enumerate(descriptions, 1):
            self.logger.info(f"\n[{idx}/{len(descriptions)}] Processing: '{text[:50]}...'")

            output_path = os.path.join(
                output_dir,
                f"model_{idx:04d}.{format_type}"
            )

            result = self.generate_single(
                text, output_path, format_type,
                validate, compute_metrics
            )
            result["index"] = idx
            results.append(result)

        # Summary
        success_count = sum(1 for r in results if r.get("success", False))
        valid_count = sum(1 for r in results if r.get("valid", False))

        self.logger.info("\n" + "=" * 70)
        self.logger.info("Batch Generation Summary")
        self.logger.info("=" * 70)
        self.logger.info(f"Total Models: {len(results)}")
        self.logger.info(f"Successful: {success_count} ({success_count/len(results)*100:.1f}%)")
        self.logger.info(f"Failed: {len(results) - success_count}")
        if validate:
            self.logger.info(f"Valid Geometry: {valid_count} ({valid_count/len(results)*100:.1f}%)")
        if compute_metrics:
            scores = [r.get("clip_score", 0) for r in results if r.get("clip_score")]
            if scores:
                self.logger.info(f"Average CLIP Score: {sum(scores)/len(scores):.4f}")
        self.logger.info("=" * 70)

        # Save results to JSON
        summary_path = os.path.join(output_dir, "generation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"\nResults saved to: {summary_path}")

        return results


def load_descriptions_from_file(file_path: str) -> List[str]:
    """Load text descriptions from file (one per line)."""
    descriptions = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                descriptions.append(line)
    return descriptions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate CAD models from text descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a single CAD model
  python scripts/inference.py \\
      --model checkpoints/final_model.pt \\
      --text "Create a rectangular bracket with mounting holes" \\
      --output outputs/bracket.step

  # Generate with validation and metrics
  python scripts/inference.py \\
      --model checkpoints/final_model.pt \\
      --text "A cylindrical housing" \\
      --output housing.gltf \\
      --format gltf \\
      --validate \\
      --metrics

  # Batch generation from file
  python scripts/inference.py \\
      --model checkpoints/final_model.pt \\
      --batch descriptions.txt \\
      --output-dir outputs/generated \\
      --format step \\
      --validate

  # Use default model (for testing without checkpoint)
  python scripts/inference.py \\
      --model checkpoints/model.pt \\
      --text "A simple cube" \\
      --output test.kcl \\
      --format kcl
"""
    )

    # Model and config
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to model checkpoint (will create default if not found)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file (optional)"
    )

    # Input (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", type=str,
        help="Single text description to generate"
    )
    input_group.add_argument(
        "--batch", type=str,
        help="File with text descriptions (one per line)"
    )

    # Output
    parser.add_argument(
        "--output", type=str,
        help="Output file path (for single generation)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/generated",
        help="Output directory (for batch generation)"
    )
    parser.add_argument(
        "--format", type=str, default="step",
        choices=["step", "gltf", "kcl"],
        help="Output format (default: step)"
    )

    # Options
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate generated geometry"
    )
    parser.add_argument(
        "--metrics", action="store_true",
        help="Compute CLIP score and other metrics"
    )

    # System
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on (default: auto)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize runner
    runner = InferenceRunner(
        model_path=args.model,
        config_path=args.config,
        device=args.device,
        log_level=args.log_level
    )

    # Single or batch generation
    if args.text:
        # Single generation
        output_path = args.output or f"output.{args.format}"

        result = runner.generate_single(
            text=args.text,
            output_path=output_path,
            format_type=args.format,
            validate=args.validate,
            compute_metrics=args.metrics
        )

        # Exit with appropriate code
        sys.exit(0 if result.get("success", False) else 1)

    else:
        # Batch generation
        descriptions = load_descriptions_from_file(args.batch)
        runner.logger.info(f"Loaded {len(descriptions)} descriptions from {args.batch}")

        results = runner.generate_batch(
            descriptions=descriptions,
            output_dir=args.output_dir,
            format_type=args.format,
            validate=args.validate,
            compute_metrics=args.metrics
        )

        # Exit with appropriate code
        success_count = sum(1 for r in results if r.get("success", False))
        sys.exit(0 if success_count == len(results) else 1)


if __name__ == "__main__":
    main()
