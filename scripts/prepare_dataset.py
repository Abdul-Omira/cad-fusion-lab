#!/usr/bin/env python
"""
Dataset Preparation Script for Text-to-CAD Model

Downloads, processes, and prepares CAD datasets for training.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.collection_orchestrator import DataCollectionOrchestrator, CollectionConfig

def setup_logging(log_level="INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare CAD dataset for training")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Output directory for processed data")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of samples to generate/process")
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation")
    parser.add_argument("--validate", action="store_true",
                        help="Enable data validation")
    parser.add_argument("--split-ratio", type=str, default="0.7,0.15,0.15",
                        help="Train/val/test split ratio")
    parser.add_argument("--log-level", type=str, default="info",
                        choices=["debug", "info", "warning", "error"])
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting dataset preparation")

    # Parse split ratio
    split_ratio = [float(x) for x in args.split_ratio.split(",")]
    assert sum(split_ratio) == 1.0, "Split ratios must sum to 1.0"

    # Configure data collection
    config = CollectionConfig(
        output_dir=args.output_dir,
        enable_augmentation=args.augment,
        enable_validation=args.validate,
        max_samples_per_source=args.num_samples
    )

    # Initialize orchestrator
    orchestrator = DataCollectionOrchestrator(config)

    # Define data sources (mock data for demonstration)
    data_sources = [
        {"type": "mock", "count": args.num_samples}
    ]

    # Collect and process data
    logger.info(f"Processing {args.num_samples} samples")
    results = orchestrator.collect_and_process_data(data_sources)

    logger.info(f"Collection complete: {results['collection_summary']['success_rate']:.2%} success rate")
    logger.info(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()
