#!/usr/bin/env python
"""
Inference Script for Text-to-CAD Model
"""

import os
import sys
import argparse
import yaml
import json
import torch
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.text_to_cad import TextToCADModel
from src.inference.pipeline import InferencePipeline, load_model_from_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Generate CAD models from text")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, required=True, help="Text description")
    parser.add_argument("--output", type=str, default="output.step", help="Output file path")
    parser.add_argument("--format", type=str, default="step", choices=["step", "gltf", "kcl"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load model
    model = load_model_from_checkpoint(args.model, args.device)
    pipeline = InferencePipeline(model, device=args.device)

    # Generate
    cad_sequence = pipeline.generate(args.text)

    # Export
    if args.format == "step":
        pipeline.export_step(cad_sequence, args.output)
    elif args.format == "gltf":
        pipeline.export_gltf(cad_sequence, args.output)
    elif args.format == "kcl":
        kcl_code = pipeline.export_kcl(cad_sequence)
        with open(args.output, "w") as f:
            f.write(kcl_code)

    print(f"Generated CAD saved to {args.output}")

if __name__ == "__main__":
    main()
