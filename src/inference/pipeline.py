"""
Inference Pipeline for Text-to-CAD Model

Provides a complete pipeline for:
1. Text processing
2. CAD sequence generation
3. Geometric validation
4. Format conversion (STEP, GLTF, KCL)
"""

import torch
from transformers import BertTokenizer
import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import json

from src.models.text_to_cad import TextToCADModel
from src.validation.geometric import GeometricValidator, KCLGenerator
from src.models.visual_feedback import VisualReward


class InferencePipeline:
    """
    Complete pipeline for text-to-CAD inference.
    """
    
    def __init__(
        self,
        model: TextToCADModel,
        tokenizer_name: str = "bert-base-uncased",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config: Optional[Dict[str, Any]] = None
    ):
        self.model = model.to(device)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.device = device
        self.model.eval()
        
        # Set up geometric validator
        self.validator = GeometricValidator()
        self.kcl_generator = KCLGenerator()
        
        # For visual feedback
        self.visual_reward = VisualReward()
        
        # Config for generation
        self.config = config or {
            "max_length": 512,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95,
            "validate_output": True
        }
        
        self.logger = logging.getLogger(__name__)
    
    @torch.no_grad()
    def generate(self, text_description: str) -> List[int]:
        """
        Generate CAD sequence from text description.
        
        Args:
            text_description: Natural language description of CAD model
            
        Returns:
            List of CAD operation tokens
        """
        self.logger.info(f"Generating CAD sequence from: '{text_description}'")
        
        # Tokenize input text
        text_tokens = self.tokenizer(
            text_description,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.cad_decoder.max_seq_length
        )
        text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}
        
        # Generate CAD sequence
        generated = self.model.generate(
            text_input_ids=text_tokens["input_ids"],
            text_attention_mask=text_tokens["attention_mask"],
            max_length=self.config.get("max_length", 512),
            temperature=self.config.get("temperature", 0.8),
            top_k=self.config.get("top_k", 50),
            top_p=self.config.get("top_p", 0.95)
        )
        
        # Convert to list of integers
        cad_sequence = generated[0].cpu().tolist()
        
        # Validate if requested
        if self.config.get("validate_output", True):
            is_valid, errors = self.validator.validate(cad_sequence)
            if not is_valid:
                self.logger.warning(f"Generated CAD sequence has validation errors: {errors}")
                # In real implementation, might handle validation errors
        
        self.logger.info(f"Successfully generated CAD sequence with {len(cad_sequence)} tokens")
        return cad_sequence
    
    def export_kcl(self, cad_sequence: List[int]) -> str:
        """
        Export CAD sequence as KCL code.
        
        Args:
            cad_sequence: List of CAD operation tokens
            
        Returns:
            KCL code as string
        """
        return self.kcl_generator.generate_kcl(cad_sequence)
    
    def export_step(self, cad_sequence: List[int], output_path: Optional[str] = None) -> str:
        """
        Export CAD sequence as STEP file.
        
        Args:
            cad_sequence: List of CAD operation tokens
            output_path: Path to save STEP file (if None, uses temp file)
            
        Returns:
            Path to saved STEP file
        """
        self.logger.info("Exporting to STEP format")
        
        # Convert sequence to KCL first
        kcl_code = self.export_kcl(cad_sequence)
        
        # In a real implementation, would use CAD kernel to export STEP
        # Here we just save the KCL code to demonstrate the pipeline
        
        if output_path is None:
            # Create a temporary file
            fd, output_path = tempfile.mkstemp(suffix=".step")
            os.close(fd)
        
        # Mock STEP export
        with open(output_path, "w") as f:
            f.write(f"ISO-10303-21;\nHEADER;\n{kcl_code}\nENDSEC;\nEND-ISO-10303-21;\n")
        
        self.logger.info(f"Exported STEP file to {output_path}")
        return output_path
    
    def export_gltf(self, cad_sequence: List[int], output_path: Optional[str] = None) -> str:
        """
        Export CAD sequence as GLTF file for web viewing.
        
        Args:
            cad_sequence: List of CAD operation tokens
            output_path: Path to save GLTF file (if None, uses temp file)
            
        Returns:
            Path to saved GLTF file
        """
        self.logger.info("Exporting to GLTF format")
        
        if output_path is None:
            # Create a temporary file
            fd, output_path = tempfile.mkstemp(suffix=".gltf")
            os.close(fd)
        
        # Mock GLTF export - in real implementation would use a conversion library
        gltf_data = {
            "asset": {
                "version": "2.0",
                "generator": "Text-to-CAD GLTF Exporter"
            },
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
            "bufferViews": [],
            "buffers": []
        }
        
        with open(output_path, "w") as f:
            json.dump(gltf_data, f, indent=2)
        
        self.logger.info(f"Exported GLTF file to {output_path}")
        return output_path
    
    def compute_visual_score(self, cad_sequence: List[int], text_prompt: str) -> float:
        """
        Compute visual-text alignment score using CLIP.
        
        Args:
            cad_sequence: CAD operation token sequence
            text_prompt: Original text description
            
        Returns:
            CLIP score (0-1)
        """
        self.logger.info("Computing visual-text alignment score")
        
        # Use the visual reward module
        reward = self.visual_reward(cad_sequence, text_prompt)
        
        self.logger.info(f"CLIP score: {reward.item():.4f}")
        return reward.item()
    
    def batch_generate(self, text_descriptions: List[str]) -> List[List[int]]:
        """
        Generate CAD sequences for multiple text descriptions.
        
        Args:
            text_descriptions: List of text descriptions
            
        Returns:
            List of CAD sequences
        """
        results = []
        
        for text in text_descriptions:
            cad_sequence = self.generate(text)
            results.append(cad_sequence)
        
        return results


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> TextToCADModel:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded TextToCADModel
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    config = checkpoint.get("config", {}).get("model", {})
    
    # Initialize model
    model = TextToCADModel(
        vocab_size=config.get("vocab_size", 10000),
        text_encoder_name=config.get("text_encoder_name", "bert-base-uncased"),
        d_model=config.get("d_model", 512),
        nhead=config.get("nhead", 8),
        num_decoder_layers=config.get("num_decoder_layers", 24),
        dim_feedforward=config.get("dim_feedforward", 2048),
        dropout=config.get("dropout", 0.1),
        max_seq_length=config.get("max_seq_length", 512)
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model
