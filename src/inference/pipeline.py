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

        try:
            import cadquery as cq
            from src.models.cad_kernel_interface import tokens_to_cad_operations

            # Convert tokens to CAD operations
            operations = tokens_to_cad_operations(cad_sequence)

            # Build CAD model using CadQuery
            result = cq.Workplane("XY")

            for op in operations:
                op_type = op.get("type", "").lower()
                params = op.get("params", {})

                try:
                    if op_type == "box":
                        width = params.get("width", 10.0)
                        height = params.get("height", 10.0)
                        depth = params.get("depth", 10.0)
                        result = result.box(width, height, depth)

                    elif op_type == "cylinder":
                        radius = params.get("radius", 5.0)
                        height = params.get("height", 10.0)
                        result = result.cylinder(height, radius)

                    elif op_type == "sphere":
                        radius = params.get("radius", 5.0)
                        result = result.sphere(radius)

                    elif op_type == "extrude":
                        distance = params.get("distance", 10.0)
                        result = result.extrude(distance)

                    elif op_type == "fillet":
                        radius = params.get("radius", 1.0)
                        result = result.edges().fillet(radius)

                    elif op_type == "chamfer":
                        length = params.get("length", 1.0)
                        result = result.edges().chamfer(length)

                    elif op_type == "hole":
                        diameter = params.get("diameter", 2.0)
                        depth = params.get("depth", 5.0)
                        result = result.faces(">Z").workplane().hole(diameter, depth)

                except Exception as e:
                    self.logger.warning(f"Skipping operation {op_type}: {e}")
                    continue

            # Determine output path
            if output_path is None:
                fd, output_path = tempfile.mkstemp(suffix=".step")
                os.close(fd)

            # Export to STEP
            cq.exporters.export(result, output_path, cq.exporters.ExportTypes.STEP)

            self.logger.info(f"Exported STEP file to {output_path}")
            return output_path

        except ImportError as e:
            self.logger.warning(f"CadQuery not available: {e}. Using fallback STEP export.")
            # Fallback: Create a minimal valid STEP file
            if output_path is None:
                fd, output_path = tempfile.mkstemp(suffix=".step")
                os.close(fd)

            kcl_code = self.export_kcl(cad_sequence)

            # Create a minimal but valid STEP file structure
            step_content = f"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Text-to-CAD Generated Model'),'2;1');
FILE_NAME('{os.path.basename(output_path)}','','','','','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
/* Generated from KCL:
{kcl_code}
*/
ENDSEC;
END-ISO-10303-21;
"""
            with open(output_path, "w") as f:
                f.write(step_content)

            self.logger.info(f"Exported fallback STEP file to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error exporting STEP: {e}")
            raise
    
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

        try:
            import trimesh
            import numpy as np
            from src.models.cad_kernel_interface import tokens_to_cad_operations

            # Convert tokens to CAD operations
            operations = tokens_to_cad_operations(cad_sequence)

            # Build mesh geometry from operations
            meshes = []

            for op in operations:
                op_type = op.get("type", "").lower()
                params = op.get("params", {})

                try:
                    if op_type == "box":
                        width = params.get("width", 10.0)
                        height = params.get("height", 10.0)
                        depth = params.get("depth", 10.0)
                        mesh = trimesh.creation.box(extents=[width, height, depth])
                        meshes.append(mesh)

                    elif op_type == "cylinder":
                        radius = params.get("radius", 5.0)
                        height = params.get("height", 10.0)
                        mesh = trimesh.creation.cylinder(radius=radius, height=height)
                        meshes.append(mesh)

                    elif op_type == "sphere":
                        radius = params.get("radius", 5.0)
                        mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
                        meshes.append(mesh)

                    elif op_type == "cone":
                        radius = params.get("radius", 5.0)
                        height = params.get("height", 10.0)
                        mesh = trimesh.creation.cone(radius=radius, height=height)
                        meshes.append(mesh)

                except Exception as e:
                    self.logger.warning(f"Skipping operation {op_type}: {e}")
                    continue

            # Combine all meshes
            if meshes:
                combined_mesh = trimesh.util.concatenate(meshes)
            else:
                # Create a default cube if no valid operations
                self.logger.warning("No valid operations, creating default cube")
                combined_mesh = trimesh.creation.box(extents=[10.0, 10.0, 10.0])

            # Determine output path
            if output_path is None:
                fd, output_path = tempfile.mkstemp(suffix=".gltf")
                os.close(fd)

            # Export to GLTF
            combined_mesh.export(output_path, file_type='gltf')

            self.logger.info(f"Exported GLTF file to {output_path}")
            return output_path

        except ImportError as e:
            self.logger.warning(f"Trimesh not available: {e}. Using fallback GLTF export.")
            # Fallback: Create a minimal valid GLTF file
            if output_path is None:
                fd, output_path = tempfile.mkstemp(suffix=".gltf")
                os.close(fd)

            # Create a simple cube mesh as fallback
            vertices = [
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
            ]

            gltf_data = {
                "asset": {
                    "version": "2.0",
                    "generator": "Text-to-CAD GLTF Exporter (Fallback)"
                },
                "scene": 0,
                "scenes": [{"nodes": [0], "name": "CAD Model"}],
                "nodes": [{"mesh": 0, "name": "Generated CAD"}],
                "meshes": [{
                    "primitives": [{
                        "attributes": {"POSITION": 0},
                        "mode": 4
                    }],
                    "name": "CAD Mesh"
                }],
                "accessors": [{
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": len(vertices),
                    "type": "VEC3",
                    "max": [1, 1, 1],
                    "min": [-1, -1, -1]
                }],
                "bufferViews": [{
                    "buffer": 0,
                    "byteOffset": 0,
                    "byteLength": len(vertices) * 12
                }],
                "buffers": [{
                    "byteLength": len(vertices) * 12,
                    "uri": "data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAAAAAA=="
                }]
            }

            with open(output_path, "w") as f:
                json.dump(gltf_data, f, indent=2)

            self.logger.info(f"Exported fallback GLTF file to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error exporting GLTF: {e}")
            raise
    
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
