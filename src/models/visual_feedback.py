"""
Visual Feedback Module for Text-to-CAD Model

Implements CADFusion-style visual feedback using CLIP-based scoring
and geometric validation for improved CAD generation quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from typing import Optional, Dict, List, Tuple
import numpy as np
from abc import ABC, abstractmethod
from src.validation.geometric import KCLGenerator
from src.models.cad_kernel_interface import CADKernelInterface


class CADRenderer(ABC):
    """Abstract base class for CAD rendering systems."""
    
    @abstractmethod
    def render(self, cad_sequence: List[int]) -> torch.Tensor:
        """Render CAD sequence to RGB image."""
        pass


class OccwlRenderer:
    """
    Renders CAD models using a conceptual CAD kernel for visual feedback.
    """
    def __init__(self, kcl_generator: KCLGenerator, cad_kernel: CADKernelInterface, image_size: tuple[int, int] = (256, 256)):
        """
        Initializes the OccwlRenderer.

        Args:
            kcl_generator: An instance of KCLGenerator to produce KCL code.
            cad_kernel: An instance of a CADKernelInterface to execute KCL and render.
            image_size: A tuple (width, height) for the rendered image.
        """
        self.kcl_generator = kcl_generator
        self.cad_kernel = cad_kernel
        self.image_size = image_size
        # Placeholder for a real rendering pipeline initialization if needed
        print(f"OccwlRenderer initialized with image size: {image_size}")

    def render(self, cad_sequence: list[str]) -> torch.Tensor:
        """
        Renders a CAD model from a parametric sequence.

        Args:
            cad_sequence: A list of strings representing the parametric CAD sequence.

        Returns:
            A torch.Tensor representing the rendered image (batch_size, channels, height, width).
            Currently returns a placeholder tensor.
        """
        # 1. Generate KCL code from the CAD sequence
        kcl_code = self.kcl_generator.sequence_to_kcl(cad_sequence)
        print(f"Generated KCL code: \\n{kcl_code}")

        # 2. Execute KCL code using the CAD Kernel to get a 3D model
        # This is a conceptual step. The actual model representation will depend on the kernel.
        try:
            conceptual_model = self.cad_kernel.execute_kcl(kcl_code)
            print("KCL code executed successfully by CAD Kernel (conceptual).")
        except Exception as e:
            print(f"Error executing KCL code with CAD Kernel: {e}")
            # Return a zero tensor or handle error appropriately
            return torch.zeros(1, 3, self.image_size[1], self.image_size[0])

        # 3. Render the 3D model to an image tensor using the CAD Kernel
        # This is also a conceptual step. The actual rendering will depend on the kernel.
        try:
            image_tensor = self.cad_kernel.render_model(conceptual_model, self.image_size)
            print(f"Model rendered to tensor of shape: {image_tensor.shape}")
        except Exception as e:
            print(f"Error rendering model with CAD Kernel: {e}")
            # Return a zero tensor or handle error appropriately
            return torch.zeros(1, 3, self.image_size[1], self.image_size[0])
        
        # For now, let's assume the cad_kernel.render_model returns a correctly shaped tensor.
        # If it's a placeholder, we might need to adjust.
        # Ensure the output is (batch_size, channels, height, width)
        # If image_tensor is (H, W, C), permute it. If it's (C, H, W), unsqueeze it.
        # This part depends on the actual output of your conceptual render_model.
        # As a placeholder, we'll continue to return a random tensor of the correct shape.
        # TODO: Replace with actual image tensor from cad_kernel.render_model
        # return torch.rand(1, 3, self.image_size[1], self.image_size[0])
        
        # Assuming render_model returns a (C, H, W) tensor, add batch dimension
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Validate shape
        if image_tensor.shape != (1, 3, self.image_size[1], self.image_size[0]):
            print(f"Warning: Rendered tensor shape {image_tensor.shape} does not match expected shape (1, 3, {self.image_size[1]}, {self.image_size[0]}). Returning placeholder.")
            return torch.rand(1, 3, self.image_size[1], self.image_size[0])
            
        return image_tensor


class VisualReward(nn.Module):
    """
    Visual feedback module using CLIP for text-image alignment.
    
    Computes reward: r = CLIP-score(R(C), T) - λ * CD(R(C), R(C_gt))
    where R is renderer, C is CAD sequence, T is text, CD is Chamfer distance
    """
    
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        renderer: Optional[CADRenderer] = None,
        chamfer_weight: float = 0.1
    ):
        super().__init__()
        
        # CLIP model for visual-text alignment
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # CAD renderer
        self.renderer = renderer or OccwlRenderer()
        
        # Hyperparameters
        self.chamfer_weight = chamfer_weight
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        cad_sequence: List[int],
        text_prompt: str,
        ground_truth_sequence: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute visual reward for CAD sequence.
        
        Args:
            cad_sequence: Generated CAD sequence
            text_prompt: Original text description
            ground_truth_sequence: Ground truth CAD sequence (for Chamfer distance)
            
        Returns:
            Reward scalar tensor
        """
        # Render CAD sequence
        rendered_image = self.renderer.render(cad_sequence)
        
        # Process inputs for CLIP
        inputs = self.clip_processor(
            text=[text_prompt],
            images=[rendered_image.permute(1, 2, 0).numpy()],
            return_tensors="pt",
            padding=True
        )
        
        # Compute CLIP score
        outputs = self.clip_model(**inputs)
        clip_score = outputs.logits_per_image[0, 0]  # Text-image similarity
        
        # Compute Chamfer distance if ground truth is available
        chamfer_distance = 0.0
        if ground_truth_sequence is not None:
            gt_rendered = self.renderer.render(ground_truth_sequence)
            chamfer_distance = self._compute_chamfer_distance(rendered_image, gt_rendered)
        
        # Combined reward
        reward = clip_score - self.chamfer_weight * chamfer_distance
        
        return reward
    
    def _compute_chamfer_distance(
        self,
        pred_image: torch.Tensor,
        gt_image: torch.Tensor
    ) -> float:
        """
        Compute approximate Chamfer distance between rendered images.
        
        Args:
            pred_image: Predicted rendered image (3, H, W)
            gt_image: Ground truth rendered image (3, H, W)
            
        Returns:
            Chamfer distance approximation
        """
        # Simplified Chamfer distance using L2 norm
        # In practice, would extract 3D points and compute actual Chamfer distance
        diff = torch.norm(pred_image - gt_image, p=2)
        return diff.item()


class GeometricValidator(nn.Module):
    """Geometric validation module for CAD sequences."""
    
    def __init__(self, min_thickness: float = 0.5):
        super().__init__()
        self.min_thickness = min_thickness
    
    def validate_sequence(self, cad_sequence: List[int]) -> Dict[str, bool]:
        """
        Validate geometric constraints of CAD sequence.
        
        Args:
            cad_sequence: CAD operation sequence
            
        Returns:
            Dictionary of validation results
        """
        results = {
            'valid_syntax': self._check_syntax(cad_sequence),
            'valid_thickness': self._check_wall_thickness(cad_sequence),
            'valid_topology': self._check_topology(cad_sequence),
            'manufacturable': True  # Placeholder
        }
        
        results['overall_valid'] = all(results.values())
        return results
    
    def _check_syntax(self, cad_sequence: List[int]) -> bool:
        """Check if CAD sequence has valid syntax."""
        # Placeholder implementation
        # Would check for proper operation ordering, balanced brackets, etc.
        return len(cad_sequence) > 0
    
    def _check_wall_thickness(self, cad_sequence: List[int]) -> bool:
        """Check if generated geometry meets minimum wall thickness."""
        # Placeholder implementation
        # Would extract 3D mesh and measure wall thickness
        return True
    
    def _check_topology(self, cad_sequence: List[int]) -> bool:
        """Check if geometry has valid topology (no self-intersections, etc.)."""
        # Placeholder implementation
        # Would perform topological analysis of generated mesh
        return True


class VisualFeedbackModule(nn.Module):
    """
    Complete visual feedback module combining CLIP scoring and geometric validation.
    """
    
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        renderer: Optional[CADRenderer] = None,
        validator: Optional[GeometricValidator] = None,
        clip_weight: float = 1.0,
        geometry_weight: float = 0.5
    ):
        super().__init__()
        
        self.visual_reward = VisualReward(clip_model_name, renderer)
        self.geometric_validator = validator or GeometricValidator()
        
        self.clip_weight = clip_weight
        self.geometry_weight = geometry_weight
    
    def compute_feedback(
        self,
        cad_sequence: List[int],
        text_prompt: str,
        ground_truth_sequence: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive feedback for CAD sequence.
        
        Args:
            cad_sequence: Generated CAD sequence
            text_prompt: Original text description
            ground_truth_sequence: Ground truth CAD sequence
            
        Returns:
            Dictionary containing various feedback scores
        """
        # Visual reward from CLIP
        visual_reward = self.visual_reward(
            cad_sequence, text_prompt, ground_truth_sequence
        )
        
        # Geometric validation
        validation_results = self.geometric_validator.validate_sequence(cad_sequence)
        geometry_score = float(validation_results['overall_valid'])
        
        # Combined feedback
        total_reward = (
            self.clip_weight * visual_reward +
            self.geometry_weight * geometry_score
        )
        
        return {
            'visual_reward': visual_reward.item(),
            'geometry_score': geometry_score,
            'total_reward': total_reward.item(),
            'validation_details': validation_results
        }
    
    def adaptive_weight_schedule(self, epoch: int, total_epochs: int) -> float:
        """
        Adaptive weighting between visual and parametric feedback.
        
        λ(t) = 1 - exp(-kt) where k controls visual/parametric balance
        
        Args:
            epoch: Current training epoch
            total_epochs: Total number of training epochs
            
        Returns:
            Adaptive weight value
        """
        k = 5.0  # Controls rate of weight change
        t = epoch / total_epochs
        return 1.0 - torch.exp(torch.tensor(-k * t)).item()


def error_to_message(error: Exception) -> str:
    """
    Convert geometric validation error to human-readable message.
    
    Args:
        error: Exception from geometric validation
        
    Returns:
        Human-readable error message
    """
    error_messages = {
        'SyntaxError': 'Invalid CAD operation sequence',
        'TopologyError': 'Generated geometry has invalid topology',
        'ThicknessError': 'Wall thickness below manufacturing limits',
        'IntersectionError': 'Self-intersecting geometry detected'
    }
    
    error_type = type(error).__name__
    return error_messages.get(error_type, f'Unknown error: {str(error)}')
