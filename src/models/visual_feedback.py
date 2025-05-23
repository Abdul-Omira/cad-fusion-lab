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
from src.models.mock_cad_kernel import MockCADKernel # Added import


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
        renderer: Optional[OccwlRenderer] = None, # Changed CADRenderer to OccwlRenderer
        chamfer_weight: float = 0.1,
        kcl_generator: Optional[KCLGenerator] = None, # Added kcl_generator
        cad_kernel: Optional[CADKernelInterface] = None, # Added cad_kernel
        image_size: Tuple[int, int] = (256, 256) # Added image_size for renderer
    ):
        super().__init__()
        
        # CLIP model for visual-text alignment
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # CAD renderer
        if renderer:
            self.renderer = renderer
        else:
            # Initialize KCLGenerator and CADKernel if not provided
            _kcl_generator = kcl_generator if kcl_generator else KCLGenerator()
            _cad_kernel = cad_kernel if cad_kernel else MockCADKernel()
            self.renderer = OccwlRenderer(
                kcl_generator=_kcl_generator,
                cad_kernel=_cad_kernel,
                image_size=image_size
            )
        
        # Hyperparameters
        self.chamfer_weight = chamfer_weight
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        cad_sequence: List[str], # Changed from List[int] to List[str]
        text_prompt: str,
        ground_truth_sequence: Optional[List[str]] = None # Changed from List[int] to List[str]
    ) -> torch.Tensor:
        """
        Compute visual reward for CAD sequence.
        
        Args:
            cad_sequence: Generated CAD sequence (list of strings)
            text_prompt: Original text description
            ground_truth_sequence: Ground truth CAD sequence (list of strings)
            
        Returns:
            Reward scalar tensor
        """
        # Render CAD sequence
        if not all(isinstance(item, str) for item in cad_sequence):
            # This check might be redundant if type hinting is enforced upstream
            # or if conversion happens before this point.
            # For robustness, keeping a check or ensuring conversion.
            try:
                # Attempt to convert if it looks like a list of lists of strings (e.g. batched tokens)
                if all(isinstance(item, list) for item in cad_sequence) and cad_sequence:
                    cad_sequence = [str(token) for sublist in cad_sequence for token in sublist]
                else:
                    cad_sequence = [str(item) for item in cad_sequence]
            except TypeError:
                 raise ValueError("cad_sequence must be a list of strings or convertible to it.")
        
        rendered_image = self.renderer.render(cad_sequence)
        
        # Process inputs for CLIP
        # CLIPProcessor expects images as PIL Images, numpy arrays (H, W, C), or torch tensors (C, H, W).
        # rendered_image from OccwlRenderer is (1, C, H, W).
        if rendered_image.ndim == 4 and rendered_image.shape[0] == 1:
            clip_image_input = rendered_image.squeeze(0)  # Shape: (C, H, W)
        elif rendered_image.ndim == 3: # Already (C,H,W)
            clip_image_input = rendered_image
        else:
            print(f"Warning: Unexpected rendered_image shape: {rendered_image.shape}. Using placeholder for CLIP.")
            # Ensure image_size is available from the renderer instance
            img_h, img_w = self.renderer.image_size[1], self.renderer.image_size[0]
            clip_image_input = torch.rand(3, img_h, img_w, device=self.clip_model.device)

        inputs = self.clip_processor(
            text=[text_prompt],
            images=clip_image_input, # Expects (C, H, W) or list of such, or (H,W,C) numpy
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to the same device as the CLIP model
        inputs = {k: v.to(self.clip_model.device) for k, v in inputs.items()}
        rendered_image = rendered_image.to(self.clip_model.device) # Ensure rendered_image is also on device for Chamfer

        # Compute CLIP score
        outputs = self.clip_model(**inputs)
        clip_score = outputs.logits_per_image[0, 0]  # Text-image similarity
        
        # Compute Chamfer distance if ground truth is available
        chamfer_distance = torch.tensor(0.0, device=self.clip_model.device) # Ensure tensor is on correct device
        if ground_truth_sequence is not None:
            if not all(isinstance(item, str) for item in ground_truth_sequence):
                try:
                    if all(isinstance(item, list) for item in ground_truth_sequence) and ground_truth_sequence:
                        ground_truth_sequence = [str(token) for sublist in ground_truth_sequence for token in sublist]
                    else:
                        ground_truth_sequence = [str(item) for item in ground_truth_sequence]
                except TypeError:
                    raise ValueError("ground_truth_sequence must be a list of strings or convertible to it.")

            gt_rendered = self.renderer.render(ground_truth_sequence)
            gt_rendered = gt_rendered.to(self.clip_model.device)
            chamfer_distance = self._compute_chamfer_distance(rendered_image, gt_rendered)
        
        # Combined reward
        reward = clip_score - self.chamfer_weight * chamfer_distance
        
        return reward
    
    def _compute_chamfer_distance(
        self,
        pred_image: torch.Tensor, # Expects (1, C, H, W) or (C, H, W)
        gt_image: torch.Tensor # Expects (1, C, H, W) or (C, H, W)
    ) -> torch.Tensor: # Return a tensor
        """
        Compute approximate Chamfer distance between rendered images.
        
        Args:
            pred_image: Predicted rendered image tensor.
            gt_image: Ground truth rendered image tensor.
            
        Returns:
            Chamfer distance approximation as a tensor.
        """
        # Simplified Chamfer distance using L2 norm
        # Ensure images are on the same device and have compatible shapes
        
        # If images have a batch dimension of 1, squeeze it
        if pred_image.ndim == 4 and pred_image.shape[0] == 1:
            pred_image = pred_image.squeeze(0)
        if gt_image.ndim == 4 and gt_image.shape[0] == 1:
            gt_image = gt_image.squeeze(0)

        if pred_image.shape != gt_image.shape:
            print(f"Warning: Shape mismatch in _compute_chamfer_distance. Pred: {pred_image.shape}, GT: {gt_image.shape}. Returning zero distance.")
            return torch.tensor(0.0, device=pred_image.device)

        diff = torch.norm(pred_image - gt_image, p=2)
        return diff # Return tensor directly


class GeometricValidator(nn.Module):
    """Geometric validation module for CAD sequences."""
    
    def __init__(self, min_thickness: float = 0.5):
        super().__init__()
        self.min_thickness = min_thickness
    
    def validate_sequence(self, cad_sequence: List[str]) -> Dict[str, bool]: # Changed List[int] to List[str]
        """
        Validate geometric constraints of CAD sequence.
        
        Args:
            cad_sequence: CAD operation sequence (list of strings)
            
        Returns:
            Dictionary of validation results
        """
        # Ensure cad_sequence is List[str] if it comes from an unknown source
        # However, VisualFeedbackModule now attempts to ensure this.
        if not isinstance(cad_sequence, list) or (cad_sequence and not isinstance(cad_sequence[0], str)):
            print(f"Warning: GeometricValidator received unexpected sequence type: {type(cad_sequence)}. Attempting conversion or using empty.")
            if isinstance(cad_sequence, list):
                try:
                    cad_sequence = [str(item) for item in cad_sequence]
                except Exception:
                    cad_sequence = [] # Fallback to empty list if conversion fails
            else:
                cad_sequence = []

        results = {
            'valid_syntax': self._check_syntax(cad_sequence),
            'valid_thickness': self._check_wall_thickness(cad_sequence),
            'valid_topology': self._check_topology(cad_sequence),
            'manufacturable': True  # Placeholder
        }
        
        # Overall validity is true if all individual checks are true
        results['overall_valid'] = all(results[key] for key in ['valid_syntax', 'valid_thickness', 'valid_topology', 'manufacturable'])
        return results
    
    def _check_syntax(self, cad_sequence: List[str]) -> bool: # Changed List[int] to List[str]
        """Check if CAD sequence has valid syntax."""
        # Placeholder implementation
        # Would check for proper operation ordering, balanced brackets, etc.
        # For a list of strings, len() still indicates if there are operations.
        return len(cad_sequence) > 0
    
    def _check_wall_thickness(self, cad_sequence: List[str]) -> bool: # Changed List[int] to List[str]
        """Check if generated geometry meets minimum wall thickness."""
        # Placeholder implementation
        # Would parse sequence, generate model, and measure wall thickness
        # For now, depends only on the sequence existing or some mock logic
        return True
    
    def _check_topology(self, cad_sequence: List[str]) -> bool: # Changed List[int] to List[str]
        """Check if geometry has valid topology (no self-intersections, etc.)."""
        # Placeholder implementation
        # Would parse sequence, generate model, and perform topological analysis
        return True


class VisualFeedbackModule(nn.Module):
    """
    Complete visual feedback module combining CLIP scoring and geometric validation.
    """
    
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        renderer: Optional[OccwlRenderer] = None, # Changed CADRenderer to OccwlRenderer
        validator: Optional[GeometricValidator] = None,
        clip_weight: float = 1.0,
        geometry_weight: float = 0.5,
        kcl_generator: Optional[KCLGenerator] = None, # Added
        cad_kernel: Optional[CADKernelInterface] = None, # Added
        image_size: Tuple[int, int] = (256, 256) # Added
    ):
        super().__init__()
        
        self.visual_reward = VisualReward(
            clip_model_name=clip_model_name, 
            renderer=renderer, 
            chamfer_weight=self.visual_reward.chamfer_weight if hasattr(self.visual_reward, 'chamfer_weight') else 0.1, # Preserve if already set
            kcl_generator=kcl_generator, 
            cad_kernel=cad_kernel, 
            image_size=image_size
        )
        self.geometric_validator = validator or GeometricValidator()
        
        self.clip_weight = clip_weight
        self.geometry_weight = geometry_weight
    
    def compute_feedback(
        self,
        cad_sequence: List[str],
        text_prompt: str,
        ground_truth_sequence: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive feedback for CAD sequence.
        
        Args:
            cad_sequence: Generated CAD sequence (list of strings)
            text_prompt: Original text description
            ground_truth_sequence: Ground truth CAD sequence (list of strings)
            
        Returns:
            Dictionary containing various feedback scores
        """
        # Visual reward from CLIP
        if not all(isinstance(item, str) for item in cad_sequence):
            try:
                if all(isinstance(item, list) for item in cad_sequence) and cad_sequence:
                    cad_sequence = [str(token) for sublist in cad_sequence for token in sublist]
                else:
                    cad_sequence = [str(item) for item in cad_sequence]
            except TypeError:
                 raise ValueError("cad_sequence must be a list of strings or convertible to it for compute_feedback.")
        
        visual_reward_tensor = self.visual_reward(
            cad_sequence, text_prompt, ground_truth_sequence
        )
        visual_reward_value = visual_reward_tensor.item()
        
        # Geometric validation
        # GeometricValidator now expects List[str], so direct pass-through is fine.
        validation_results = self.geometric_validator.validate_sequence(cad_sequence)
        geometry_score = float(validation_results.get('overall_valid', 0.0))
        
        # Combined feedback
        total_reward = (
            self.clip_weight * visual_reward_value +
            self.geometry_weight * geometry_score
        )
        
        return {
            'visual_reward': visual_reward_value,
            'geometry_score': geometry_score,
            'total_reward': total_reward,
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
