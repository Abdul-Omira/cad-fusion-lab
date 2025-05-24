"""
Geometric Validation Module

Validates CAD geometry for:
- Topological correctness
- Manufacturing constraints (wall thickness, etc.)
- Self-intersections
- Watertightness

Generates KCL code for feature-by-feature validation.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import json
import os
from pathlib import Path

# Import CAD kernel libraries
try:
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeShape
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepGProp import brepgprop_VolumeProperties
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.TopoDS import TopoDS_Shape
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeThickSolid
    CAD_KERNEL_AVAILABLE = True
except ImportError:
    CAD_KERNEL_AVAILABLE = False
    if not hasattr(logging, '_opencascade_warning_shown'):
        logging.warning("OpenCascade CAD kernel not available. Using mock validation for this session.")
        logging._opencascade_warning_shown = True


@dataclass
class GeometricError:
    """Base class for geometric validation errors."""
    message: str
    location: Optional[Dict[str, float]] = None
    severity: str = "error"


class SyntaxError(GeometricError):
    """Error in CAD operation sequence syntax."""
    pass


class TopologyError(GeometricError):
    """Error in geometric topology."""
    pass


class ThicknessError(GeometricError):
    """Wall thickness below manufacturing limit."""
    pass


class IntersectionError(GeometricError):
    """Self-intersecting geometry detected."""
    pass


class KCLGenerator:
    """
    Generates KCL (Kernel Command Language) code from CAD sequence tokens.
    
    KCL is a domain-specific language for parametric CAD operations.
    """
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        
        # Special tokens
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
        self.operation_start = 4
        
        # Token to operation mapping (simplified)
        self.operations = {
            4: "Box",
            5: "Cylinder",
            6: "Sphere",
            7: "Cone",
            8: "Translate",
            9: "Rotate",
            10: "Scale",
            11: "Union",
            12: "Difference",
            13: "Intersection",
            14: "Fillet",
            15: "Chamfer",
            16: "Extrude",
            17: "Revolve",
            18: "Shell",
            19: "Draft",
            20: "Pattern",
        }
    
    def generate_kcl(self, cad_sequence: List[int]) -> str:
        """
        Generate KCL code from token sequence.
        
        Args:
            cad_sequence: List of CAD operation tokens
            
        Returns:
            KCL code as string
        """
        if not cad_sequence:
            return ""
        
        # Skip special tokens
        if cad_sequence[0] == self.sos_token:
            cad_sequence = cad_sequence[1:]
        
        if cad_sequence[-1] == self.eos_token:
            cad_sequence = cad_sequence[:-1]
        
        kcl_code = []
        i = 0
        
        # Process operations and parameters
        while i < len(cad_sequence):
            token = cad_sequence[i]
            
            # Skip padding
            if token == self.pad_token:
                i += 1
                continue
            
            # Process operation
            if token >= self.operation_start and token < 1000:
                op_name = self.operations.get(token, f"Operation_{token}")
                
                # Get parameters (simplified)
                params = []
                j = i + 1
                while j < len(cad_sequence) and j < i + 4 and cad_sequence[j] >= 1000:
                    param_value = cad_sequence[j] - 1000
                    params.append(param_value)
                    j += 1
                
                # Generate KCL for this operation
                if params:
                    param_str = ", ".join([str(p) for p in params])
                    kcl_code.append(f"{op_name}({param_str})")
                else:
                    kcl_code.append(f"{op_name}()")
                
                i = j
            else:
                # Skip unknown tokens
                i += 1
        
        return "\n".join(kcl_code)


class GeometricValidator:
    """
    Validates CAD geometry for manufacturability and correctness.
    """
    
    def __init__(
        self, 
        min_thickness: float = 0.5,
        check_topology: bool = True,
        check_self_intersection: bool = True,
        check_watertightness: bool = True,
        mesh_quality: float = 0.1
    ):
        self.min_thickness = min_thickness
        self.check_topology = check_topology
        self.check_self_intersection = check_self_intersection
        self.check_watertightness = check_watertightness
        self.mesh_quality = mesh_quality
        self.kcl_generator = KCLGenerator()
        self.logger = logging.getLogger(__name__)
        
        if not CAD_KERNEL_AVAILABLE:
            self.logger.warning("Using mock validation due to missing CAD kernel")
    
    def validate(self, cad_sequence: List[int]) -> Tuple[bool, List[GeometricError]]:
        """
        Validate CAD sequence for geometric and manufacturing constraints.
        
        Args:
            cad_sequence: CAD operation token sequence
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Generate KCL code
        kcl_code = self.kcl_generator.generate_kcl(cad_sequence)
        
        # Step 1: Check syntax
        syntax_valid, syntax_errors = self._check_syntax(kcl_code)
        errors.extend(syntax_errors)
        
        if not syntax_valid:
            self.logger.warning("Invalid CAD syntax, skipping further validation")
            return False, errors
        
        # Step 2: Execute KCL (mocked for this implementation)
        cad_model = self._execute_kcl(kcl_code)
        
        if cad_model is None:
            errors.append(SyntaxError(message="Failed to generate CAD model"))
            return False, errors
        
        # Step 3: Check wall thickness
        thickness_valid, thickness_errors = self._check_wall_thickness(cad_model)
        errors.extend(thickness_errors)
        
        # Step 4: Check topology if enabled
        if self.check_topology:
            topology_valid, topology_errors = self._check_topology(cad_model)
            errors.extend(topology_errors)
        
        # Step 5: Check self-intersections if enabled
        if self.check_self_intersection:
            intersection_valid, intersection_errors = self._check_self_intersection(cad_model)
            errors.extend(intersection_errors)
        
        # Step 6: Check watertightness if enabled
        if self.check_watertightness:
            watertight_valid, watertight_errors = self._check_watertightness(cad_model)
            errors.extend(watertight_errors)
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _check_syntax(self, kcl_code: str) -> Tuple[bool, List[GeometricError]]:
        """Check KCL syntax."""
        # Simplified syntax check for demonstration
        errors = []
        is_valid = True
        
        if not kcl_code.strip():
            errors.append(SyntaxError(message="Empty KCL code"))
            is_valid = False
        
        # Check for invalid operations (mocked)
        if "Invalid" in kcl_code:
            errors.append(SyntaxError(message="Invalid operation detected"))
            is_valid = False
        
        # Check for balanced parentheses
        if kcl_code.count('(') != kcl_code.count(')'):
            errors.append(SyntaxError(message="Unbalanced parentheses"))
            is_valid = False
        
        return is_valid, errors
    
    def _execute_kcl(self, kcl_code: str) -> Optional["TopoDS_Shape"]:
        """
        Execute KCL code to generate CAD model using OpenCascade.
        
        Args:
            kcl_code: KCL code string
            
        Returns:
            OpenCascade shape object or None if execution fails
        """
        if not CAD_KERNEL_AVAILABLE:
            return self._mock_cad_model()
            
        try:
            # Create a temporary file for the KCL code
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            kcl_file = temp_dir / "temp.kcl"
            
            with open(kcl_file, "w") as f:
                f.write(kcl_code)
            
            # Execute KCL using OpenCascade
            # This is a simplified example - in practice, you'd need a proper KCL interpreter
            shape = self._interpret_kcl(kcl_file)
            
            # Clean up
            kcl_file.unlink()
            
            return shape
            
        except Exception as e:
            self.logger.error(f"Failed to execute KCL: {str(e)}")
            return None
    
    def _check_wall_thickness(self, shape: "TopoDS_Shape") -> Tuple[bool, List[GeometricError]]:
        """
        Check if wall thickness meets manufacturing requirements.
        
        Args:
            shape: OpenCascade shape object
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if not CAD_KERNEL_AVAILABLE:
            return True, []
            
        errors = []
        try:
            # Create mesh for thickness analysis
            mesh = BRepMesh_IncrementalMesh(shape, self.mesh_quality)
            mesh.Perform()
            
            # Analyze thickness at multiple points
            explorer = TopExp_Explorer(shape, TopAbs_FACE)
            while explorer.More():
                face = explorer.Current()
                # Get face normal and center point
                surface = BRep_Tool.Surface(face)
                umin, umax, vmin, vmax = surface.Bounds()
                center_u = (umin + umax) / 2
                center_v = (vmin + vmax) / 2
                center_point = surface.Value(center_u, center_v)
                normal = surface.Normal(center_u, center_v)
                
                # Create offset solid for thickness check
                offset = BRepOffsetAPI_MakeThickSolid()
                offset.MakeThickSolidByJoin(shape, [face], -self.min_thickness, 0.1)
                
                if not offset.IsDone():
                    errors.append(ThicknessError(
                        message=f"Wall thickness below {self.min_thickness}mm",
                        location={"x": center_point.X(), "y": center_point.Y(), "z": center_point.Z()}
                    ))
                
                explorer.Next()
            
            return len(errors) == 0, errors
            
        except Exception as e:
            self.logger.error(f"Thickness check failed: {str(e)}")
            errors.append(ThicknessError(message=f"Thickness check failed: {str(e)}"))
            return False, errors
    
    def _check_topology(self, shape: "TopoDS_Shape") -> Tuple[bool, List[GeometricError]]:
        """
        Check topological correctness of the CAD model.
        
        Args:
            shape: OpenCascade shape object
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if not CAD_KERNEL_AVAILABLE:
            return True, []
            
        errors = []
        try:
            # Create topology analyzer
            analyzer = BRepCheck_Analyzer(shape)
            
            if not analyzer.IsValid():
                errors.append(TopologyError(
                    message="Invalid topology detected",
                    severity="error"
                ))
            
            return len(errors) == 0, errors
            
        except Exception as e:
            self.logger.error(f"Topology check failed: {str(e)}")
            errors.append(TopologyError(message=f"Topology check failed: {str(e)}"))
            return False, errors
    
    def _check_self_intersection(self, shape: "TopoDS_Shape") -> Tuple[bool, List[GeometricError]]:
        """
        Check for self-intersections in the CAD model.
        
        Args:
            shape: OpenCascade shape object
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if not CAD_KERNEL_AVAILABLE:
            return True, []
            
        errors = []
        try:
            # Create section analyzer
            section = BRepAlgoAPI_Section(shape, shape)
            section.Build()
            
            if section.IsDone() and section.Shape().NbChildren() > 0:
                errors.append(IntersectionError(
                    message="Self-intersection detected",
                    severity="error"
                ))
            
            return len(errors) == 0, errors
            
        except Exception as e:
            self.logger.error(f"Intersection check failed: {str(e)}")
            errors.append(IntersectionError(message=f"Intersection check failed: {str(e)}"))
            return False, errors
    
    def _check_watertightness(self, shape: "TopoDS_Shape") -> Tuple[bool, List[GeometricError]]:
        """
        Check if the CAD model is watertight.
        
        Args:
            shape: OpenCascade shape object
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if not CAD_KERNEL_AVAILABLE:
            return True, []
            
        errors = []
        try:
            # Check volume properties
            props = GProp_GProps()
            brepgprop_VolumeProperties(shape, props)
            
            if props.Mass() <= 0:
                errors.append(GeometricError(
                    message="Model is not watertight (zero or negative volume)",
                    severity="error"
                ))
            
            return len(errors) == 0, errors
            
        except Exception as e:
            self.logger.error(f"Watertightness check failed: {str(e)}")
            errors.append(GeometricError(message=f"Watertightness check failed: {str(e)}"))
            return False, errors
    
    def _mock_cad_model(self) -> Dict[str, Any]:
        """Return a mock CAD model for testing when CAD kernel is not available."""
        return {
            "type": "mock",
            "valid": True,
            "properties": {
                "volume": 1000.0,
                "surface_area": 600.0,
                "bounding_box": {
                    "min": [0, 0, 0],
                    "max": [10, 10, 10]
                }
            }
        }
    
    def export_validation_report(
        self,
        cad_sequence: List[int],
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        Export validation results to a file.
        
        Args:
            cad_sequence: CAD operation token sequence
            output_path: Path to save the report
            format: Output format ("json" or "html")
        """
        is_valid, errors = self.validate(cad_sequence)
        
        if format == "json":
            report = {
                "is_valid": is_valid,
                "errors": [
                    {
                        "type": error.__class__.__name__,
                        "message": error.message,
                        "location": error.location,
                        "severity": error.severity
                    }
                    for error in errors
                ]
            }
            
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
                
        elif format == "html":
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>CAD Validation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .error {{ color: red; }}
                    .warning {{ color: orange; }}
                    .success {{ color: green; }}
                </style>
            </head>
            <body>
                <h1>CAD Validation Report</h1>
                <p class="{'success' if is_valid else 'error'}">
                    Status: {'Valid' if is_valid else 'Invalid'}
                </p>
                <h2>Errors</h2>
                <ul>
            """
            
            for error in errors:
                html += f"""
                    <li class="{error.severity}">
                        <strong>{error.__class__.__name__}:</strong> {error.message}
                        {f'<br>Location: {error.location}' if error.location else ''}
                    </li>
                """
            
            html += """
                </ul>
            </body>
            </html>
            """
            
            with open(output_path, "w") as f:
                f.write(html)
        
        else:
            raise ValueError(f"Unsupported format: {format}")


def check_manufacturing_validity(cad_sequence: List[int]) -> bool:
    """
    Check if generated CAD model is manufacturable.
    
    Args:
        cad_sequence: CAD operation token sequence
        
    Returns:
        Whether the model is valid for manufacturing
    """
    validator = GeometricValidator()
    is_valid, _ = validator.validate(cad_sequence)
    return is_valid