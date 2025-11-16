"""
CAD-Specific Tokenizer

Implements structured tokenization for CAD operations following Text2CAD
and CADFusion approaches. Converts between CAD operations and token sequences.
"""

import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class CADOperation:
    """Structured CAD operation with parameters and constraints."""
    operation_type: str  # e.g., 'sketch', 'extrude', 'fillet', 'chamfer'
    sketch_type: Optional[str] = None  # e.g., 'rectangle', 'circle', 'polygon'
    parameters: Dict[str, float] = None  # e.g., {'width': 10, 'height': 5}
    constraints: List[str] = None  # e.g., ['parallel_to_xy', 'centered']
    plane: Optional[str] = None  # e.g., 'XY', 'XZ', 'YZ'

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.constraints is None:
            self.constraints = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation_type': self.operation_type,
            'sketch_type': self.sketch_type,
            'parameters': self.parameters,
            'constraints': self.constraints,
            'plane': self.plane
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CADOperation':
        """Create from dictionary."""
        return cls(**data)


class CADVocabulary:
    """
    CAD-specific vocabulary following modern best practices.

    Based on research from Text2CAD, CADFusion, and GenCAD.
    """

    # Operation types
    OPERATIONS = [
        'sketch', 'extrude', 'revolve', 'sweep', 'loft',
        'fillet', 'chamfer', 'hole', 'shell', 'draft',
        'pattern_linear', 'pattern_circular', 'mirror',
        'boolean_union', 'boolean_subtract', 'boolean_intersect'
    ]

    # Sketch types
    SKETCHES = [
        'rectangle', 'circle', 'polygon', 'ellipse', 'arc',
        'line', 'spline', 'point', 'construction_line'
    ]

    # Planes
    PLANES = ['XY', 'XZ', 'YZ', 'custom']

    # Constraints
    CONSTRAINTS = [
        'horizontal', 'vertical', 'parallel', 'perpendicular',
        'tangent', 'concentric', 'coincident', 'midpoint',
        'symmetric', 'equal', 'fixed'
    ]

    # Special tokens
    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2,
        '<UNK>': 3,
        '<SEP>': 4,  # Separator between operations
    }

    def __init__(self):
        """Initialize vocabulary."""
        self.token_to_id = {}
        self.id_to_token = {}
        self._build_vocabulary()

    def _build_vocabulary(self):
        """Build the vocabulary mapping."""
        idx = len(self.SPECIAL_TOKENS)

        # Add special tokens
        for token, token_id in self.SPECIAL_TOKENS.items():
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

        # Add operations
        for op in self.OPERATIONS:
            self.token_to_id[f'OP:{op}'] = idx
            self.id_to_token[idx] = f'OP:{op}'
            idx += 1

        # Add sketches
        for sketch in self.SKETCHES:
            self.token_to_id[f'SKETCH:{sketch}'] = idx
            self.id_to_token[idx] = f'SKETCH:{sketch}'
            idx += 1

        # Add planes
        for plane in self.PLANES:
            self.token_to_id[f'PLANE:{plane}'] = idx
            self.id_to_token[idx] = f'PLANE:{plane}'
            idx += 1

        # Add constraints
        for constraint in self.CONSTRAINTS:
            self.token_to_id[f'CONSTRAINT:{constraint}'] = idx
            self.id_to_token[idx] = f'CONSTRAINT:{constraint}'
            idx += 1

        # Add parameter tokens (quantized values)
        # Following DeepCAD's 8-bit quantization for parameters
        for i in range(256):
            param_token = f'PARAM:{i}'
            self.token_to_id[param_token] = idx
            self.id_to_token[idx] = param_token
            idx += 1

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.token_to_id)

    def encode_token(self, token: str) -> int:
        """Encode a single token to ID."""
        return self.token_to_id.get(token, self.SPECIAL_TOKENS['<UNK>'])

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to string."""
        return self.id_to_token.get(token_id, '<UNK>')


class CADTokenizer:
    """
    Tokenizer for CAD operations following modern architectures.

    Converts between:
    - CAD operations (structured) ↔ Token sequences (numeric)
    - Token sequences ↔ Text representation

    Based on Text2CAD, CADFusion, and GenCAD approaches.
    """

    def __init__(self, quantization_bits: int = 8):
        """
        Initialize tokenizer.

        Args:
            quantization_bits: Bits for parameter quantization (default 8)
        """
        self.vocabulary = CADVocabulary()
        self.quantization_bits = quantization_bits
        self.quantization_levels = 2 ** quantization_bits

    def quantize_parameter(self, value: float, min_val: float = 0.0, max_val: float = 100.0) -> int:
        """
        Quantize a continuous parameter to discrete levels.

        Args:
            value: Parameter value to quantize
            min_val: Minimum parameter value
            max_val: Maximum parameter value

        Returns:
            Quantized value (0 to quantization_levels-1)
        """
        # Clip to range
        value = np.clip(value, min_val, max_val)

        # Normalize to [0, 1]
        normalized = (value - min_val) / (max_val - min_val)

        # Quantize
        quantized = int(normalized * (self.quantization_levels - 1))

        return quantized

    def dequantize_parameter(self, quantized: int, min_val: float = 0.0, max_val: float = 100.0) -> float:
        """
        Dequantize a discrete level to continuous parameter.

        Args:
            quantized: Quantized value
            min_val: Minimum parameter value
            max_val: Maximum parameter value

        Returns:
            Continuous parameter value
        """
        # Normalize to [0, 1]
        normalized = quantized / (self.quantization_levels - 1)

        # Scale to range
        value = min_val + normalized * (max_val - min_val)

        return value

    def encode_operation(self, operation: CADOperation) -> List[int]:
        """
        Encode a single CAD operation to token sequence.

        Args:
            operation: CAD operation to encode

        Returns:
            List of token IDs
        """
        tokens = []

        # Add operation type
        tokens.append(self.vocabulary.encode_token(f'OP:{operation.operation_type}'))

        # Add sketch type if present
        if operation.sketch_type:
            tokens.append(self.vocabulary.encode_token(f'SKETCH:{operation.sketch_type}'))

        # Add plane if present
        if operation.plane:
            tokens.append(self.vocabulary.encode_token(f'PLANE:{operation.plane}'))

        # Add parameters (quantized)
        for param_name, param_value in operation.parameters.items():
            quantized = self.quantize_parameter(param_value)
            tokens.append(self.vocabulary.encode_token(f'PARAM:{quantized}'))

        # Add constraints
        for constraint in operation.constraints:
            tokens.append(self.vocabulary.encode_token(f'CONSTRAINT:{constraint}'))

        # Add separator
        tokens.append(self.vocabulary.SPECIAL_TOKENS['<SEP>'])

        return tokens

    def decode_operation(self, tokens: List[int]) -> CADOperation:
        """
        Decode token sequence to CAD operation.

        Args:
            tokens: List of token IDs

        Returns:
            CAD operation
        """
        operation_type = None
        sketch_type = None
        plane = None
        parameters = {}
        constraints = []

        param_count = 0

        for token_id in tokens:
            token_str = self.vocabulary.decode_token(token_id)

            if token_str.startswith('OP:'):
                operation_type = token_str.replace('OP:', '')
            elif token_str.startswith('SKETCH:'):
                sketch_type = token_str.replace('SKETCH:', '')
            elif token_str.startswith('PLANE:'):
                plane = token_str.replace('PLANE:', '')
            elif token_str.startswith('PARAM:'):
                quantized = int(token_str.replace('PARAM:', ''))
                value = self.dequantize_parameter(quantized)
                parameters[f'param_{param_count}'] = value
                param_count += 1
            elif token_str.startswith('CONSTRAINT:'):
                constraints.append(token_str.replace('CONSTRAINT:', ''))
            elif token_str == '<SEP>':
                break

        return CADOperation(
            operation_type=operation_type or 'sketch',
            sketch_type=sketch_type,
            parameters=parameters,
            constraints=constraints,
            plane=plane
        )

    def encode(self, operations: List[CADOperation]) -> List[int]:
        """
        Encode a sequence of CAD operations.

        Args:
            operations: List of CAD operations

        Returns:
            Token sequence
        """
        tokens = [self.vocabulary.SPECIAL_TOKENS['<START>']]

        for operation in operations:
            tokens.extend(self.encode_operation(operation))

        tokens.append(self.vocabulary.SPECIAL_TOKENS['<END>'])

        return tokens

    def decode(self, tokens: List[int]) -> List[CADOperation]:
        """
        Decode token sequence to CAD operations.

        Args:
            tokens: Token sequence

        Returns:
            List of CAD operations
        """
        operations = []
        current_tokens = []

        for token_id in tokens:
            if token_id == self.vocabulary.SPECIAL_TOKENS['<START>']:
                continue
            elif token_id == self.vocabulary.SPECIAL_TOKENS['<END>']:
                break
            elif token_id == self.vocabulary.SPECIAL_TOKENS['<SEP>']:
                if current_tokens:
                    operations.append(self.decode_operation(current_tokens))
                    current_tokens = []
            else:
                current_tokens.append(token_id)

        # Handle last operation if no separator at end
        if current_tokens:
            operations.append(self.decode_operation(current_tokens))

        return operations

    def to_text(self, operations: List[CADOperation]) -> str:
        """
        Convert CAD operations to human-readable text.

        Args:
            operations: List of CAD operations

        Returns:
            Text representation
        """
        lines = []

        for i, op in enumerate(operations, 1):
            line = f"{i}. {op.operation_type}"

            if op.sketch_type:
                line += f" ({op.sketch_type})"

            if op.plane:
                line += f" on {op.plane}"

            if op.parameters:
                params = ", ".join(f"{k}={v:.2f}" for k, v in op.parameters.items())
                line += f" [{params}]"

            if op.constraints:
                line += f" with constraints: {', '.join(op.constraints)}"

            lines.append(line)

        return "\n".join(lines)

    def from_text(self, text: str) -> List[CADOperation]:
        """
        Parse human-readable text to CAD operations.

        Args:
            text: Text representation

        Returns:
            List of CAD operations

        Note: This is a simplified parser for demonstration
        """
        operations = []
        # This would need a more sophisticated parser in production
        # For now, return empty list
        return operations

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocabulary.vocab_size

    def save_vocabulary(self, path: str):
        """Save vocabulary to file."""
        vocab_data = {
            'token_to_id': self.vocabulary.token_to_id,
            'id_to_token': {int(k): v for k, v in self.vocabulary.id_to_token.items()},
            'quantization_bits': self.quantization_bits
        }
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)

    def load_vocabulary(self, path: str):
        """Load vocabulary from file."""
        with open(path, 'r') as f:
            vocab_data = json.load(f)

        self.vocabulary.token_to_id = vocab_data['token_to_id']
        self.vocabulary.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
        self.quantization_bits = vocab_data['quantization_bits']


def example_usage():
    """Example usage of CAD tokenizer."""
    # Create tokenizer
    tokenizer = CADTokenizer()

    # Create some CAD operations
    operations = [
        CADOperation(
            operation_type='sketch',
            sketch_type='rectangle',
            plane='XY',
            parameters={'width': 10.0, 'height': 5.0},
            constraints=['centered', 'horizontal']
        ),
        CADOperation(
            operation_type='extrude',
            parameters={'depth': 3.0, 'draft_angle': 5.0},
            constraints=['perpendicular']
        ),
        CADOperation(
            operation_type='fillet',
            parameters={'radius': 1.0},
            constraints=['all_edges']
        )
    ]

    # Encode to tokens
    tokens = tokenizer.encode(operations)
    print(f"Encoded to {len(tokens)} tokens")
    print(f"Token sequence: {tokens}")

    # Decode back
    decoded_ops = tokenizer.decode(tokens)
    print(f"\nDecoded {len(decoded_ops)} operations:")

    # Convert to text
    text_repr = tokenizer.to_text(decoded_ops)
    print(f"\nText representation:\n{text_repr}")

    print(f"\nVocabulary size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    example_usage()
