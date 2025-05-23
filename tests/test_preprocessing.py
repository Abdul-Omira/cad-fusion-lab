"""
Tests for data preprocessing pipeline
"""

import pytest
pytest.importorskip("torch")
pytest.importorskip("numpy")
import torch
import numpy as np
import json
import os
import tempfile
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessing import (
    CADSequence,
    TextAnnotation,
    GeometricHasher,
    CADTokenizer,
    DeepCADProcessor,
    TextToCADDataset
)


class TestGeometricHasher:
    """Test the geometric hash-based deduplication."""
    
    @pytest.fixture
    def hasher(self):
        """Create a geometric hasher."""
        return GeometricHasher(precision=6)
    
    def test_compute_hash(self, hasher):
        """Test hash computation."""
        seq1 = [1, 2, 3, 4, 5]
        seq2 = [1, 2, 3, 4, 5]  # Identical
        seq3 = [5, 4, 3, 2, 1]  # Same elements, different order
        seq4 = [1, 2, 3, 4, 6]  # Different
        
        hash1 = hasher.compute_hash(seq1)
        hash2 = hasher.compute_hash(seq2)
        hash3 = hasher.compute_hash(seq3)
        hash4 = hasher.compute_hash(seq4)
        
        # Check hash properties
        assert isinstance(hash1, str)
        assert len(hash1) == hasher.precision
        
        # Check equality
        assert hash1 == hash2  # Identical sequences should have same hash
        assert hash1 == hash3  # Normalized sequences should have same hash
        assert hash1 != hash4  # Different sequences should have different hash


class TestCADTokenizer:
    """Test CAD sequence tokenization."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a CAD tokenizer."""
        return CADTokenizer(vocab_size=10000)
    
    def test_tokenizer_init(self, tokenizer):
        """Test tokenizer initialization."""
        assert tokenizer.vocab_size == 10000
        assert tokenizer.pad_token == 0
        assert tokenizer.sos_token == 1
        assert tokenizer.eos_token == 2
        assert tokenizer.unk_token == 3
        
    def test_encode_sequence(self, tokenizer):
        """Test encoding CAD sequence to tokens."""
        cad_sequence = CADSequence(
            operations=[10, 20, 30],
            parameters=[0.1, 0.5, 0.9],
            quantized_params=[25, 127, 230],
            geometric_hash="abcdef",
            complexity_level=1
        )
        
        tokens = tokenizer.encode_sequence(cad_sequence)
        
        # Check token structure
        assert tokens[0] == tokenizer.sos_token  # Start with SOS
        assert tokens[-1] == tokenizer.eos_token  # End with EOS
        
        # Check operations and parameters are interleaved
        assert len(tokens) == len(cad_sequence.operations) * 2 + 2  # ops*2 + SOS + EOS
        
    def test_decode_sequence(self, tokenizer):
        """Test decoding token sequence back to CAD sequence."""
        original_sequence = CADSequence(
            operations=[10, 20, 30],
            parameters=[0.1, 0.5, 0.9],
            quantized_params=[25, 127, 230],
            geometric_hash="abcdef",
            complexity_level=1
        )
        
        tokens = tokenizer.encode_sequence(original_sequence)
        decoded = tokenizer.decode_sequence(tokens)
        
        # Check operations are preserved
        assert decoded.operations == original_sequence.operations
        assert decoded.quantized_params == original_sequence.quantized_params


class TestDeepCADProcessor:
    """Test the DeepCAD dataset processor."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
            yield input_dir, output_dir
    
    def test_process_dataset(self, temp_dirs):
        """Test dataset processing pipeline."""
        input_dir, output_dir = temp_dirs
        
        # Create a sample dataset structure
        processor = DeepCADProcessor(input_dir, output_dir)
        
        # Process dataset (with mocked data)
        stats = processor.process_dataset()
        
        # Check statistics
        assert isinstance(stats, dict)
        assert "total_sequences" in stats
        assert "unique_sequences" in stats
        assert "duplicates_removed" in stats
        
        # Check output files
        assert os.path.exists(os.path.join(output_dir, "train.json"))
        assert os.path.exists(os.path.join(output_dir, "val.json"))
        assert os.path.exists(os.path.join(output_dir, "test.json"))
        
        # Check duplicate removal
        assert stats["duplicates_removed"] >= 0
        assert stats["total_sequences"] >= stats["unique_sequences"]
    
    def test_quantize_param(self):
        """Test parameter quantization."""
        # 8-bit quantization tests
        assert DeepCADProcessor.quantize_param(0.0, bits=8) == 0
        assert DeepCADProcessor.quantize_param(1.0, bits=8) == 255
        assert DeepCADProcessor.quantize_param(0.5, bits=8) == 127
        
        # 12-bit quantization tests
        assert DeepCADProcessor.quantize_param(0.0, bits=12) == 0
        assert DeepCADProcessor.quantize_param(1.0, bits=12) == 4095
        assert DeepCADProcessor.quantize_param(0.5, bits=12) == 2047


class TestTextToCADDataset:
    """Test the PyTorch dataset for text-to-CAD training."""
    
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample data files for testing."""
        # Create CAD data file
        cad_data = [
            {
                "operations": [10, 20],
                "parameters": [0.1, 0.5],
                "quantized_params": [25, 127],
                "geometric_hash": "abcdef",
                "complexity_level": 1,
                "tokens": [1, 10, 1025, 20, 1127, 2]  # SOS, op, param, op, param, EOS
            }
        ]
        
        # Create text data file
        text_data = [
            {
                "basic_description": "A simple box",
                "detailed_description": "A rectangular box with dimensions 10x5x2",
                "complexity_level": 1,
                "tokens": [101, 102, 103, 104]
            }
        ]
        
        # Write to temp files
        cad_file = tmp_path / "cad_data.json"
        text_file = tmp_path / "text_data.json"
        
        with open(cad_file, "w") as f:
            json.dump(cad_data, f)
            
        with open(text_file, "w") as f:
            json.dump(text_data, f)
            
        return str(cad_file), str(text_file)
    
    def test_dataset_init(self, sample_data):
        """Test dataset initialization."""
        cad_file, text_file = sample_data
        
        # Initialize dataset
        dataset = TextToCADDataset(
            cad_data_path=cad_file,
            annotation_data_path=text_file,
            max_text_length=64,
            max_cad_length=64
        )
        
        # Check dataset properties
        assert len(dataset) == 1  # One sample in our test data
        assert dataset.max_text_length == 64
        assert dataset.max_cad_length == 64
        
    def test_getitem(self, sample_data):
        """Test retrieving an item from the dataset."""
        cad_file, text_file = sample_data
        
        # Initialize dataset
        dataset = TextToCADDataset(
            cad_data_path=cad_file,
            annotation_data_path=text_file,
            max_text_length=64,
            max_cad_length=64
        )
        
        # Get a sample
        sample = dataset[0]
        
        # Check sample structure
        assert "text_input_ids" in sample
        assert "text_attention_mask" in sample
        assert "cad_input" in sample
        assert "cad_target" in sample
        assert "complexity_level" in sample
        
        # Check tensor properties
        assert isinstance(sample["text_input_ids"], torch.Tensor)
        assert isinstance(sample["cad_input"], torch.Tensor)
        assert sample["cad_input"].dtype == torch.long
        assert sample["complexity_level"].item() == 1