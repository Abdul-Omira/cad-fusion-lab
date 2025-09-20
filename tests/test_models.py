"""
Tests for the Text-to-CAD model components.
"""

import pytest
import torch
import numpy as np
from src.models.text_to_cad import (
    TextToCADModel,
    TextEncoder,
    CADSequenceDecoder,
    AdaptiveLayer,
    ModelError,
    InputError,
    GenerationError
)

@pytest.fixture
def model():
    """Create a test model instance."""
    return TextToCADModel(
        vocab_size=1000,
        text_encoder_name="bert-base-uncased",
        d_model=128,  # Smaller for testing
        nhead=4,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_length=64,
        offline_mode=True  # Use offline mode for tests
    )

@pytest.fixture
def sample_batch():
    """Create a sample batch of data."""
    batch_size = 2
    text_len = 10
    cad_len = 8
    
    return {
        "text_input_ids": torch.randint(0, 1000, (batch_size, text_len)),
        "text_attention_mask": torch.ones(batch_size, text_len),
        "cad_sequence": torch.randint(0, 1000, (batch_size, cad_len)),
        "labels": torch.randint(0, 1000, (batch_size, cad_len))
    }

def test_model_initialization(model):
    """Test model initialization and architecture."""
    assert isinstance(model, TextToCADModel)
    assert isinstance(model.text_encoder, TextEncoder)
    assert isinstance(model.cad_decoder, CADSequenceDecoder)
    
    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "Model should have parameters"
    
    # Check special tokens
    assert model.pad_token_id == 0
    assert model.sos_token_id == 1
    assert model.eos_token_id == 2

def test_forward_pass(model, sample_batch):
    """Test model forward pass."""
    # Forward pass
    logits = model(
        text_input_ids=sample_batch["text_input_ids"],
        text_attention_mask=sample_batch["text_attention_mask"],
        cad_sequence=sample_batch["cad_sequence"]
    )
    
    # Check output shape
    batch_size, seq_len, vocab_size = logits.shape
    assert batch_size == sample_batch["text_input_ids"].size(0)
    assert seq_len == sample_batch["cad_sequence"].size(1)
    assert vocab_size == 1000  # vocab_size from model initialization

def test_loss_computation(model, sample_batch):
    """Test loss computation."""
    # Compute loss
    loss, metrics = model.compute_loss(
        text_input_ids=sample_batch["text_input_ids"],
        text_attention_mask=sample_batch["text_attention_mask"],
        cad_sequence=sample_batch["cad_sequence"],
        labels=sample_batch["labels"]
    )
    
    # Check loss
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    # Check metrics
    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert "perplexity" in metrics
    assert metrics["loss"] > 0
    assert metrics["perplexity"] > 0

def test_generation(model, sample_batch):
    """Test sequence generation."""
    # Generate sequences
    sequences = model.generate(
        text_input_ids=sample_batch["text_input_ids"],
        text_attention_mask=sample_batch["text_attention_mask"],
        max_length=16,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=2
    )
    
    # Check output
    assert isinstance(sequences, list)
    assert len(sequences) == sample_batch["text_input_ids"].size(0) * 2  # num_return_sequences=2
    assert all(isinstance(seq, list) for seq in sequences)
    assert all(len(seq) <= 16 for seq in sequences)  # max_length=16

def test_input_validation(model):
    """Test input validation."""
    # Test invalid input shapes
    with pytest.raises(InputError):
        model.forward(
            text_input_ids=torch.randn(2, 3, 4),  # Invalid shape
            text_attention_mask=torch.ones(2, 3),
            cad_sequence=torch.ones(2, 4)
        )
    
    # Test mismatched batch sizes
    with pytest.raises(InputError):
        model.forward(
            text_input_ids=torch.ones(2, 3),
            text_attention_mask=torch.ones(2, 3),
            cad_sequence=torch.ones(3, 4)  # Different batch size
        )

def test_error_handling(model, sample_batch):
    """Test error handling."""
    # Test model error
    with pytest.raises(ModelError):
        # Simulate an error by passing invalid data
        model.forward(
            text_input_ids=torch.tensor([[999999]]),  # Invalid token ID
            text_attention_mask=torch.ones(1, 1),
            cad_sequence=torch.ones(1, 1)
        )
    
    # Test generation error
    with pytest.raises(GenerationError):
        model.generate(
            text_input_ids=torch.tensor([[999999]]),  # Invalid token ID
            text_attention_mask=torch.ones(1, 1)
        )

def test_model_save_load(model, tmp_path):
    """Test model saving and loading."""
    # Save model
    save_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), save_path)
    
    # Create new model
    new_model = TextToCADModel(
        vocab_size=1000,
        text_encoder_name="bert-base-uncased",
        d_model=128,
        nhead=4,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_length=64
    )
    
    # Load state dict
    new_model.load_state_dict(torch.load(save_path))
    
    # Compare parameters
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)

def test_adaptive_layer():
    """Test adaptive layer functionality."""
    layer = AdaptiveLayer(in_dim=768, out_dim=512)
    x = torch.randn(2, 10, 768)
    
    # Forward pass
    output = layer(x)
    
    # Check output
    assert output.shape == (2, 10, 512)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_text_encoder():
    """Test text encoder functionality."""
    encoder = TextEncoder(model_name="bert-base-uncased", output_dim=512, offline_mode=True)
    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones(2, 10)
    
    # Forward pass
    output = encoder(input_ids, attention_mask)
    
    # Check output
    assert output.shape == (2, 10, 512)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_cad_decoder():
    """Test CAD sequence decoder functionality."""
    decoder = CADSequenceDecoder(
        vocab_size=1000,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_length=64
    )
    
    # Create input tensors
    tgt = torch.randint(0, 1000, (2, 8))
    memory = torch.randn(2, 10, 128)
    
    # Forward pass
    output = decoder(tgt, memory)
    
    # Check output
    assert output.shape == (2, 8, 1000)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    # Test attention mask
    tgt_mask = decoder.generate_square_subsequent_mask(8)
    assert tgt_mask.shape == (8, 8)
    # Check that diagonal is 0
    assert torch.all(torch.diag(tgt_mask) == 0.0)
    # Check that lower triangle is 0 
    lower_tri = torch.tril(tgt_mask, diagonal=-1)
    assert torch.all(lower_tri == 0.0)
    # Check that upper triangle (above diagonal) is -inf
    for i in range(8):
        for j in range(i + 1, 8):
            assert tgt_mask[i, j] == float('-inf')
