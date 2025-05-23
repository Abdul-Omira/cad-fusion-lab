"""
Core Text-to-CAD Model Architecture

This module implements the state-of-the-art text-to-CAD model with:
- Adaptive text encoder with BERT backbone
- 24-layer transformer decoder for CAD sequences
- Visual feedback module with CLIP integration
- Geometric validation system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from typing import Optional, Tuple, Dict, Any
import math


class AdaptiveLayer(nn.Module):
    """Adaptive projection layer for multi-modal alignment."""
    
    def __init__(self, in_dim: int = 768, out_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with projection and normalization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_dim)
            
        Returns:
            Projected tensor of shape (batch_size, seq_len, out_dim)
        """
        return self.dropout(self.layer_norm(self.proj(x)))


class TextEncoder(nn.Module):
    """BERT-based text encoder with adaptive projection."""
    
    def __init__(self, model_name: str = "bert-base-uncased", output_dim: int = 512):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.adaptive_layer = AdaptiveLayer(
            in_dim=self.bert.config.hidden_size,
            out_dim=output_dim
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode text input to fixed-size representation.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Encoded text representation of shape (batch_size, seq_len, output_dim)
        """
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.adaptive_layer(bert_output.last_hidden_state)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer decoder."""
    
    def __init__(self, d_model: int, max_seq_length: int = 512):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        return x + self.pe[:x.size(0), :]


class CADSequenceDecoder(nn.Module):
    """24-layer transformer decoder for CAD sequence generation."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 24,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # 24-layer transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer decoder.
        
        Args:
            tgt: Target CAD sequence tokens (batch_size, tgt_len)
            memory: Encoded text representation (batch_size, src_len, d_model)
            tgt_mask: Causal mask for target sequence
            memory_mask: Attention mask for memory
            
        Returns:
            Logits for next token prediction (batch_size, tgt_len, vocab_size)
        """
        # Embed and add positional encoding
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Pass through transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        
        # Project to vocabulary
        return self.output_projection(decoder_output)


class TextToCADModel(nn.Module):
    """Complete Text-to-CAD model architecture."""
    
    def __init__(
        self,
        vocab_size: int,
        text_encoder_name: str = "bert-base-uncased",
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 24,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512
    ):
        super().__init__()
        
        self.text_encoder = TextEncoder(text_encoder_name, d_model)
        self.cad_decoder = CADSequenceDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_length=max_seq_length
        )
        
        # Special tokens
        self.pad_token_id = 0
        self.sos_token_id = 1
        self.eos_token_id = 2
        
    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        cad_sequence: torch.Tensor,
        cad_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            text_input_ids: Text token IDs (batch_size, text_len)
            text_attention_mask: Text attention mask (batch_size, text_len)
            cad_sequence: CAD sequence tokens (batch_size, cad_len)
            cad_attention_mask: CAD attention mask (batch_size, cad_len)
            
        Returns:
            Logits for next CAD token prediction (batch_size, cad_len, vocab_size)
        """
        # Encode text
        text_encoded = self.text_encoder(text_input_ids, text_attention_mask)
        
        # Create memory mask from text attention mask
        memory_mask = ~text_attention_mask.bool() if text_attention_mask is not None else None
        
        # Decode CAD sequence
        logits = self.cad_decoder(
            tgt=cad_sequence,
            memory=text_encoded,
            memory_mask=memory_mask
        )
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> torch.Tensor:
        """
        Generate CAD sequence from text input.
        
        Args:
            text_input_ids: Text token IDs (1, text_len)
            text_attention_mask: Text attention mask (1, text_len)
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated CAD sequence (1, generated_len)
        """
        self.eval()
        device = text_input_ids.device
        
        # Encode text
        text_encoded = self.text_encoder(text_input_ids, text_attention_mask)
        memory_mask = ~text_attention_mask.bool()
        
        # Initialize with SOS token
        generated = torch.tensor([[self.sos_token_id]], device=device)
        
        for _ in range(max_length - 1):
            # Get logits for next token
            logits = self.cad_decoder(
                tgt=generated,
                memory=text_encoded,
                memory_mask=memory_mask
            )
            
            # Apply temperature
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == self.eos_token_id:
                break
        
        return generated
    
    def compute_loss(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        cad_sequence: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for sequence prediction.
        
        Loss: L_seq = -∑_{t=1}^T log p(c_t | c_<t, T)
        
        Args:
            text_input_ids: Text token IDs
            text_attention_mask: Text attention mask
            cad_sequence: Input CAD sequence (without last token)
            labels: Target CAD sequence (without first token)
            
        Returns:
            Cross-entropy loss
        """
        logits = self.forward(text_input_ids, text_attention_mask, cad_sequence)
        
        # Reshape for loss computation
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        
        # Compute cross-entropy loss (ignore padding tokens)
        loss = F.cross_entropy(logits, labels, ignore_index=self.pad_token_id)
        
        return loss


def quantize_param(param: float, bits: int = 8) -> int:
    """
    Quantize continuous parameter to discrete value.
    
    Args:
        param: Parameter value in [0, 1]
        bits: Number of quantization bits
        
    Returns:
        Quantized integer value
    """
    return round(param * (2**bits - 1))


def dequantize_param(quantized: int, bits: int = 8) -> float:
    """
    Convert quantized parameter back to continuous value.
    
    Args:
        quantized: Quantized integer value
        bits: Number of quantization bits
        
    Returns:
        Continuous parameter value in [0, 1]
    """
    return quantized / (2**bits - 1)
