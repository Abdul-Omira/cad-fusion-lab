"""
Dataset class for Text-to-CAD model training.
"""

import json
import torch
from pathlib import Path
from typing import Dict, Tuple, Any
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class TextToCADDataset(Dataset):
    """Dataset class for Text-to-CAD model training."""
    
    def __init__(
        self,
        data_dir: Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the data
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length for text
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Load data samples
        self.samples = self._load_samples()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata."""
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def _load_samples(self) -> list:
        """Load data samples."""
        samples = []
        for sample_file in self.data_dir.glob("*.json"):
            with open(sample_file, 'r') as f:
                sample = json.load(f)
                samples.append(sample)
        return samples
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Dictionary containing:
                - input_ids: Tokenized text input
                - attention_mask: Attention mask for the input
                - cad_features: CAD model features
        """
        sample = self.samples[idx]
        if "description" not in sample:
            raise KeyError(f"Sample at idx {idx} (id: {sample.get('id', 'unknown')}) is missing the 'description' field.")
        # Process text
        text = sample["description"]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # Process CAD features
        cad_features = torch.tensor(sample["cad_features"], dtype=torch.float32)
        # If CAD features are supposed to be token indices, uncomment the following line:
        # cad_features = torch.tensor(sample["cad_features"], dtype=torch.long)
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "cad_features": cad_features
        } 