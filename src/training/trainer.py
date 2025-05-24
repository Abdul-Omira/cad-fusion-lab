"""
Training Pipeline for Text-to-CAD Model

Implements multi-stage training:
- Sequential pre-training on CAD sequences
- Visual feedback fine-tuning with PPO
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
import wandb
import logging
from typing import Dict, Any, Optional

from src.models.text_to_cad import TextToCADModel
from src.models.visual_feedback import VisualFeedbackModule
from src.data.preprocessing import TextToCADDataset


class Trainer:
    """Handles model training and fine-tuning."""
    
    def __init__(
        self,
        model: TextToCADModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        visual_feedback_module: Optional[VisualFeedbackModule] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.visual_feedback_module = visual_feedback_module
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        num_training_steps = len(train_loader) * config.get('num_epochs', 10)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.get('warmup_steps', 0),
            num_training_steps=num_training_steps
        )
        
        # Logging
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'text-to-cad'),
                config=config
            )
            wandb.watch(self.model)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, epoch: int) -> float:
        """Run one epoch of training."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            loss = self.model.compute_loss(
                text_input_ids=batch['input_ids'],
                text_attention_mask=batch['attention_mask'],
                cad_sequence=batch['cad_features'],
                labels=batch['cad_features']
            )
            
            # Backward pass and optimization
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Log batch metrics
            if batch_idx % self.config.get('log_interval', 100) == 0:
                self.logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}"
                )
                if self.config.get('use_wandb', False):
                    wandb.log({'train_loss_batch': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def evaluate_epoch(self) -> float:
        """Run one epoch of evaluation."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                loss = self.model.compute_loss(
                    text_input_ids=batch['text_input_ids'],
                    text_attention_mask=batch['text_attention_mask'],
                    cad_sequence=batch['cad_features'],
                    labels=batch['cad_features']
                )
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        """Run full training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.get('num_epochs', 10)):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate_epoch()
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.get('num_epochs', 10)}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss_epoch': train_loss,
                    'val_loss_epoch': val_loss
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_interval', 1) == 0:
                self._save_checkpoint(epoch)
        
        self.logger.info("Training finished.")
        if self.config.get('use_wandb', False):
            wandb.finish()

    def finetune_with_ppo(self):
        """Fine-tune model using PPO with visual feedback."""
        if not self.visual_feedback_module:
            self.logger.error("Visual feedback module not provided for PPO fine-tuning.")
            return
            
        self.logger.info("Starting PPO fine-tuning with visual feedback...")
        
        # PPO-specific components (simplified)
        # Actor: self.model
        # Critic: Separate value network (or shared with actor)
        # Reward: From self.visual_feedback_module
        
        # Example PPO loop (highly simplified)
        for epoch in range(self.config.get('ppo_epochs', 5)):
            for batch in self.train_loader:
                # Generate CAD sequences (rollouts)
                # Compute rewards using visual feedback module
                # Update actor and critic networks
                pass # Placeholder for PPO implementation
            
            self.logger.info(f"PPO Epoch {epoch+1} completed.")
            # Log PPO metrics
        
        self.logger.info("PPO fine-tuning finished.")

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")


# Example usage (placeholder)
if __name__ == "__main__":
    # Dummy config and data loaders for demonstration
    config = {
        'learning_rate': 1e-4,
        'num_epochs': 1,
        'use_wandb': False,
        'vocab_size': 10000 # Example vocab size
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dummy model
    model = TextToCADModel(vocab_size=config['vocab_size'])
    
    # Dummy data loaders
    # Replace with actual data loading from src.data.preprocessing
    train_loader = DataLoader([]) # Placeholder
    val_loader = DataLoader([])   # Placeholder
    
    trainer = Trainer(model, train_loader, val_loader, config, device)
    # trainer.train()