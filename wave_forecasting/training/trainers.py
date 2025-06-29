# training/trainers.py
"""Training loops for different model types"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any
import os

class BaseTrainer:
    """Base trainer class"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.lr_decay_epochs,
            gamma=config.lr_decay_factor
        )
        self.criterion = WavePhysicsLoss(config)
    
    def save_checkpoint(self, epoch: int, loss: float, path: str):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

class SpatialTrainer(BaseTrainer):
    """Trainer for spatial wave prediction models"""
    
    def __init__(self, model: SpatialWaveGNN, config: TrainingConfig, 
                 edge_index: torch.Tensor, edge_attr: torch.Tensor):
        super().__init__(model, config)
        self.edge_index = edge_index
        self.edge_attr = edge_attr
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total': [], 'mse': [], 'physics': []}
        
        for batch_idx, batch in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            features = batch['features']  # [batch_size, num_nodes, num_features]
            targets = batch['targets']    # [batch_size, num_nodes, 3]
            
            # Process each sample in batch separately (GNN requirement)
            batch_predictions = []
            batch_size = features.shape[0]
            
            for i in range(batch_size):
                sample_features = features[i]  # [num_nodes, num_features]
                pred = self.model(sample_features, self.edge_index, self.edge_attr)
                batch_predictions.append(pred)
            
            predictions = torch.stack(batch_predictions)  # [batch_size, num_nodes, 3]
            
            # Compute loss
            loss_dict = self.criterion(predictions, targets)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
            
            # Track losses
            for key in epoch_losses:
                epoch_losses[key].append(loss_dict[f'{key}_loss'].item())
        
        return {key: np.mean(values) for key, values in epoch_losses.items()}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None,
              checkpoint_dir: str = "checkpoints") -> Dict[str, Any]:
        """Full training loop"""
        
        print(f"🚀 Starting spatial training: {self.config.num_epochs} epochs")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        train_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.config.num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log progress
            print(f"Epoch {epoch+1:2d}/{self.config.num_epochs}: "
                  f"Train Loss={train_metrics['total']:.4f}, "
                  f"MSE={train_metrics['mse']:.4f}, "
                  f"Physics={train_metrics['physics']:.4f}")
            
            if val_metrics:
                print(f"   Val Loss={val_metrics['total']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"{checkpoint_dir}/spatial_epoch_{epoch+1}.pt"
                self.save_checkpoint(epoch, train_metrics['total'], checkpoint_path)
            
            # Track history
            train_history['train_loss'].append(train_metrics['total'])
            if val_metrics:
                train_history['val_loss'].append(val_metrics['total'])
        
        print("✅ Spatial training complete!")
        return train_history
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        val_losses = {'total': [], 'mse': [], 'physics': []}
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features']
                targets = batch['targets']
                
                # Process batch
                batch_predictions = []
                for i in range(features.shape[0]):
                    pred = self.model(features[i], self.edge_index, self.edge_attr)
                    batch_predictions.append(pred)
                
                predictions = torch.stack(batch_predictions)
                loss_dict = self.criterion(predictions, targets)
                
                for key in val_losses:
                    val_losses[key].append(loss_dict[f'{key}_loss'].item())
        
        return {key: np.mean(values) for key, values in val_losses.items()}

class TemporalTrainer(BaseTrainer):
    """Trainer for temporal wave prediction models"""
    
    def __init__(self, model: TemporalWaveGNN, config: TrainingConfig,
                 edge_index: torch.Tensor, edge_attr: torch.Tensor):
        super().__init__(model, config)
        self.edge_index = edge_index
        self.edge_attr = edge_attr
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total': [], 'mse': [], 'physics': []}
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            input_sequences = batch['input_sequences']  # [batch_size, seq_len, nodes, features]
            targets = batch['targets']                  # [batch_size, nodes, 3]
            
            # Process each sample separately
            batch_predictions = []
            for i in range(input_sequences.shape[0]):
                sample_sequence = input_sequences[i]  # [seq_len, nodes, features]
                pred = self.model(sample_sequence, self.edge_index, self.edge_attr)
                batch_predictions.append(pred)
            
            predictions = torch.stack(batch_predictions)
            
            # Compute loss
            loss_dict = self.criterion(predictions, targets)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
            
            # Track losses
            for key in epoch_losses:
                epoch_losses[key].append(loss_dict[f'{key}_loss'].item())
        
        return {key: np.mean(values) for key, values in epoch_losses.items()}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None,
              checkpoint_dir: str = "checkpoints") -> Dict[str, Any]:
        """Full temporal training loop"""
        
        print(f"🚀 Starting temporal training: {self.config.num_epochs} epochs")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        train_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.config.num_epochs):
            train_metrics = self.train_epoch(train_loader)
            
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate(val_loader)
            
            self.scheduler.step()
            
            print(f"Epoch {epoch+1:2d}/{self.config.num_epochs}: "
                  f"Train Loss={train_metrics['total']:.4f}")
            
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"{checkpoint_dir}/temporal_epoch_{epoch+1}.pt"
                self.save_checkpoint(epoch, train_metrics['total'], checkpoint_path)
            
            train_history['train_loss'].append(train_metrics['total'])
            if val_metrics:
                train_history['val_loss'].append(val_metrics['total'])
        
        print("✅ Temporal training complete!")
        return train_history
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate temporal model"""
        self.model.eval()
        val_losses = {'total': [], 'mse': [], 'physics': []}
        
        with torch.no_grad():
            for batch in dataloader:
                input_sequences = batch['input_sequences']
                targets = batch['targets']
                
                batch_predictions = []
                for i in range(input_sequences.shape[0]):
                    pred = self.model(input_sequences[i], self.edge_index, self.edge_attr)
                    batch_predictions.append(pred)
                
                predictions = torch.stack(batch_predictions)
                loss_dict = self.criterion(predictions, targets)
                
                for key in val_losses:
                    val_losses[key].append(loss_dict[f'{key}_loss'].item())
        
        return {key: np.mean(values) for key, values in val_losses.items()}
