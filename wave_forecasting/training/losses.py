# training/losses.py
"""Loss functions"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.base import ModelConfig, TrainingConfig

class WavePhysicsLoss(nn.Module):
    """Physics-informed loss function for wave prediction"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.mse_weight = config.mse_weight
        self.physics_weight = config.physics_weight
    
    def forward(self, predictions, targets, features=None):
        """
        predictions: [batch, nodes, 3] - [swh, mwd, mwp] predictions
        targets: [batch, nodes, 3] - [swh, mwd, mwp] ground truth
        """
        
        # Basic MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Physics constraints
        physics_loss = 0.0
        
        # 1. Wave height must be positive
        negative_height_penalty = F.relu(-predictions[..., 0]).mean()
        
        # 2. Wave direction constraints (reasonable range)
        extreme_direction_penalty = F.relu(torch.abs(predictions[..., 1]) - 5.0).mean()
        
        # 3. Wave period must be positive and reasonable
        period_penalty = F.relu(-predictions[..., 2]).mean()
        
        physics_loss = (negative_height_penalty + 
                       extreme_direction_penalty + 
                       period_penalty)
        
        total_loss = (self.mse_weight * mse_loss + 
                     self.physics_weight * physics_loss)
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'physics_loss': physics_loss
        }
