import torch.nn.functional as F
import torch.nn as nn
import torch

class WavePhysicsLoss(nn.Module):
    """Loss function with basic wave physics constraints"""
    
    def __init__(self, mse_weight=1.0, physics_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.physics_weight = physics_weight
    
    def forward(self, predictions, targets, features=None):
        """
        predictions: [batch, nodes, 3] - [swh, mwd, mwp] predictions
        targets: [batch, nodes, 3] - [swh, mwd, mwp] ground truth
        features: [batch, nodes, all_features] - input features (optional)
        """
        
        # Basic MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Physics constraints
        physics_loss = 0.0
        
        # 1. Wave height should be positive
        negative_height_penalty = F.relu(-predictions[:, :, 0]).mean()
        
        # 2. Wave direction should be in reasonable range (normalize to 0-2π)
        # For now, just penalize extreme values
        extreme_direction_penalty = F.relu(torch.abs(predictions[:, :, 1]) - 5.0).mean()
        
        # 3. Wave period should be positive and reasonable (2-20 seconds typical)
        period_penalty = F.relu(-predictions[:, :, 2]).mean()
        
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

