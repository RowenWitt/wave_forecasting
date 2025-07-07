#!/usr/bin/env python3
"""
MWD Circular Loss and Normalization Fixes
Proper handling of wave direction as circular variable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple
import math

class CircularNormalizer:
    """
    Normalizer for circular variables like wave direction (0-360°)
    Converts to unit circle representation: [cos(θ), sin(θ)]
    """
    
    def __init__(self):
        self.fitted = False
        self.mean_cos = 0.0
        self.mean_sin = 0.0
        self.std_cos = 1.0
        self.std_sin = 1.0
    
    def fit(self, angles_deg: np.ndarray):
        """
        Fit normalizer on circular data
        
        Args:
            angles_deg: Wave directions in degrees [0, 360)
        """
        # Convert to radians
        angles_rad = np.deg2rad(angles_deg)
        
        # Convert to unit circle
        cos_vals = np.cos(angles_rad)
        sin_vals = np.sin(angles_rad)
        
        # Compute statistics
        self.mean_cos = np.mean(cos_vals)
        self.mean_sin = np.mean(sin_vals)
        self.std_cos = np.std(cos_vals)
        self.std_sin = np.std(sin_vals)
        
        # Avoid division by zero
        self.std_cos = max(self.std_cos, 1e-6)
        self.std_sin = max(self.std_sin, 1e-6)
        
        self.fitted = True
        
        print(f"CircularNormalizer fitted:")
        print(f"  Mean direction: {np.rad2deg(np.arctan2(self.mean_sin, self.mean_cos)):.1f}°")
        print(f"  Std cos: {self.std_cos:.3f}, sin: {self.std_sin:.3f}")
    
    def transform(self, angles_deg: np.ndarray) -> np.ndarray:
        """
        Transform angles to normalized [cos, sin] representation
        
        Args:
            angles_deg: Wave directions in degrees [N,] or [N, 1]
        
        Returns:
            normalized_circular: [N, 2] normalized [cos, sin] values
        """
        if not self.fitted:
            raise ValueError("CircularNormalizer not fitted")
        
        angles_deg = angles_deg.flatten()
        
        # Convert to radians
        angles_rad = np.deg2rad(angles_deg)
        
        # Convert to unit circle
        cos_vals = np.cos(angles_rad)
        sin_vals = np.sin(angles_rad)
        
        # Normalize
        cos_norm = (cos_vals - self.mean_cos) / self.std_cos
        sin_norm = (sin_vals - self.mean_sin) / self.std_sin
        
        # Stack as [cos, sin] pairs
        return np.column_stack([cos_norm, sin_norm])
    
    def inverse_transform(self, normalized_circular: np.ndarray) -> np.ndarray:
        """
        Transform normalized [cos, sin] back to angles in degrees
        
        Args:
            normalized_circular: [N, 2] normalized [cos, sin] values
        
        Returns:
            angles_deg: Wave directions in degrees [0, 360)
        """
        if not self.fitted:
            raise ValueError("CircularNormalizer not fitted")
        
        # Denormalize
        cos_vals = normalized_circular[:, 0] * self.std_cos + self.mean_cos
        sin_vals = normalized_circular[:, 1] * self.std_sin + self.mean_sin
        
        # Convert back to angles
        angles_rad = np.arctan2(sin_vals, cos_vals)
        angles_deg = np.rad2deg(angles_rad)
        
        # Ensure [0, 360) range
        angles_deg = np.where(angles_deg < 0, angles_deg + 360, angles_deg)
        
        return angles_deg

class VariableSpecificNormalizer:
    """
    Separate normalizers for each wave variable with proper circular handling for MWD
    """
    
    def __init__(self):
        self.swh_scaler = StandardScaler()
        self.mwd_normalizer = CircularNormalizer()
        self.mwp_scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, targets: np.ndarray):
        """
        Fit normalizers on wave data
        
        Args:
            targets: [N, 3] array of [SWH, MWD, MWP] values
        """
        swh = targets[:, 0:1]  # Keep 2D for sklearn
        mwd = targets[:, 1]    # 1D for circular normalizer
        mwp = targets[:, 2:3]  # Keep 2D for sklearn
        
        # Fit each normalizer
        self.swh_scaler.fit(swh)
        self.mwd_normalizer.fit(mwd)
        self.mwp_scaler.fit(mwp)
        
        self.fitted = True
        
        print(f"VariableSpecificNormalizer fitted:")
        print(f"  SWH range: {swh.min():.2f} to {swh.max():.2f} m")
        print(f"  MWD range: {mwd.min():.1f} to {mwd.max():.1f} °")
        print(f"  MWP range: {mwp.min():.2f} to {mwp.max():.2f} s")
    
    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """
        Transform targets with variable-specific normalization
        
        Args:
            targets: [N, 3] array of [SWH, MWD, MWP] values
        
        Returns:
            normalized: [N, 4] array of [SWH_norm, MWD_cos_norm, MWD_sin_norm, MWP_norm]
        """
        if not self.fitted:
            raise ValueError("VariableSpecificNormalizer not fitted")
        
        swh = targets[:, 0:1]
        mwd = targets[:, 1]
        mwp = targets[:, 2:3]
        
        # Normalize each variable
        swh_norm = self.swh_scaler.transform(swh)  # [N, 1]
        mwd_norm = self.mwd_normalizer.transform(mwd)  # [N, 2] cos, sin
        mwp_norm = self.mwp_scaler.transform(mwp)  # [N, 1]
        
        # Concatenate: [SWH, MWD_cos, MWD_sin, MWP]
        return np.concatenate([swh_norm, mwd_norm, mwp_norm], axis=1)
    
    def inverse_transform_targets(self, normalized: np.ndarray) -> np.ndarray:
        """
        Transform normalized targets back to original scale
        
        Args:
            normalized: [N, 4] array of [SWH_norm, MWD_cos_norm, MWD_sin_norm, MWP_norm]
        
        Returns:
            targets: [N, 3] array of [SWH, MWD, MWP] values
        """
        if not self.fitted:
            raise ValueError("VariableSpecificNormalizer not fitted")
        
        # Handle case where input might be [N, 3] instead of [N, 4]
        if normalized.shape[1] == 3:
            # Assume it's already in [SWH, MWD, MWP] format
            return normalized
        
        if normalized.shape[1] != 4:
            raise ValueError(f"Expected 4 features, got {normalized.shape[1]}")
        
        swh_norm = normalized[:, 0:1]    # [N, 1]
        mwd_norm = normalized[:, 1:3]    # [N, 2] cos, sin
        mwp_norm = normalized[:, 3:4]    # [N, 1]
        
        # Debug print
        print(f"Debug shapes: swh_norm={swh_norm.shape}, mwd_norm={mwd_norm.shape}, mwp_norm={mwp_norm.shape}")
        
        # Denormalize each variable
        swh = self.swh_scaler.inverse_transform(swh_norm)       # [N, 1]
        mwd = self.mwd_normalizer.inverse_transform(mwd_norm)   # [N,] (1D)
        mwp = self.mwp_scaler.inverse_transform(mwp_norm)       # [N, 1]
        
        # Ensure consistent shapes and concatenate
        swh_flat = swh.flatten()
        mwp_flat = mwp.flatten()
        
        # Concatenate back to [SWH, MWD, MWP]
        return np.column_stack([swh_flat, mwd, mwp_flat])

class CircularLoss(nn.Module):
    """
    Loss functions with proper circular handling for wave direction
    """
    
    def __init__(self, 
                 mse_weight: float = 1.0,
                 circular_weight: float = 1.0,
                 physics_weight: float = 0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.circular_weight = circular_weight
        self.physics_weight = physics_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss with circular handling for MWD
        
        Args:
            predictions: [batch_size, num_nodes, 4] for [SWH, MWD_cos, MWD_sin, MWP]
            targets: [batch_size, num_nodes, 4] for [SWH, MWD_cos, MWD_sin, MWP]
        """
        
        # Split predictions and targets
        pred_swh = predictions[:, :, 0]      # [batch, nodes]
        pred_mwd_cos = predictions[:, :, 1]  # [batch, nodes]
        pred_mwd_sin = predictions[:, :, 2]  # [batch, nodes]
        pred_mwp = predictions[:, :, 3]      # [batch, nodes]
        
        true_swh = targets[:, :, 0]
        true_mwd_cos = targets[:, :, 1]
        true_mwd_sin = targets[:, :, 2]
        true_mwp = targets[:, :, 3]
        
        # Standard MSE loss for SWH and MWP
        swh_loss = F.mse_loss(pred_swh, true_swh)
        mwp_loss = F.mse_loss(pred_mwp, true_mwp)
        
        # Circular loss for MWD (cos, sin components)
        mwd_cos_loss = F.mse_loss(pred_mwd_cos, true_mwd_cos)
        mwd_sin_loss = F.mse_loss(pred_mwd_sin, true_mwd_sin)
        circular_loss = mwd_cos_loss + mwd_sin_loss
        
        # Alternative: Angular distance loss
        # Convert back to angles for angular distance
        pred_angles = torch.atan2(pred_mwd_sin, pred_mwd_cos)
        true_angles = torch.atan2(true_mwd_sin, true_mwd_cos)
        
        # Compute shortest angular distance
        angle_diff = pred_angles - true_angles
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))  # Wrap to [-π, π]
        angular_loss = torch.mean(angle_diff ** 2)
        
        # Physics constraints
        physics_loss = self._compute_physics_constraints(predictions)
        
        # Total loss
        total_loss = (self.mse_weight * (swh_loss + mwp_loss) +
                     self.circular_weight * (circular_loss + angular_loss) +
                     self.physics_weight * physics_loss)
        
        return {
            'total_loss': total_loss,
            'swh_loss': swh_loss,
            'mwd_circular_loss': circular_loss,
            'mwd_angular_loss': angular_loss,
            'mwp_loss': mwp_loss,
            'physics_loss': physics_loss
        }
    
    def _compute_physics_constraints(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute physics constraints"""
        
        pred_swh = predictions[:, :, 0]
        pred_mwp = predictions[:, :, 3]
        
        # Physical bounds
        swh_penalty = F.relu(-pred_swh).mean()  # SWH >= 0
        mwp_penalty = F.relu(1 - pred_mwp).mean() + F.relu(pred_mwp - 25).mean()  # 1 <= MWP <= 25
        
        # Unit circle constraint for MWD
        pred_mwd_cos = predictions[:, :, 1]
        pred_mwd_sin = predictions[:, :, 2]
        circle_penalty = F.mse_loss(pred_mwd_cos**2 + pred_mwd_sin**2, 
                                   torch.ones_like(pred_mwd_cos))
        
        return swh_penalty + mwp_penalty + circle_penalty

def compute_circular_rmse(pred_angles: np.ndarray, true_angles: np.ndarray) -> float:
    """
    Compute RMSE for circular variables (proper angular distance)
    
    Args:
        pred_angles: Predicted angles in degrees
        true_angles: True angles in degrees
    
    Returns:
        Circular RMSE in degrees
    """
    # Convert to radians
    pred_rad = np.deg2rad(pred_angles)
    true_rad = np.deg2rad(true_angles)
    
    # Compute angular differences
    diff = pred_rad - true_rad
    
    # Wrap to [-π, π]
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    
    # Convert back to degrees and compute RMSE
    diff_deg = np.rad2deg(diff)
    rmse = np.sqrt(np.mean(diff_deg ** 2))
    
    return rmse

def evaluate_model_with_circular_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model with proper circular metrics for MWD
    
    Args:
        predictions: [N, 3] predictions [SWH, MWD, MWP]
        targets: [N, 3] targets [SWH, MWD, MWP]
    
    Returns:
        Dictionary of metrics
    """
    
    # Standard RMSE for SWH and MWP
    swh_rmse = np.sqrt(np.mean((predictions[:, 0] - targets[:, 0])**2))
    mwp_rmse = np.sqrt(np.mean((predictions[:, 2] - targets[:, 2])**2))
    
    # Circular RMSE for MWD
    mwd_circular_rmse = compute_circular_rmse(predictions[:, 1], targets[:, 1])
    
    # Overall RMSE (using circular for MWD)
    overall_rmse = np.mean([swh_rmse, mwd_circular_rmse, mwp_rmse])
    
    return {
        'swh_rmse': swh_rmse,
        'mwd_rmse': mwd_circular_rmse,  # Proper circular metric
        'mwp_rmse': mwp_rmse,
        'overall_rmse': overall_rmse
    }

# Example usage and testing
def test_circular_normalizer():
    """Test the circular normalizer"""
    
    print("🧪 Testing Circular Normalizer")
    print("-" * 40)
    
    # Test data with various angles including wraparound
    test_angles = np.array([0, 45, 90, 135, 180, 225, 270, 315, 359])
    
    # Fit normalizer
    normalizer = CircularNormalizer()
    normalizer.fit(test_angles)
    
    # Transform and inverse transform
    normalized = normalizer.transform(test_angles)
    reconstructed = normalizer.inverse_transform(normalized)
    
    print(f"Original angles: {test_angles}")
    print(f"Normalized shape: {normalized.shape}")
    print(f"Reconstructed: {reconstructed}")
    print(f"Max reconstruction error: {np.max(np.abs(test_angles - reconstructed)):.6f}°")
    
    # Test wraparound cases
    wraparound_test = np.array([359, 1, 358, 2])
    wraparound_norm = normalizer.transform(wraparound_test)
    wraparound_recon = normalizer.inverse_transform(wraparound_norm)
    
    print(f"\nWraparound test:")
    print(f"Original: {wraparound_test}")
    print(f"Reconstructed: {wraparound_recon}")
    
def test_circular_rmse():
    """Test circular RMSE calculation"""
    
    print("\n🧪 Testing Circular RMSE")
    print("-" * 40)
    
    # Test cases
    test_cases = [
        ([0, 90, 180, 270], [10, 100, 190, 280], "Small errors"),
        ([350, 10, 350, 10], [10, 350, 10, 350], "Wraparound errors"),
        ([0, 0, 0, 0], [180, 180, 180, 180], "Opposite directions")
    ]
    
    for pred, true, description in test_cases:
        pred_arr = np.array(pred)
        true_arr = np.array(true)
        
        # Standard RMSE (incorrect for circular)
        standard_rmse = np.sqrt(np.mean((pred_arr - true_arr)**2))
        
        # Circular RMSE (correct)
        circular_rmse = compute_circular_rmse(pred_arr, true_arr)
        
        print(f"{description}:")
        print(f"  Predicted: {pred}")
        print(f"  True: {true}")
        print(f"  Standard RMSE: {standard_rmse:.1f}°")
        print(f"  Circular RMSE: {circular_rmse:.1f}°")
        print()

if __name__ == "__main__":
    test_circular_normalizer()
    test_circular_rmse()