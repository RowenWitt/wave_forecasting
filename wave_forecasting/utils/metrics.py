# utils/metrics.py
"""Performance evaluation metrics"""
import numpy as np
import torch
from typing import Dict, Tuple
from torch.utils.data import DataLoader

def calculate_wave_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Calculate wave-specific performance metrics"""
    
    predictions = predictions.detach().numpy()
    targets = targets.detach().numpy()
    
    metrics = {}
    wave_names = ['swh', 'mwd', 'mwp']
    
    for i, name in enumerate(wave_names):
        pred_vals = predictions[..., i].flatten()
        true_vals = targets[..., i].flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(pred_vals) | np.isnan(true_vals))
        pred_vals = pred_vals[valid_mask]
        true_vals = true_vals[valid_mask]
        
        if len(pred_vals) == 0:
            continue
        
        # Basic metrics
        mae = np.mean(np.abs(pred_vals - true_vals))
        rmse = np.sqrt(np.mean((pred_vals - true_vals)**2))
        
        # Skill score (relative to climatology)
        climatology_var = np.var(true_vals)
        model_var = np.var(pred_vals - true_vals)
        skill_score = 1 - (model_var / climatology_var) if climatology_var > 0 else 0
        
        # Correlation
        correlation = np.corrcoef(pred_vals, true_vals)[0, 1] if len(pred_vals) > 1 else 0
        
        metrics[f'{name}_mae'] = mae
        metrics[f'{name}_rmse'] = rmse
        metrics[f'{name}_skill'] = skill_score
        metrics[f'{name}_corr'] = correlation
    
    return metrics

def evaluate_model_performance(model, test_loader: DataLoader, edge_index: torch.Tensor,
                              edge_attr: torch.Tensor) -> Dict[str, float]:
    """Comprehensive model evaluation"""
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features']
            targets = batch['targets']
            
            # Handle both spatial and temporal models
            if hasattr(model, 'forecast_decoder'):  # Temporal model
                if 'input_sequences' in batch:
                    input_sequences = batch['input_sequences']
                    batch_predictions = []
                    for i in range(input_sequences.shape[0]):
                        pred = model(input_sequences[i], edge_index, edge_attr)
                        batch_predictions.append(pred)
                    predictions = torch.stack(batch_predictions)
                else:
                    # Convert features to sequence format
                    features_seq = features.unsqueeze(1)  # Add sequence dimension
                    predictions = model(features_seq, edge_index, edge_attr)
            else:  # Spatial model
                batch_predictions = []
                for i in range(features.shape[0]):
                    pred = model(features[i], edge_index, edge_attr)
                    batch_predictions.append(pred)
                predictions = torch.stack(batch_predictions)
            
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_wave_metrics(all_predictions, all_targets)
    
    return metrics
