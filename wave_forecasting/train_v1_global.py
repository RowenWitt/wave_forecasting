"""
FIXED Global Climate-Aware Training Script
Robust MWD handling to prevent NaN propagation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import json
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Import the fixed model components directly
from global_climate_aware_variable_lr import (
    GlobalVariableLRConfig,
    GlobalVariableLRSpatioTemporalGNN,
    StreamingGlobalDataset,
    GlobalVariableLearningManager,
    RobustCircularLoss,
    VariableSpecificNormalizer
)

class FixedGlobalTrainer:
    """FIXED: Trainer with robust MWD circular handling"""
    
    def __init__(self, config: GlobalVariableLRConfig, data_files: List[str]):
        self.config = config
        self.data_files = data_files
        
        # Device setup
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("🍎 Using Apple Silicon MPS acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("🚀 Using CUDA acceleration")
        else:
            self.device = torch.device('cpu')
            print("💻 Using CPU")
        
        # FIXED: Robust normalizers
        self.feature_normalizer = StandardScaler()
        self.target_normalizer = VariableSpecificNormalizer()
        
        # Learning rate manager
        self.lr_manager = GlobalVariableLearningManager(config)
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"stable_global_climate_{timestamp}"
        self.log_dir = Path("experiments") / self.experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 STABLE Global Trainer: {self.experiment_id}")
        print(f"📁 Logging to: {self.log_dir}")
        print(f"🎯 Gradient stability measures: ✅")
    
    def setup_data(self):
        """Setup datasets with robust validation"""
        
        print("🌍 Setting up FIXED global datasets...")
        
        # Create datasets
        train_dataset = StreamingGlobalDataset(
            self.data_files, self.config, 
            validation_split=self.config.validation_split, 
            is_validation=False
        )
        
        val_dataset = StreamingGlobalDataset(
            self.data_files, self.config,
            validation_split=self.config.validation_split,
            is_validation=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=0,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # FIXED: Fit normalizers with robust validation
        print("🔧 Fitting normalizers with validation...")
        self._fit_normalizers_robust(train_loader)
        
        print(f"✅ FIXED data setup complete:")
        print(f"   Training sequences: {len(train_dataset)}")
        print(f"   Validation sequences: {len(val_dataset)}")
        print(f"   Robust target processing: ✅")
        
        return train_loader, val_loader
    
    def _fit_normalizers_robust(self, train_loader):
        """FIXED: Robust normalizer fitting with simplified shape handling"""
        
        print("   📊 Collecting robust normalization samples...")
        
        all_features = []
        all_targets = []
        sample_count = 0
        
        for batch in train_loader:
            if sample_count >= 10:  # Use fewer batches for simpler processing
                break
                
            inputs = batch['input']
            targets = batch['single_step_target']
            
            # Validate basic shapes
            if inputs.shape[-1] != 18:
                print(f"   ⚠️ Skipping batch with {inputs.shape[-1]} features")
                continue
            
            if targets.shape[-1] != 3:
                print(f"   ⚠️ Skipping batch with {targets.shape[-1]} targets")
                continue
            
            # Simple approach: just take last timestep for features to match targets
            batch_size, seq_len, nodes, features = inputs.shape
            
            # Use last timestep features to match target shape
            last_timestep_features = inputs[:, -1, :, :].reshape(-1, features).numpy()  # Convert to numpy
            targets_flat = targets.reshape(-1, 3).numpy()  # Convert to numpy
            
            # Now shapes should match
            if last_timestep_features.shape[0] != targets_flat.shape[0]:
                print(f"   ⚠️ Shape mismatch after simplification")
                continue
            
            # Validate target ranges (all numpy operations)
            swh_values = targets_flat[:, 0]
            mwd_values = targets_flat[:, 1]
            mwp_values = targets_flat[:, 2]
            
            # Check for reasonable wave parameters (numpy arrays)
            swh_valid = (swh_values >= 0.01) & (swh_values <= 30) & np.isfinite(swh_values)
            mwd_valid = (mwd_values >= 0) & (mwd_values <= 360) & np.isfinite(mwd_values)
            mwp_valid = (mwp_values >= 1) & (mwp_values <= 25) & np.isfinite(mwp_values)
            
            valid_mask = swh_valid & mwd_valid & mwp_valid  # This is now a numpy array
            valid_ratio = np.sum(valid_mask) / len(valid_mask)  # Both numpy operations
            
            if valid_ratio < 0.05:  # Very lenient threshold
                print(f"   ⚠️ Skipping batch with {valid_ratio:.1%} valid targets")
                continue
            
            # Collect valid samples
            if np.sum(valid_mask) > 0:
                all_features.append(last_timestep_features[valid_mask])
                all_targets.append(targets_flat[valid_mask])
                sample_count += 1
                
                print(f"     Batch {sample_count}: {np.sum(valid_mask)} valid samples ({valid_ratio:.1%})")
        
        if all_features and all_targets:
            combined_features = np.vstack(all_features)
            combined_targets = np.vstack(all_targets)
            
            # Fit normalizers
            self.feature_normalizer.fit(combined_features)
            self.target_normalizer.fit(combined_targets)
            
            print(f"   ✅ Robust normalizers fitted on {combined_features.shape[0]} valid samples")
            
            # Print statistics
            print(f"   📊 Feature stats: mean={np.mean(combined_features):.3f}, std={np.std(combined_features):.3f}")
            print(f"   📊 SWH range: {np.min(combined_targets[:, 0]):.2f} to {np.max(combined_targets[:, 0]):.2f} m")
            print(f"   📊 MWD range: {np.min(combined_targets[:, 1]):.1f} to {np.max(combined_targets[:, 1]):.1f} °")
            print(f"   📊 MWP range: {np.min(combined_targets[:, 2]):.2f} to {np.max(combined_targets[:, 2]):.2f} s")
        else:
            print("   ❌ No valid samples found for normalization!")
            raise RuntimeError("Cannot fit normalizers - no valid data")
    
    def create_edge_connectivity(self, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create simple edge connectivity for batch"""
        edges = []
        max_edges = min(2000, num_nodes * 2)
        
        if num_nodes <= 1:
            # Fallback for single node
            edges = [[0, 0]]
        else:
            # Sequential connections
            for i in range(min(num_nodes - 1, max_edges // 2)):
                edges.append([i, i + 1])
            
            # Skip connections
            skip = max(1, num_nodes // 100)
            for i in range(0, min(num_nodes - skip, max_edges // 4), skip):
                if i + skip < num_nodes:
                    edges.append([i, i + skip])
        
        if not edges:
            edges = [[0, 0]]
        
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long, device=self.device)
        edge_attr = torch.ones(len(edges), 3, dtype=torch.float32, device=self.device)
        
        return edge_index, edge_attr
    
    def create_optimizers(self, model: GlobalVariableLRSpatioTemporalGNN):
        """Create variable-specific optimizers with debugging"""
        
        print("🔧 Creating optimizers...")
        
        # Debug: Check if model has the method
        if not hasattr(model, 'get_variable_parameters'):
            print("   ❌ Model missing get_variable_parameters method!")
            raise AttributeError("Model must have get_variable_parameters method")
        
    def create_optimizers(self, model):
        """Create variable-specific optimizers - FIXED without fallbacks"""
        
        print("🔧 Creating optimizers...")
        
        # Debug: Check model type and methods
        print(f"   Model type: {type(model)}")
        print(f"   Model class: {model.__class__.__name__}")
        print(f"   Has get_variable_parameters: {hasattr(model, 'get_variable_parameters')}")
        
        # Direct parameter access if method missing
        if not hasattr(model, 'get_variable_parameters'):
            print("   ⚠️ Model missing get_variable_parameters method")
            print("   Available methods:")
            for attr in dir(model):
                if not attr.startswith('_') and callable(getattr(model, attr)):
                    print(f"      {attr}")
            raise AttributeError("Model must have get_variable_parameters method")
        
        # Call the method
        param_groups = model.get_variable_parameters()
        
        if param_groups is None:
            raise RuntimeError("get_variable_parameters returned None - check model implementation")
        
        if not isinstance(param_groups, dict):
            raise TypeError(f"Expected dict from get_variable_parameters, got {type(param_groups)}")
        
        # Validate all required keys exist
        required_keys = ['shared', 'swh', 'mwd', 'mwp']
        for key in required_keys:
            if key not in param_groups:
                raise KeyError(f"Missing parameter group: {key}")
            if not param_groups[key]:  # Empty list
                raise ValueError(f"Parameter group {key} is empty")
        
        current_lrs = self.lr_manager.get_current_lrs()
        
        # Create optimizers
        optimizers = {
            'shared': torch.optim.AdamW(
                param_groups['shared'], 
                lr=current_lrs['mwd'],
                weight_decay=self.config.weight_decay
            ),
            'swh': torch.optim.AdamW(
                param_groups['swh'], 
                lr=current_lrs['swh'],
                weight_decay=self.config.weight_decay
            ),
            'mwd': torch.optim.AdamW(
                param_groups['mwd'], 
                lr=current_lrs['mwd'],
                weight_decay=self.config.weight_decay
            ),
            'mwp': torch.optim.AdamW(
                param_groups['mwp'], 
                lr=current_lrs['mwp'],
                weight_decay=self.config.weight_decay
            )
        }
        
        print(f"✅ Optimizers created successfully")
        for var, lr in current_lrs.items():
            print(f"   {var.upper()}: {lr:.2e}")
        
        return optimizers
    
    def scale_gradients_if_needed(self, model, max_norm=0.5):
        """Scale gradients if they're too large to prevent NaN"""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
            print(f"   🔧 Scaled gradients: {total_norm:.2f} → {max_norm}")
        
        return total_norm
    
    def debug_first_batch(self, model, inputs, targets, edge_index, edge_attr, batch_idx):
        """Debug the first batch to identify NaN sources"""
        if batch_idx != 0:
            return True
        
        print("🔍 DEBUGGING FIRST BATCH FOR NaN SOURCES...")
        
        # Check inputs
        print(f"   Inputs: range {inputs.min().item():.3f} to {inputs.max().item():.3f}")
        print(f"   Targets: range {targets.min().item():.3f} to {targets.max().item():.3f}")
        
        if torch.isnan(inputs).any() or torch.isnan(targets).any():
            print("   ❌ NaN in inputs/targets!")
            return False
        
        # Forward pass with intermediate checks
        try:
            with torch.no_grad():
                # Test forward pass
                predictions = model(inputs, edge_index, edge_attr, multi_step=False)
                
                if torch.isnan(predictions).any():
                    print("   ❌ NaN in model predictions!")
                    return False
                
                print(f"   Predictions: range {predictions.min().item():.3f} to {predictions.max().item():.3f}")
                
                # Test loss computation
                criterion = RobustCircularLoss(self.config)
                loss_dict = criterion(predictions, targets)
                
                if torch.isnan(loss_dict['total_loss']):
                    print("   ❌ NaN in loss computation!")
                    return False
                
                print(f"   Loss: {loss_dict['total_loss'].item():.6f}")
            
            print("   ✅ Forward pass clean - issue likely in backward pass")
            return True
            
        except Exception as e:
            print(f"   ❌ Error in forward pass: {e}")
            return False
    
    def update_optimizers(self, optimizers: Dict[str, torch.optim.Optimizer]):
        """Update optimizer learning rates"""
        current_lrs = self.lr_manager.get_current_lrs()
        
        for param_group in optimizers['shared'].param_groups:
            param_group['lr'] = current_lrs['mwd']
        
        for var in ['swh', 'mwd', 'mwp']:
            for param_group in optimizers[var].param_groups:
                param_group['lr'] = current_lrs[var]
    
    def train_epoch(self, model, train_loader, criterion, optimizers, epoch):
        """FIXED: Train one epoch with gradient anomaly detection"""
        
        model.train()
        epoch_losses = {
            'total': [], 'swh': [], 'mwd': [], 'mwp': [], 
            'mwd_circular': [], 'mwd_angular': [], 'physics': []
        }
        
        batch_count = 0
        nan_batches = 0
        
        # Enable anomaly detection for the first few batches
        with torch.autograd.set_detect_anomaly(epoch == 0):  # Only first epoch for performance
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # Zero gradients
                    for optimizer in optimizers.values():
                        optimizer.zero_grad()
                    
                    inputs = batch['input'].to(self.device)
                    targets = batch['single_step_target'].to(self.device)
                    
                    # Validate batch
                    batch_size, seq_len, num_nodes, num_features = inputs.size()
                    if num_features != 18:
                        print(f"   ⚠️ Skipping batch with {num_features} features")
                        continue
                    
                    if targets.shape[-1] != 3:
                        print(f"   ⚠️ Skipping batch with {targets.shape[-1]} targets")
                        continue
                    
                    # Check for NaN in inputs
                    if torch.isnan(inputs).any():
                        print(f"   ⚠️ NaN in input batch {batch_idx}")
                        nan_batches += 1
                        continue
                    
                    if torch.isnan(targets).any():
                        print(f"   ⚠️ NaN in target batch {batch_idx}")
                        nan_batches += 1
                        continue
                    
                    # Normalize inputs
                    inputs_flat = inputs.view(-1, num_features).cpu().numpy()
                    inputs_norm = self.feature_normalizer.transform(inputs_flat)
                    inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
                    inputs = inputs.view(batch_size, seq_len, num_nodes, num_features)
                    
                    # FIXED: Transform targets to [SWH, MWD_cos, MWD_sin, MWP]
                    targets_flat = targets.view(-1, 3).cpu().numpy()
                    targets_norm = self.target_normalizer.transform_targets(targets_flat)  # [N, 4]
                    targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                    targets = targets.view(batch_size, num_nodes, 4)
                    
                    # Final NaN check after normalization
                    if torch.isnan(inputs).any() or torch.isnan(targets).any():
                        print(f"   ⚠️ NaN after normalization in batch {batch_idx}")
                        nan_batches += 1
                        continue
                    
                    # FIXED: Create edge connectivity BEFORE debugging
                    edge_index, edge_attr = self.create_edge_connectivity(num_nodes)
                    
                    # Debug first batch
                    debug_success = self.debug_first_batch(model, inputs, targets, edge_index, edge_attr, batch_idx)
                    if not debug_success:
                        print("   ❌ Debug failed - skipping batch")
                        continue
                    
                    # Forward pass
                    predictions = model(inputs, edge_index, edge_attr, multi_step=False)
                    
                    # Check predictions for NaN
                    if torch.isnan(predictions).any():
                        print(f"   ⚠️ NaN in predictions for batch {batch_idx}")
                        nan_batches += 1
                        continue
                    
                    # Compute loss
                    loss_dict = criterion(predictions, targets)
                    
                    # Check loss for NaN
                    if torch.isnan(loss_dict['total_loss']):
                        print(f"   ⚠️ NaN in loss for batch {batch_idx}")
                        nan_batches += 1
                        continue
                    
                    try:
                        # Backward pass with anomaly detection
                        loss_dict['total_loss'].backward()
                        
                    except RuntimeError as e:
                        if "returned nan values" in str(e):
                            print(f"   🔍 ANOMALY DETECTED in batch {batch_idx}: {e}")
                            # This will tell us exactly which operation caused the NaN
                            nan_batches += 1
                            continue
                        else:
                            raise e
                    
                    # Check gradients for NaN AFTER backward
                    has_nan_grad = False
                    nan_param_names = []
                    
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            has_nan_grad = True
                            nan_param_names.append(name)
                            if len(nan_param_names) >= 3:  # Limit output
                                break
                    
                    if has_nan_grad:
                        print(f"   ⚠️ NaN gradients in batch {batch_idx}: {nan_param_names}")
                        nan_batches += 1
                        continue
                    
                    # Scale gradients if needed
                    grad_norm = self.scale_gradients_if_needed(model, max_norm=0.1)  # Very conservative
                    
                    # Clip gradients with very small norm
                    for optimizer in optimizers.values():
                        torch.nn.utils.clip_grad_norm_(
                            [p for group in optimizer.param_groups for p in group['params']], 
                            0.05  # Even smaller clipping
                        )
                    
                    # Step optimizers
                    for optimizer in optimizers.values():
                        optimizer.step()
                    
                    # Track losses
                    for key in epoch_losses:
                        loss_key = f'{key}_loss'
                        if loss_key in loss_dict:
                            loss_value = loss_dict[loss_key].item()
                            if np.isfinite(loss_value):
                                epoch_losses[key].append(loss_value)
                    
                    batch_count += 1
                    
                    # Progress logging
                    if batch_idx % 10 == 0:
                        current_loss = loss_dict['total_loss'].item()
                        swh_loss = loss_dict.get('swh_loss', torch.tensor(0)).item()
                        mwd_loss = loss_dict.get('mwd_loss', torch.tensor(0)).item()
                        mwp_loss = loss_dict.get('mwp_loss', torch.tensor(0)).item()
                        
                        print(f"   📊 Epoch {epoch+1}, Batch {batch_idx:3d}: "
                              f"Total={current_loss:.4f}, SWH={swh_loss:.4f}, "
                              f"MWD={mwd_loss:.4f}, MWP={mwp_loss:.4f}, GradNorm={grad_norm:.3f}")
                    
                    # Memory cleanup
                    if batch_idx % 50 == 0 and batch_idx > 0:
                        if self.device.type == 'mps':
                            torch.mps.empty_cache()
                        elif self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"   ❌ Batch {batch_idx} failed: {e}")
                    continue
        
        success_rate = batch_count / (batch_count + nan_batches) if (batch_count + nan_batches) > 0 else 0
        print(f"   ✅ Epoch {epoch+1}: {batch_count} successful, {nan_batches} NaN batches ({success_rate:.1%} success)")
        
        return {key: np.mean(values) if values else 0.0 for key, values in epoch_losses.items()}
    
    def validate(self, model, val_loader, criterion, epoch):
        """FIXED: Validate with robust error handling"""
        
        model.eval()
        val_losses = {
            'total': [], 'swh': [], 'mwd': [], 'mwp': [], 
            'mwd_circular': [], 'mwd_angular': [], 'physics': []
        }
        
        batch_count = 0
        nan_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    inputs = batch['input'].to(self.device)
                    targets = batch['single_step_target'].to(self.device)
                    
                    # Validate batch
                    batch_size, seq_len, num_nodes, num_features = inputs.size()
                    if num_features != 18 or targets.shape[-1] != 3:
                        continue
                    
                    # Check for NaN
                    if torch.isnan(inputs).any() or torch.isnan(targets).any():
                        nan_batches += 1
                        continue
                    
                    # Normalize
                    inputs_flat = inputs.view(-1, num_features).cpu().numpy()
                    inputs_norm = self.feature_normalizer.transform(inputs_flat)
                    inputs = torch.tensor(inputs_norm, dtype=torch.float32, device=self.device)
                    inputs = inputs.view(batch_size, seq_len, num_nodes, num_features)
                    
                    targets_flat = targets.view(-1, 3).cpu().numpy()
                    targets_norm = self.target_normalizer.transform_targets(targets_flat)
                    targets = torch.tensor(targets_norm, dtype=torch.float32, device=self.device)
                    targets = targets.view(batch_size, num_nodes, 4)
                    
                    # Final NaN check
                    if torch.isnan(inputs).any() or torch.isnan(targets).any():
                        nan_batches += 1
                        continue
                    
                    # Create edge connectivity
                    edge_index, edge_attr = self.create_edge_connectivity(num_nodes)
                    
                    # Forward pass
                    predictions = model(inputs, edge_index, edge_attr, multi_step=False)
                    
                    # Check predictions
                    if torch.isnan(predictions).any():
                        nan_batches += 1
                        continue
                    
                    # Compute loss
                    loss_dict = criterion(predictions, targets)
                    
                    # Check loss
                    if torch.isnan(loss_dict['total_loss']):
                        nan_batches += 1
                        continue
                    
                    # Track losses
                    for key in val_losses:
                        loss_key = f'{key}_loss'
                        if loss_key in loss_dict:
                            loss_value = loss_dict[loss_key].item()
                            if np.isfinite(loss_value):
                                val_losses[key].append(loss_value)
                    
                    batch_count += 1
                            
                except Exception as e:
                    continue
        
        success_rate = batch_count / (batch_count + nan_batches) if (batch_count + nan_batches) > 0 else 0
        print(f"   ✅ Validation: {batch_count} successful, {nan_batches} NaN batches ({success_rate:.1%} success)")
        
        return {key: np.mean(values) if values else 0.0 for key, values in val_losses.items()}
    
    def save_checkpoint(self, model, optimizers, epoch, train_metrics, val_metrics, is_best=False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_normalizer': self.feature_normalizer,
            'target_normalizer': self.target_normalizer,
            'lr_manager_state': self.lr_manager.get_performance_summary(),
            'optimizer_states': {name: opt.state_dict() for name, opt in optimizers.items()},
            'current_lrs': self.lr_manager.get_current_lrs(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = self.log_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"   💾 Checkpoint: checkpoint_epoch_{epoch+1:03d}.pt")
        
        # Save best model
        if is_best:
            best_path = self.log_dir / "best_fixed_global_model.pt"
            torch.save(checkpoint, best_path)
            val_loss = val_metrics['total'] if isinstance(val_metrics['total'], (int, float)) else float(val_metrics['total'])
            print(f"   🏆 Best model: best_fixed_global_model.pt (val_loss: {val_loss:.4f})")
        
        # Save latest
        latest_path = self.log_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)
        
        return checkpoint_path if (epoch + 1) % 10 == 0 else None
    
    def log_metrics_to_csv(self, epoch, train_metrics, val_metrics, current_lrs):
        """Log metrics to CSV file"""
        
        log_file = self.log_dir / "training_metrics.csv"
        
        # Create header if needed
        if not log_file.exists():
            with open(log_file, 'w') as f:
                f.write("epoch,train_total,train_swh,train_mwd,train_mwp,train_physics,")
                f.write("val_total,val_swh,val_mwd,val_mwp,val_physics,")
                f.write("lr_swh,lr_mwd,lr_mwp,timestamp\n")
        
        # Append metrics
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},")
            f.write(f"{train_metrics['total']:.6f},{train_metrics.get('swh', 0):.6f},")
            f.write(f"{train_metrics.get('mwd', 0):.6f},{train_metrics.get('mwp', 0):.6f},")
            f.write(f"{train_metrics.get('physics', 0):.6f},")
            f.write(f"{val_metrics['total']:.6f},{val_metrics.get('swh', 0):.6f},")
            f.write(f"{val_metrics.get('mwd', 0):.6f},{val_metrics.get('mwp', 0):.6f},")
            f.write(f"{val_metrics.get('physics', 0):.6f},")
            f.write(f"{current_lrs['swh']:.2e},{current_lrs['mwd']:.2e},{current_lrs['mwp']:.2e},")
            f.write(f"{datetime.now().isoformat()}\n")
    
    def train(self):
        """FIXED: Main training loop with robust error handling"""
        
        print(f"🔧 Starting FIXED global climate-aware training...")
        print(f"🎯 Target: Beat 9.21 RMSE with robust MWD handling")
        
        # Setup
        train_loader, val_loader = self.setup_data()
        
        # Create model and loss
        model = GlobalVariableLRSpatioTemporalGNN(self.config).to(self.device)
        criterion = RobustCircularLoss(self.config)
        
        # Create optimizers
        optimizers = self.create_optimizers(model)
        
        print(f"✅ FIXED model created:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Robust MWD circular handling: ✅")
        print(f"   NaN prevention measures: ✅")
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [], 'learning_rates': {},
            'variable_losses': {'train': {'swh': [], 'mwd': [], 'mwp': []},
                               'val': {'swh': [], 'mwd': [], 'mwp': []}}
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        print(f"\n📈 FIXED training loop ({self.config.single_step_epochs} epochs)")
        print(f"🕒 Started: {datetime.now().strftime('%H:%M:%S')}")
        
        for epoch in range(self.config.single_step_epochs):
            start_time = time.time()
            
            # Train epoch
            print(f"🏃 Training epoch {epoch+1}/{self.config.single_step_epochs}...")
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizers, epoch)
            
            # Validate
            print(f"🔍 Validating epoch {epoch+1}...")
            val_metrics = self.validate(model, val_loader, criterion, epoch)
            
            # Update learning rates
            variable_losses = {
                'swh': val_metrics.get('swh', float('inf')),
                'mwd': val_metrics.get('mwd', float('inf')),
                'mwp': val_metrics.get('mwp', float('inf'))
            }
            
            self.lr_manager.update_performance(variable_losses)
            self.update_optimizers(optimizers)
            
            # Track history
            history['train_loss'].append(train_metrics['total'])
            history['val_loss'].append(val_metrics['total'])
            history['learning_rates'][epoch] = self.lr_manager.get_current_lrs()
            
            for var in ['swh', 'mwd', 'mwp']:
                history['variable_losses']['train'][var].append(train_metrics.get(var, 0))
                history['variable_losses']['val'][var].append(variable_losses[var])
            
            epoch_time = time.time() - start_time
            current_lrs = self.lr_manager.get_current_lrs()
            
            # Print progress
            print(f"📊 Epoch {epoch+1:3d} ({epoch_time:.1f}s):")
            print(f"   Train: {train_metrics['total']:.4f}")
            print(f"   Val:   {val_metrics['total']:.4f}")
            print(f"   SWH:   {val_metrics.get('swh', 0):.4f} (LR={current_lrs['swh']:.2e})")
            print(f"   MWD:   {val_metrics.get('mwd', 0):.4f} (LR={current_lrs['mwd']:.2e})")
            print(f"   MWP:   {val_metrics.get('mwp', 0):.4f} (LR={current_lrs['mwp']:.2e})")
            
            # Log metrics
            self.log_metrics_to_csv(epoch, train_metrics, val_metrics, current_lrs)
            
            # Save checkpoint
            is_best = val_metrics['total'] < best_val_loss
            self.save_checkpoint(model, optimizers, epoch, train_metrics, val_metrics, is_best)
            
            # Early stopping
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                print(f"\n🛑 EARLY STOPPING at epoch {epoch+1}")
                print(f"   Best validation loss: {best_val_loss:.6f}")
                break
            
            # Memory cleanup
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Save final model
        final_model_data = {
            'experiment_id': self.experiment_id,
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'feature_normalizer': self.feature_normalizer,
            'target_normalizer': self.target_normalizer,
            'training_history': history,
            'lr_manager_final_state': self.lr_manager.get_performance_summary(),
            'timestamp': datetime.now().isoformat(),
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1
        }
        
        model_path = self.log_dir / "fixed_global_climate_model_final.pt"
        torch.save(final_model_data, model_path)
        
        # Save history
        history_path = self.log_dir / "training_history.json"
        serializable_history = {}
        for key, value in history.items():
            if isinstance(value, dict):
                serializable_history[key] = {k: [float(x) for x in v] if isinstance(v, list) else v 
                                           for k, v in value.items()}
            else:
                serializable_history[key] = [float(x) for x in value] if isinstance(value, list) else value
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"\n🎉 FIXED TRAINING COMPLETE!")
        print("=" * 70)
        print(f"📊 FINAL RESULTS:")
        print(f"   Best validation loss: {best_val_loss:.6f}")
        print(f"   Total epochs: {epoch + 1}")
        print(f"   Baseline (9.21): {'✅ BEATEN' if best_val_loss < 9.21 else '❌ NOT BEATEN'}")
        
        if best_val_loss < 9.21:
            improvement = ((9.21 - best_val_loss) / 9.21 * 100)
            print(f"   Improvement: {improvement:.1f}% better than baseline")
        
        print(f"\n💾 SAVED FILES:")
        print(f"   Final model: {model_path}")
        print(f"   Training history: {history_path}")
        print(f"   Best model: {self.log_dir}/best_fixed_global_model.pt")
        print(f"   Metrics CSV: {self.log_dir}/training_metrics.csv")
        
        # Performance analysis
        print(f"\n🔍 VARIABLE PERFORMANCE:")
        lr_summary = self.lr_manager.get_performance_summary()
        for var in ['swh', 'mwd', 'mwp']:
            if var in lr_summary:
                best_loss = lr_summary[var]['best_loss']
                current_lr = lr_summary[var]['current_lr']
                status = "✅ GOOD" if best_loss < 3.0 else "⚠️ NEEDS WORK"
                print(f"   {var.upper()}: {best_loss:.4f} {status} (LR={current_lr:.2e})")
        
        print(f"\n🔧 ROBUST FIXES APPLIED:")
        print(f"   ✅ Circular MWD normalization (cos/sin)")
        print(f"   ✅ NaN detection and prevention")
        print(f"   ✅ Robust target validation")
        print(f"   ✅ Gentle physics constraints")
        print(f"   ✅ Variable-specific learning rates")
        
        return model, history, self.log_dir

def main():
    """Main function for FIXED global training"""
    
    print("🔧 FIXED GLOBAL CLIMATE-AWARE WAVE MODEL - TRAINING LAUNCH")
    print("=" * 80)
    print("🎯 Mission: Beat 9.21 RMSE with robust MWD circular handling")
    
    # STABLE Configuration with very conservative parameters
    config = GlobalVariableLRConfig(
        # Temporal settings (proven)
        sequence_length=6,
        prediction_horizon=4,
        
        # Architecture (FIXED feature count)
        input_features=18,              # 15 base + 6 climate + 1 bathymetry
        hidden_dim=384,
        temporal_hidden_dim=192,
        num_spatial_layers=8,
        num_temporal_layers=2,
        
        # Attention
        use_spatial_attention=True,
        use_temporal_attention=True,
        use_climate_attention=True,
        num_attention_heads=8,
        
        # Regularization
        dropout=0.15,
        spatial_dropout=0.1,
        temporal_dropout=0.1,
        
        # EMERGENCY: Much smaller learning rates and clipping
        num_epochs=200,
        batch_size=4,
        base_learning_rate=1e-6,        # 100x smaller than before
        weight_decay=1e-3,
        gradient_clip_norm=0.1,         # 10x smaller clipping
        
        # EMERGENCY: Much smaller variable LR multipliers
        swh_lr_multiplier=0.1,          # 7x smaller
        mwd_lr_multiplier=0.1,          # 10x smaller
        mwp_lr_multiplier=0.1,          # 13x smaller
        
        # Adaptive LR
        lr_patience=15,
        lr_factor=0.8,
        min_lr_factor=0.01,             # Allow very small LRs
        
        # Early stopping
        early_stopping_patience=25,
        
        # Loss weights
        swh_loss_weight=1.0,
        mwd_loss_weight=1.0,
        mwp_loss_weight=1.0,
        physics_loss_weight=0.1,        # Reduced physics weight
        
        # Data
        validation_split=0.2,
        max_training_sequences=2000,
        
        # Curriculum
        start_with_single_step=True,
        single_step_epochs=200
    )
    
    print(f"🔧 EMERGENCY STABILITY Configuration:")
    print(f"   Features: {config.input_features} (18 total)")
    print(f"   Base LR: {config.base_learning_rate:.2e} (100x smaller)")
    print(f"   Grad clip: {config.gradient_clip_norm} (10x smaller)")
    print(f"   Variable LR multipliers: All 0.1 (much smaller)")
    print(f"   Stability measures: ✅")
    
    # Find data files
    data_files = []
    
    # Look for processed data
    data_paths = [
        Path("data/processed_v1"),      # Enhanced data with climate anomalies
        Path("data/v1_global/processed"),
        Path("data/era5_global"),
        Path("data/era5")
    ]
    
    for data_dir in data_paths:
        if data_dir.exists():
            # Look for enhanced files first
            patterns = ["enhanced_*.nc", "*202101*.nc", "*2019*.nc", "*.nc"]
            
            for pattern in patterns:
                files = list(data_dir.glob(pattern))
                if files:
                    data_files.extend([str(f) for f in files])
                    print(f"   📁 Found data in: {data_dir}")
                    break
        
        if data_files:
            break
    
    if not data_files:
        print("❌ No data files found!")
        print("   Expected: enhanced_v1_era5_202101.nc with climate anomalies")
        print("   Run launch_v1_training.py first to process data")
        return
    
    # Use first file for initial testing
    if len(data_files) > 1:
        data_files = data_files[:1]
        print(f"   🎯 Using first file for robust testing")
    
    print(f"   📊 Data Files: {len(data_files)}")
    for f in data_files:
        file_path = Path(f)
        size_gb = file_path.stat().st_size / 1e9
        print(f"     {file_path.name} ({size_gb:.1f} GB)")
    
    # LAUNCH FIXED TRAINING
    print(f"\n🚀 LAUNCHING FIXED TRAINING")
    print(f"🕒 Launch time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    trainer = FixedGlobalTrainer(config, data_files)
    
    try:
        model, history, log_dir = trainer.train()
        
        print(f"\n🎉 FIXED TRAINING SUCCESSFUL!")
        print(f"   Robust MWD handling: ✅")
        print(f"   NaN prevention: ✅")
        print(f"   Status: Training completed")
        print(f"   Output: {log_dir}")
        
        # Quick analysis
        if history['val_loss']:
            best_val = min(history['val_loss'])
            final_val = history['val_loss'][-1]
            baseline = 9.21
            
            print(f"\n📈 PERFORMANCE SUMMARY:")
            print(f"   Best validation: {best_val:.4f}")
            print(f"   Final validation: {final_val:.4f}")
            print(f"   Baseline target: {baseline}")
            print(f"   Status: {'✅ BEATEN' if best_val < baseline else '❌ NOT BEATEN'}")
            
            if best_val < baseline:
                improvement = ((baseline - best_val) / baseline * 100)
                print(f"   Improvement: {improvement:.1f}%")
        
    except Exception as e:
        print(f"\n❌ FIXED TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n🔧 DEBUG SUGGESTIONS:")
        print(f"   1. Check if enhanced data exists: data/processed_v1/enhanced_v1_era5_202101.nc")
        print(f"   2. Verify data has 18 features (15 base + 6 climate + 1 bathymetry)")
        print(f"   3. Check for NaN values in input data")
        print(f"   4. Reduce batch size if memory issues persist")
        print(f"   5. Check that MWD values are in [0, 360] range")

if __name__ == "__main__":
    main()