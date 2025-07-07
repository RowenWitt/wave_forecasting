# utils/logging.py
"""Experiment logging and tracking system"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import matplotlib.pyplot as plt

from config.base import ExperimentConfig

class ExperimentLogger:
    """Comprehensive experiment logging and tracking"""
    
    def __init__(self, config: ExperimentConfig, log_dir: str = "logs"):
        self.config = config
        self.log_dir = Path(log_dir)
        
        # Create unique experiment ID with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{config.name}_{self.timestamp}"
        
        # Create experiment directory
        self.experiment_dir = self.log_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging paths
        self.config_file = self.experiment_dir / "config.json"
        self.results_file = self.experiment_dir / "results.json"
        self.metrics_file = self.experiment_dir / "metrics.json"
        self.plots_dir = self.experiment_dir / "plots"
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        
        # Create subdirectories
        self.plots_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Initialize experiment log
        self.experiment_log = {
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp,
            'status': 'initialized',
            'config': self._config_to_dict(),
            'git_commit': self._get_git_commit(),
            'system_info': self._get_system_info(),
            'training_history': [],
            'final_metrics': {},
            'notes': []
        }
        
        # Save initial config
        self._save_config()
        
        print(f"📝 Experiment logger initialized: {self.experiment_id}")
        print(f"   Log directory: {self.experiment_dir}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to serializable dictionary"""
        
        def serialize_dataclass(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: serialize_dataclass(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (list, tuple)):
                return [serialize_dataclass(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: serialize_dataclass(v) for k, v in obj.items()}
            else:
                return obj
        
        return serialize_dataclass(self.config)
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash for reproducibility"""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for debugging"""
        import platform
        import psutil
        
        try:
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            }
        except:
            return {'error': 'Could not collect system info'}
    
    def _save_config(self):
        """Save experiment configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.experiment_log, f, indent=2, default=str)
    
    def log_training_start(self, model_info: Dict[str, Any]):
        """Log training start with model information"""
        self.experiment_log['status'] = 'training'
        self.experiment_log['model_info'] = model_info
        self.experiment_log['training_start_time'] = datetime.now().isoformat()
        
        print(f"🚀 Training started for {self.experiment_id}")
        self._save_config()
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float], 
                  learning_rate: float = None):
        """Log metrics for a single epoch"""
        
        epoch_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'learning_rate': learning_rate
        }
        
        self.experiment_log['training_history'].append(epoch_data)
        
        # Save updated log every 10 epochs
        if epoch % 10 == 0:
            self._save_config()
    
    def log_validation(self, epoch: int, val_metrics: Dict[str, float]):
        """Log validation metrics"""
        if self.experiment_log['training_history']:
            # Add to latest epoch entry
            self.experiment_log['training_history'][-1]['validation_metrics'] = val_metrics
        
        print(f"Epoch {epoch}: Val Loss = {val_metrics.get('total', 'N/A'):.4f}")
    
    def log_training_complete(self, final_metrics: Dict[str, float], 
                            model_path: str = None):
        """Log training completion"""
        self.experiment_log['status'] = 'completed'
        self.experiment_log['training_end_time'] = datetime.now().isoformat()
        self.experiment_log['final_metrics'] = final_metrics
        
        if model_path:
            self.experiment_log['final_model_path'] = str(model_path)
        
        # Calculate training duration
        if 'training_start_time' in self.experiment_log:
            start_time = datetime.fromisoformat(self.experiment_log['training_start_time'])
            end_time = datetime.fromisoformat(self.experiment_log['training_end_time'])
            duration = end_time - start_time
            self.experiment_log['training_duration_seconds'] = duration.total_seconds()
            self.experiment_log['training_duration_human'] = str(duration)
        
        print(f"✅ Training completed for {self.experiment_id}")
        print(f"   Final metrics: {final_metrics}")
        
        self._save_config()
        self._generate_summary_plots()
    
    def log_error(self, error: Exception):
        """Log training errors"""
        self.experiment_log['status'] = 'failed'
        self.experiment_log['error'] = {
            'type': type(error).__name__,
            'message': str(error),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"❌ Training failed for {self.experiment_id}: {error}")
        self._save_config()
    
    def add_note(self, note: str):
        """Add a note to the experiment"""
        note_entry = {
            'timestamp': datetime.now().isoformat(),
            'note': note
        }
        self.experiment_log['notes'].append(note_entry)
        print(f"📝 Note added: {note}")
        self._save_config()
    
    def save_checkpoint(self, model_state: Dict[str, Any], epoch: int, 
                       metrics: Dict[str, float]):
        """Save model checkpoint with metadata"""
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_data = {
            'experiment_id': self.experiment_id,
            'epoch': epoch,
            'metrics': metrics,
            'model_state_dict': model_state['model_state_dict'],
            'optimizer_state_dict': model_state['optimizer_state_dict'],
            'config': self._config_to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update experiment log
        if 'checkpoints' not in self.experiment_log:
            self.experiment_log['checkpoints'] = []
        
        self.experiment_log['checkpoints'].append({
            'epoch': epoch,
            'path': str(checkpoint_path),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def _generate_summary_plots(self):
        """Generate summary plots for the experiment"""
        if not self.experiment_log['training_history']:
            return
        
        # Extract training history
        epochs = [entry['epoch'] for entry in self.experiment_log['training_history']]
        train_losses = [entry['metrics'].get('total', 0) for entry in self.experiment_log['training_history']]
        
        # Check for validation data
        val_losses = []
        has_validation = False
        for entry in self.experiment_log['training_history']:
            if 'validation_metrics' in entry:
                val_losses.append(entry['validation_metrics'].get('total', 0))
                has_validation = True
            else:
                val_losses.append(None)
        
        # Create training plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Experiment: {self.experiment_id}', fontsize=16)
        
        # Loss plot
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        if has_validation and any(v is not None for v in val_losses):
            val_epochs = [e for e, v in zip(epochs, val_losses) if v is not None]
            val_losses_clean = [v for v in val_losses if v is not None]
            axes[0, 0].plot(val_epochs, val_losses_clean, 'r-', label='Validation Loss', linewidth=2)
        
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MSE vs Physics loss breakdown
        mse_losses = [entry['metrics'].get('mse', 0) for entry in self.experiment_log['training_history']]
        physics_losses = [entry['metrics'].get('physics', 0) for entry in self.experiment_log['training_history']]
        
        axes[0, 1].plot(epochs, mse_losses, 'g-', label='MSE Loss', linewidth=2)
        axes[0, 1].plot(epochs, physics_losses, 'm-', label='Physics Loss', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        learning_rates = [entry.get('learning_rate') for entry in self.experiment_log['training_history']]
        if any(lr is not None for lr in learning_rates):
            lr_epochs = [e for e, lr in zip(epochs, learning_rates) if lr is not None]
            lr_values = [lr for lr in learning_rates if lr is not None]
            
            axes[1, 0].plot(lr_epochs, lr_values, 'orange', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
Experiment Summary:
• Name: {self.config.name}
• Duration: {self.experiment_log.get('training_duration_human', 'N/A')}
• Epochs: {len(epochs)}
• Best Train Loss: {min(train_losses):.4f}
• Final Train Loss: {train_losses[-1]:.4f}
• Model: {self.config.model.hidden_dim}d, {self.config.model.num_spatial_layers} layers
• Batch Size: {self.config.training.batch_size}
• Learning Rate: {self.config.training.learning_rate}
        """
        
        if has_validation and val_losses_clean:
            summary_text += f"• Best Val Loss: {min(val_losses_clean):.4f}\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / "training_summary.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Summary plots saved to {plot_path}")
    
    def generate_report(self) -> str:
        """Generate a markdown report for the experiment"""
        
        report_path = self.experiment_dir / "experiment_report.md"
        
        # Calculate some statistics
        if self.experiment_log['training_history']:
            train_losses = [entry['metrics'].get('total', 0) for entry in self.experiment_log['training_history']]
            best_loss = min(train_losses)
            final_loss = train_losses[-1]
            improvement = ((train_losses[0] - final_loss) / train_losses[0]) * 100
        else:
            best_loss = final_loss = improvement = 0
        
        report_content = f"""# Experiment Report: {self.experiment_id}

## Overview
- **Experiment Name**: {self.config.name}
- **Timestamp**: {self.timestamp}
- **Status**: {self.experiment_log['status']}
- **Duration**: {self.experiment_log.get('training_duration_human', 'N/A')}

## Configuration
### Model Architecture
- **Hidden Dimension**: {self.config.model.hidden_dim}
- **Spatial Layers**: {self.config.model.num_spatial_layers}
- **Temporal Layers**: {self.config.model.num_temporal_layers}

### Training Parameters
- **Epochs**: {self.config.training.num_epochs}
- **Batch Size**: {self.config.training.batch_size}
- **Learning Rate**: {self.config.training.learning_rate}
- **Weight Decay**: {self.config.training.weight_decay}

### Data Configuration
- **Sequence Length**: {self.config.sequence_length}
- **Forecast Horizon**: {self.config.forecast_horizon}h
- **Geographic Bounds**: {self.config.data.lat_bounds} × {self.config.data.lon_bounds}

## Training Results
- **Best Training Loss**: {best_loss:.6f}
- **Final Training Loss**: {final_loss:.6f}
- **Improvement**: {improvement:.1f}%
- **Total Epochs Completed**: {len(self.experiment_log['training_history'])}

## Final Metrics
"""
        
        for metric, value in self.experiment_log.get('final_metrics', {}).items():
            report_content += f"- **{metric}**: {value:.6f}\n"
        
        report_content += f"""
## System Information
- **Platform**: {self.experiment_log['system_info'].get('platform', 'Unknown')}
- **Python Version**: {self.experiment_log['system_info'].get('python_version', 'Unknown')}
- **PyTorch Version**: {self.experiment_log['system_info'].get('torch_version', 'Unknown')}
- **CUDA Available**: {self.experiment_log['system_info'].get('cuda_available', False)}
- **Memory**: {self.experiment_log['system_info'].get('memory_gb', 'Unknown')} GB

## Files Generated
- **Config**: `{self.config_file.name}`
- **Training History**: `{self.results_file.name}`
- **Summary Plot**: `plots/training_summary.png`
- **Checkpoints**: `checkpoints/`

## Notes
"""
        
        for note in self.experiment_log.get('notes', []):
            report_content += f"- **{note['timestamp']}**: {note['note']}\n"
        
        if self.experiment_log.get('git_commit'):
            report_content += f"\n## Reproducibility\n- **Git Commit**: `{self.experiment_log['git_commit']}`\n"
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"📄 Experiment report saved to {report_path}")
        return str(report_path)

def list_experiments(log_dir: str = "logs") -> None:
    """List all logged experiments"""
    
    log_path = Path(log_dir)
    if not log_path.exists():
        print("No experiments found.")
        return
    
    experiments = []
    for exp_dir in log_path.iterdir():
        if exp_dir.is_dir():
            config_file = exp_dir / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    experiments.append((exp_dir.name, config))
                except:
                    continue
    
    if not experiments:
        print("No valid experiments found.")
        return
    
    print("📊 Experiment History:")
    print("=" * 80)
    
    for exp_name, config in sorted(experiments, key=lambda x: x[0]):
        status = config.get('status', 'unknown')
        timestamp = config.get('timestamp', 'unknown')
        name = config.get('config', {}).get('name', 'unnamed')
        
        duration = config.get('training_duration_human', 'N/A')
        final_loss = 'N/A'
        
        if config.get('training_history'):
            final_loss = f"{config['training_history'][-1]['metrics'].get('total', 0):.4f}"
        
        print(f"🔹 {exp_name}")
        print(f"   Name: {name} | Status: {status} | Duration: {duration}")
        print(f"   Final Loss: {final_loss} | Timestamp: {timestamp}")
        print()
