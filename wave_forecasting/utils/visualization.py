# utils/visualization.py
"""Visualization utilities"""
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.datasets import MeshDataLoader
from data.preprocessing import clean_features_for_training

def visualize_predictions(model, mesh_loader: MeshDataLoader, edge_index: torch.Tensor,
                         edge_attr: torch.Tensor, time_idx: int = 0, save_path: str = None):
    """Visualize model predictions on the mesh"""
    
    model.eval()
    
    # Get data
    input_data = mesh_loader.load_features(time_idx=time_idx)
    features = clean_features_for_training(
        torch.tensor(input_data['features'], dtype=torch.float32)
    )
    coordinates = input_data['coordinates']
    
    # Predict
    with torch.no_grad():
        predictions = model(features, edge_index, edge_attr)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    wave_names = ['Significant Wave Height (m)', 'Mean Wave Direction (°)', 'Mean Wave Period (s)']
    cmaps = ['Blues', 'twilight', 'Greens']
    
    for i in range(3):
        scatter = axes[i].scatter(
            coordinates[:, 1],  # longitude
            coordinates[:, 0],  # latitude
            c=predictions[:, i].numpy(),
            s=15, alpha=0.7, cmap=cmaps[i]
        )
        axes[i].set_xlabel('Longitude')
        axes[i].set_ylabel('Latitude')
        axes[i].set_title(f'Predicted {wave_names[i]}')
        axes[i].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """Plot training history"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    
    if 'val_loss' in history and history['val_loss']:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
