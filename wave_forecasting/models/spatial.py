# models/spatial.py
"""Spatial wave prediction models"""
import torch.nn as nn
from config.base import ModelConfig
from models.base import WaveMessageLayer
class SpatialWaveGNN(nn.Module):
    """GNN for single-timestep wave prediction"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),  # Will be set properly in forward
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            WaveMessageLayer(config.hidden_dim, config.edge_features, config.hidden_dim)
            for _ in range(config.num_spatial_layers)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_features)
        )
        
        self._input_features = None
    
    def _setup_encoder(self, input_features: int):
        """Setup encoder with correct input dimension"""
        if self._input_features != input_features:
            self.encoder[0] = nn.Linear(input_features, self.config.hidden_dim)
            self._input_features = input_features
    
    def forward(self, x, edge_index, edge_attr):
        # Setup encoder if needed
        if x.dim() == 2:  # [nodes, features]
            self._setup_encoder(x.shape[1])
        
        # Encode
        h = self.encoder(x)
        
        # Message passing with residual connections
        for layer in self.message_layers:
            h_new = layer(h, edge_index, edge_attr)
            h = h + h_new
        
        # Decode
        predictions = self.decoder(h)
        return predictions
