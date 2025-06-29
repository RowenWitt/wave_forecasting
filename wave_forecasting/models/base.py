# models/base.py
"""Base model components"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class WaveMessageLayer(MessagePassing):
    """Single message passing layer for wave propagation"""
    
    def __init__(self, node_features: int, edge_features: int, hidden_dim: int = 64):
        super().__init__(aggr='add')
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        
        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_features + edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(node_features + hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, node_features)
        )
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)
    
    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)
