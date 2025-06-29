# models/temporal.py
"""Temporal sequence models"""

class TemporalWaveGNN(nn.Module):
    """GNN with temporal processing for multi-step forecasting"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Spatial encoder (per timestep)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),  # Will be set in forward
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Spatial message passing layers
        self.spatial_layers = nn.ModuleList([
            WaveMessageLayer(config.hidden_dim, config.edge_features, config.hidden_dim)
            for _ in range(config.num_spatial_layers)
        ])
        
        # Temporal processing
        self.temporal_processor = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_temporal_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Forecast decoder
        self.forecast_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.output_features)
        )
        
        self._input_features = None
    
    def _setup_spatial_encoder(self, input_features: int):
        """Setup spatial encoder with correct input dimension"""
        if self._input_features != input_features:
            self.spatial_encoder[0] = nn.Linear(input_features, self.config.hidden_dim)
            self._input_features = input_features
    
    def forward(self, input_sequence, edge_index, edge_attr):
        """
        input_sequence: [batch_size, seq_len, num_nodes, input_features] 
        OR [seq_len, num_nodes, input_features] for single sample
        """
        
        # Handle both batched and unbatched inputs
        if input_sequence.dim() == 3:
            input_sequence = input_sequence.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, num_nodes, input_features = input_sequence.shape
        
        # Setup spatial encoder
        self._setup_spatial_encoder(input_features)
        
        # Process each timestep spatially
        spatial_features = []
        
        for t in range(seq_len):
            timestep_input = input_sequence[:, t]  # [batch_size, num_nodes, features]
            
            # Process each sample in batch separately (GNN requirement)
            batch_spatial = []
            for b in range(batch_size):
                sample_features = timestep_input[b]  # [num_nodes, features]
                
                # Spatial encoding
                h = self.spatial_encoder(sample_features)
                
                # Spatial message passing
                for layer in self.spatial_layers:
                    h_new = layer(h, edge_index, edge_attr)
                    h = h + h_new
                
                batch_spatial.append(h)
            
            spatial_out = torch.stack(batch_spatial)  # [batch_size, num_nodes, hidden_dim]
            spatial_features.append(spatial_out)
        
        # Stack temporal features: [batch_size, seq_len, num_nodes, hidden_dim]
        temporal_input = torch.stack(spatial_features, dim=1)
        
        # Process temporal dimension node-by-node
        # Reshape to [batch_size * num_nodes, seq_len, hidden_dim]
        temp_input_reshaped = temporal_input.view(batch_size * num_nodes, seq_len, self.config.hidden_dim)
        
        # LSTM processing
        temporal_out, _ = self.temporal_processor(temp_input_reshaped)
        
        # Take final timestep output
        final_temporal = temporal_out[:, -1, :]  # [batch_size * num_nodes, hidden_dim]
        
        # Reshape back: [batch_size, num_nodes, hidden_dim]
        final_features = final_temporal.view(batch_size, num_nodes, self.config.hidden_dim)
        
        # Forecast decoding
        predictions = self.forecast_decoder(final_features)  # [batch_size, num_nodes, 3]
        
        if squeeze_output:
            predictions = predictions.squeeze(0)
        
        return predictions
