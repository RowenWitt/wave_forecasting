import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import MessagePassing
from build_mesh import IcosahedralMesh
from load_transform_interpolate import load_and_inspect_datasets, process_gebco_for_waves, MultiResolutionInterpolator, MeshDataLoader


class WaveMessageLayer(MessagePassing):
    """Single message passing layer for wave propagation"""
    
    def __init__(self, node_features, edge_features, hidden_dim=64):
        super().__init__(aggr='add')  # Sum messages from neighbors
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        
        # Message function: combines sender, receiver, edge info
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_features + edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function: combines node state with messages
        self.update_mlp = nn.Sequential(
            nn.Linear(node_features + hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, node_features)
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        x: [num_nodes, node_features] - current node states
        edge_index: [2, num_edges] - connectivity 
        edge_attr: [num_edges, edge_features] - edge properties
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        """
        Compute messages from sender (x_j) to receiver (x_i)
        """
        # Concatenate sender, receiver, edge info
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)
    
    def update(self, aggr_out, x):
        """
        Update node features with aggregated messages
        """
        # Combine original features with messages
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)

class SimpleWaveGNN(nn.Module):
    """Complete wave prediction model"""
    
    def __init__(self, input_features, hidden_dim=64, num_layers=8):
        super().__init__()
        
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder: map input features to hidden representation
        self.encoder = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            WaveMessageLayer(hidden_dim, edge_features=3, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Decoder: map back to wave predictions
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # [swh, mwd, mwp] predictions
        )
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass through wave GNN
        """
        # Encode input features
        h = self.encoder(x)
        
        # Message passing with residual connections
        for layer in self.message_layers:
            h_new = layer(h, edge_index, edge_attr)
            h = h + h_new  # Residual connection
        
        # Decode to predictions
        predictions = self.decoder(h)
        
        return predictions

# Quick test setup
# mesh = IcosahedralMesh(refinement_level=4)  # Start with level 4 for POC
# mesh_loader = MeshDataLoader(mesh, interpolator)
# n_features = len(test_features['feature_names'])
# model = SimpleWaveGNN(input_features=n_features, hidden_dim=64, num_layers=4)

# print(f"✅ Model created:")
# print(f"  Input features: {n_features}")
# print(f"  Hidden dim: 64")
# print(f"  Message layers: 4")
# print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

def compute_edge_features(mesh, pacific_node_indices):
    """Compute edge features for message passing - FIXED for edge_index shape"""
    
    # Get Pacific node coordinates
    mesh_lats, mesh_lons = mesh.vertices_to_lat_lon()
    pacific_lats = mesh_lats[pacific_node_indices]
    pacific_lons = mesh_lons[pacific_node_indices]
    
    # Build edge connectivity within Pacific nodes
    edges = []
    edge_features = []
    
    n_pacific = len(pacific_node_indices)
    print(f"Computing edges for {n_pacific} Pacific nodes...")
    
    for i in range(n_pacific):
        for j in range(i+1, n_pacific):
            # Compute great circle distance
            lat1, lon1 = pacific_lats[i], pacific_lons[i]
            lat2, lon2 = pacific_lats[j], pacific_lons[j]
            
            distance = great_circle_distance(lat1, lon1, lat2, lon2)
            
            # Only connect nearby nodes (< 300km for POC)
            if distance < 300:  # km
                edges.append([i, j])
                edges.append([j, i])  # Bidirectional
                
                # Edge features: [distance, lat_diff, lon_diff]
                edge_feat = [distance/1000.0, (lat2-lat1)/10.0, (lon2-lon1)/10.0]
                edge_features.extend([edge_feat, edge_feat])
    
    # Handle case with no edges found
    if len(edges) == 0:
        print("⚠️  No edges found! Creating minimal connectivity...")
        # Connect each node to itself (self-loops) as fallback
        for i in range(min(n_pacific, 5)):  # Just connect first 5 nodes to themselves
            edges.append([i, i])
            edge_features.append([0.0, 0.0, 0.0])
    
    # CRITICAL FIX: Ensure proper 2D shape
    if len(edges) > 0:
        edge_array = np.array(edges)
        print(f"Edge array shape before transpose: {edge_array.shape}")
        
        # Make sure we have [num_edges, 2] shape
        if edge_array.ndim == 1:
            edge_array = edge_array.reshape(-1, 2)
        
        # Transpose to [2, num_edges] as required by PyTorch Geometric
        edge_index = torch.tensor(edge_array.T, dtype=torch.long)
    else:
        # Empty graph fallback - create minimal valid structure
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    
    print(f"✅ Edge computation complete:")
    print(f"  Edges found: {len(edges)}")
    print(f"  Edge index shape: {edge_index.shape}")  # Should be [2, num_edges]
    print(f"  Edge attr shape: {edge_attr.shape}")
    print(f"  Edge index range: {edge_index.min().item() if edge_index.numel() > 0 else 'N/A'} to {edge_index.max().item() if edge_index.numel() > 0 else 'N/A'}")
    
    return edge_index, edge_attr

def great_circle_distance(lat1, lon1, lat2, lon2):
    """Compute distance between points on sphere (km)"""
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

# # Compute edges for Pacific region
# mesh = IcosahedralMesh(refinement_level=4)  # Start with level 4 for POC
# test_features = mesh_loader.load_mesh_features(time_idx=0)
# edge_index, edge_attr = compute_edge_features(mesh, test_features['node_indices'])

def diagnose_nan_issues(features, edge_index, edge_attr, test_features):
    """Find the source of NaN values"""
    
    print("🔍 NaN Diagnosis:")
    print(f"  Input features have NaN: {torch.isnan(features).any().item()}")
    print(f"  Edge attr has NaN: {torch.isnan(edge_attr).any().item()}")
    
    if torch.isnan(features).any():
        nan_mask = torch.isnan(features).any(dim=1)
        print(f"    Nodes with NaN features: {nan_mask.sum().item()}/{len(features)}")
        
        # Check which features have NaN
        for i, feat_name in enumerate(test_features['feature_names']):
            nan_count = torch.isnan(features[:, i]).sum().item()
            if nan_count > 0:
                print(f"    Feature '{feat_name}' has {nan_count} NaN values")
    
    if torch.isnan(edge_attr).any():
        nan_edges = torch.isnan(edge_attr).any(dim=1).sum().item()
        print(f"    Edges with NaN attr: {nan_edges}/{len(edge_attr)}")

# Add this right before the forward pass

def clean_features_for_training(features):
    """Clean NaN values and normalize features"""
    
    # Replace NaN with zeros (better: replace with meaningful defaults)
    features_clean = torch.where(torch.isnan(features), torch.zeros_like(features), features)
    
    # Handle infinite values
    features_clean = torch.where(torch.isinf(features_clean), torch.zeros_like(features_clean), features_clean)
    
    # Basic normalization to prevent exploding gradients
    # Clamp extreme values
    features_clean = torch.clamp(features_clean, -10.0, 10.0)
    
    # Simple standardization (mean=0, std=1) per feature
    for i in range(features_clean.shape[1]):
        col = features_clean[:, i]
        if col.std() > 1e-6:  # Avoid division by zero
            features_clean[:, i] = (col - col.mean()) / col.std()
    
    return features_clean

# Use cleaned features
# features_clean = clean_features_for_training(features)

# Test the complete pipeline
def test_complete_pipeline():
    """Test data loading -> mesh -> model -> predictions"""
    era5_atmo, era5_waves, gebco = load_and_inspect_datasets()
    processed_gebco = process_gebco_for_waves(gebco)
    interpolator = MultiResolutionInterpolator(era5_atmo, era5_waves, processed_gebco)
    mesh = IcosahedralMesh(refinement_level=4)  # Start with level 4 for POC
    mesh_loader = MeshDataLoader(mesh, interpolator)
    test_features = mesh_loader.load_mesh_features(time_idx=0)
    n_features = len(test_features['feature_names'])
    edge_index, edge_attr = compute_edge_features(mesh, test_features['node_indices'])
    model = SimpleWaveGNN(input_features=n_features, hidden_dim=64, num_layers=4)

    print("Testing complete pipeline...")
    
    # Load features onto mesh
    mesh_data = mesh_loader.load_mesh_features(time_idx=0)
    features = torch.tensor(mesh_data['features'], dtype=torch.float32)
    
    print(f"✅ Features loaded: {features.shape}")
    
    # Forward pass through model
    diagnose_nan_issues(features, edge_index, edge_attr, test_features)
    features_clean = clean_features_for_training(features)
    print(f"✅ Features cleaned. NaN check: {torch.isnan(features_clean).any().item()}")
    with torch.no_grad():
        predictions = model(features_clean, edge_index, edge_attr)
    
    print(f"✅ Forward pass complete: {predictions.shape}")
    print(f"Prediction ranges:")
    print(f"  SWH: {predictions[:, 0].min():.3f} to {predictions[:, 0].max():.3f}")
    print(f"  MWD: {predictions[:, 1].min():.3f} to {predictions[:, 1].max():.3f}")
    print(f"  MWP: {predictions[:, 2].min():.3f} to {predictions[:, 2].max():.3f}")
    
    return predictions

# Test it!
# test_predictions = test_complete_pipeline()

import torch
import numpy as np

def train_wave_model_fixed(model, train_loader, criterion, num_epochs=20, lr=0.001):
    """Fixed training loop that handles batched inputs properly for GNN"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    print(f"🚀 Starting training: {num_epochs} epochs, lr={lr}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        epoch_losses = {'total': [], 'mse': [], 'physics': []}
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Get batch data
            features = batch['features']  # [batch_size, num_nodes, num_features]
            targets = batch['targets']    # [batch_size, num_nodes, 3]
            
            # Get edge data from dataset (shared across all samples)
            edge_index = train_loader.dataset.edge_index  # [2, num_edges]
            edge_attr = train_loader.dataset.edge_attr    # [num_edges, edge_features]
            
            # DEBUG: Print shapes
            if batch_idx == 0 and epoch == 0:
                print(f"🔍 Debug shapes:")
                print(f"  Features: {features.shape}")
                print(f"  Targets: {targets.shape}")
                print(f"  Edge index: {edge_index.shape}")
                print(f"  Edge attr: {edge_attr.shape}")
            
            # Process each sample in the batch individually
            # because GNN layers expect 2D node features, not 3D batched features
            batch_predictions = []
            batch_size = features.shape[0]
            
            for i in range(batch_size):
                # Extract single sample (remove batch dimension)
                sample_features = features[i]  # [num_nodes, num_features]
                
                # Forward pass through GNN
                sample_pred = model(sample_features, edge_index, edge_attr)  # [num_nodes, 3]
                batch_predictions.append(sample_pred)
            
            # Stack predictions back into batch format
            predictions = torch.stack(batch_predictions)  # [batch_size, num_nodes, 3]
            
            # Compute loss
            loss_dict = criterion(predictions, targets)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track losses
            epoch_losses['total'].append(loss_dict['total_loss'].item())
            epoch_losses['mse'].append(loss_dict['mse_loss'].item())
            epoch_losses['physics'].append(loss_dict['physics_loss'].item())
            
            # Debug first batch
            if batch_idx == 0 and epoch == 0:
                print(f"✅ First batch processed successfully!")
                print(f"  Prediction shape: {predictions.shape}")
                print(f"  Loss: {loss_dict['total_loss'].item():.4f}")
        
        # Print epoch summary
        avg_total = np.mean(epoch_losses['total'])
        avg_mse = np.mean(epoch_losses['mse'])
        avg_physics = np.mean(epoch_losses['physics'])
        
        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"Total={avg_total:.4f}, MSE={avg_mse:.4f}, Physics={avg_physics:.4f}")
        
        # Learning rate decay
        if (epoch + 1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"   Reduced LR to {optimizer.param_groups[0]['lr']:.6f}")
    
    print("✅ Training complete!")
    return model

# Alternative: Modify your model to handle batched inputs natively
class BatchFriendlyWaveGNN(torch.nn.Module):
    """Modified GNN that can handle batched inputs directly"""
    
    def __init__(self, input_features, hidden_dim=64, num_layers=4):
        super().__init__()
        
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Replace message passing with transformer layers that handle batches naturally
        self.spatial_layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim*2,
                dropout=0.1,
                batch_first=True  # Important!
            )
            for _ in range(num_layers)
        ])
        
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 3)  # [swh, mwd, mwp]
        )
    
    def forward(self, x, edge_index=None, edge_attr=None):
        """
        Forward pass that can handle both batched and unbatched inputs
        x: [batch_size, num_nodes, input_features] OR [num_nodes, input_features]
        """
        # Handle both batched and unbatched inputs
        if x.dim() == 2:
            # Unbatched: [num_nodes, features] -> add batch dim
            x = x.unsqueeze(0)  # [1, num_nodes, features]
            squeeze_output = True
        else:
            # Already batched: [batch_size, num_nodes, features]
            squeeze_output = False
        
        batch_size, num_nodes, _ = x.shape
        
        # Encode
        h = self.encoder(x)  # [batch_size, num_nodes, hidden_dim]
        
        # Apply spatial reasoning layers
        for layer in self.spatial_layers:
            h = layer(h)  # Self-attention over spatial nodes
        
        # Decode
        predictions = self.decoder(h)  # [batch_size, num_nodes, 3]
        
        # Remove batch dimension if input was unbatched
        if squeeze_output:
            predictions = predictions.squeeze(0)  # [num_nodes, 3]
        
        return predictions

# Simplified training function for batch-friendly model
def train_batch_friendly_model(model, train_loader, criterion, num_epochs=20, lr=0.001):
    """Simplified training for models that handle batches natively"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    print(f"🚀 Starting training: {num_epochs} epochs, lr={lr}")
    
    for epoch in range(num_epochs):
        epoch_losses = {'total': [], 'mse': [], 'physics': []}
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            features = batch['features']  # [batch_size, num_nodes, num_features]
            targets = batch['targets']    # [batch_size, num_nodes, 3]
            
            # Model handles batched inputs directly
            predictions = model(features)  # [batch_size, num_nodes, 3]
            
            # Compute loss
            loss_dict = criterion(predictions, targets)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track losses
            epoch_losses['total'].append(loss_dict['total_loss'].item())
            epoch_losses['mse'].append(loss_dict['mse_loss'].item())
            epoch_losses['physics'].append(loss_dict['physics_loss'].item())
        
        # Print epoch summary
        avg_total = np.mean(epoch_losses['total'])
        avg_mse = np.mean(epoch_losses['mse'])
        avg_physics = np.mean(epoch_losses['physics'])
        
        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"Total={avg_total:.4f}, MSE={avg_mse:.4f}, Physics={avg_physics:.4f}")
    
    return model

# Usage instructions:
# """
# OPTION 1 (Keep your current GNN model):
# Replace your train_wave_model call with:
# trained_model = train_wave_model_fixed(model, train_loader, criterion, num_epochs=20, lr=0.001)

# OPTION 2 (Use batch-friendly model):
# Replace your model creation with:
# model = BatchFriendlyWaveGNN(input_features=n_features, hidden_dim=64, num_layers=4)
# trained_model = train_batch_friendly_model(model, train_loader, criterion, num_epochs=20, lr=0.001)
# """