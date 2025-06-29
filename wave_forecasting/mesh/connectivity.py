# mesh/connectivity.py
"""Edge computation and graph connectivity"""
import torch
import numpy as np
from typing import Tuple

from mesh.icosahedral import IcosahedralMesh


def great_circle_distance(lat1, lon1, lat2, lon2):
    """Compute distance between points on sphere (km)"""
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def compute_regional_edges(mesh: IcosahedralMesh, node_indices: np.ndarray, 
                          max_distance_km: float = 300.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute edge connectivity for a subset of mesh nodes"""
    
    # Get coordinates for selected nodes
    mesh_lats, mesh_lons = mesh.vertices_to_lat_lon()
    region_lats = mesh_lats[node_indices]
    region_lons = mesh_lons[node_indices]
    
    # Build edge connectivity
    edges = []
    edge_features = []
    
    n_nodes = len(node_indices)
    print(f"Computing edges for {n_nodes} nodes...")
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            lat1, lon1 = region_lats[i], region_lons[i]
            lat2, lon2 = region_lats[j], region_lons[j]
            
            distance = great_circle_distance(lat1, lon1, lat2, lon2)
            
            if distance < max_distance_km:
                # Bidirectional edges
                edges.extend([[i, j], [j, i]])
                
                # Edge features: [distance, lat_diff, lon_diff]
                edge_feat = [distance/1000.0, (lat2-lat1)/10.0, (lon2-lon1)/10.0]
                edge_features.extend([edge_feat, edge_feat])
    
    # Handle edge case with no connections
    if len(edges) == 0:
        print("⚠️  No edges found! Creating self-loops...")
        for i in range(min(n_nodes, 5)):
            edges.append([i, i])
            edge_features.append([0.0, 0.0, 0.0])
    
    # Convert to PyTorch tensors
    edge_array = np.array(edges)
    edge_index = torch.tensor(edge_array.T, dtype=torch.long)  # [2, num_edges]
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    
    print(f"✅ Edge computation complete: {edge_index.shape[1]} edges")
    return edge_index, edge_attr
