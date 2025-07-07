#!/usr/bin/env python3
"""
Mesh Connectivity Analyzer
Check for nonsensical connections in the icosahedral mesh
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
import warnings

sys.path.insert(0, str(Path.cwd()))

# Import your existing components
from mesh.icosahedral import IcosahedralMesh
from mesh.connectivity import compute_regional_edges
from config.base import MeshConfig, DataConfig

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth (in km)
    """
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def analyze_mesh_connectivity(mesh_config: MeshConfig, data_config: DataConfig):
    """
    Analyze mesh connectivity for potential wraparound issues
    """
    
    print("🔍 MESH CONNECTIVITY ANALYSIS")
    print("=" * 50)
    
    # Create mesh
    print("Building icosahedral mesh...")
    mesh = IcosahedralMesh(mesh_config)
    
    # Get regional indices
    print("Filtering regional nodes...")
    lat_bounds = data_config.lat_bounds
    lon_bounds = data_config.lon_bounds
    
    print(f"   Region bounds: {lat_bounds[0]:.1f}°N to {lat_bounds[1]:.1f}°N")
    print(f"                  {lon_bounds[0]:.1f}°E to {lon_bounds[1]:.1f}°E")
    
    regional_indices = mesh.filter_region(lat_bounds, lon_bounds)
    print(f"   Regional nodes: {len(regional_indices)}")
    
    # Get lat/lon for regional nodes
    lat, lon = mesh.vertices_to_lat_lon()
    regional_lat = lat[regional_indices]
    regional_lon = lon[regional_indices]
    
    # Create edges for regional nodes
    print("Computing regional edges...")
    edge_index, edge_attr = compute_regional_edges(mesh, regional_indices, mesh_config.max_edge_distance_km)
    
    print(f"   Regional edges: {edge_index.shape[1]}")
    
    # Analyze edges
    print("\n📊 EDGE ANALYSIS")
    print("-" * 30)
    
    # Calculate distances for all edges
    distances = []
    suspicious_edges = []
    longitude_jumps = []
    
    for i in range(edge_index.shape[1]):
        node1_idx = edge_index[0, i]
        node2_idx = edge_index[1, i]
        
        # Get original indices in the full mesh
        orig_idx1 = regional_indices[node1_idx]
        orig_idx2 = regional_indices[node2_idx]
        
        # Get lat/lon
        lat1, lon1 = lat[orig_idx1], lon[orig_idx1]
        lat2, lon2 = lat[orig_idx2], lon[orig_idx2]
        
        # Calculate distance
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        distances.append(distance)
        
        # Check for suspicious connections
        lon_diff = abs(lon2 - lon1)
        
        # Handle longitude wraparound (e.g., 350° to 10°)
        if lon_diff > 180:
            lon_diff = 360 - lon_diff
        
        longitude_jumps.append(lon_diff)
        
        # Flag suspicious edges
        if distance > mesh_config.max_edge_distance_km * 1.5:  # 50% over the limit
            suspicious_edges.append({
                'edge_idx': i,
                'node1': (lat1, lon1),
                'node2': (lat2, lon2),
                'distance': distance,
                'lon_jump': lon_diff
            })
    
    distances = np.array(distances)
    longitude_jumps = np.array(longitude_jumps)
    
    # Statistics
    print(f"📈 Edge Distance Statistics:")
    print(f"   Mean distance: {distances.mean():.1f} km")
    print(f"   Median distance: {np.median(distances):.1f} km")
    print(f"   Max distance: {distances.max():.1f} km")
    print(f"   Min distance: {distances.min():.1f} km")
    print(f"   Standard deviation: {distances.std():.1f} km")
    
    print(f"\n📍 Longitude Jump Statistics:")
    print(f"   Mean longitude jump: {longitude_jumps.mean():.1f}°")
    print(f"   Median longitude jump: {np.median(longitude_jumps):.1f}°")
    print(f"   Max longitude jump: {longitude_jumps.max():.1f}°")
    print(f"   >90° longitude jumps: {np.sum(longitude_jumps > 90)}")
    
    # Check for wraparound issues
    print(f"\n🚨 WRAPAROUND ISSUE DETECTION")
    print("-" * 40)
    
    # Issue 1: Excessive longitude jumps
    large_lon_jumps = np.sum(longitude_jumps > 90)
    if large_lon_jumps > 0:
        print(f"   ⚠️  {large_lon_jumps} edges have >90° longitude jumps")
        print(f"       This suggests potential wraparound issues!")
    else:
        print(f"   ✅ No excessive longitude jumps detected")
    
    # Issue 2: Distances much larger than expected
    very_long_edges = np.sum(distances > mesh_config.max_edge_distance_km * 2)
    if very_long_edges > 0:
        print(f"   ⚠️  {very_long_edges} edges are >2x the max distance")
        print(f"       This suggests mesh connectivity issues!")
    else:
        print(f"   ✅ No extremely long edges detected")
    
    # Issue 3: Suspicious edges
    if suspicious_edges:
        print(f"   ⚠️  {len(suspicious_edges)} suspicious edges found")
        print(f"       Top 5 most suspicious:")
        
        # Sort by distance
        suspicious_edges.sort(key=lambda x: x['distance'], reverse=True)
        
        for i, edge in enumerate(suspicious_edges[:5]):
            lat1, lon1 = edge['node1']
            lat2, lon2 = edge['node2']
            print(f"       {i+1}. ({lat1:.1f}°N, {lon1:.1f}°E) ↔ ({lat2:.1f}°N, {lon2:.1f}°E)")
            print(f"          Distance: {edge['distance']:.1f} km, Lon jump: {edge['lon_jump']:.1f}°")
    else:
        print(f"   ✅ No suspicious edges detected")
    
    # Check regional coverage
    print(f"\n🌍 REGIONAL COVERAGE CHECK")
    print("-" * 35)
    
    # Check if nodes are evenly distributed
    lon_span = lon_bounds[1] - lon_bounds[0]
    lat_span = lat_bounds[1] - lat_bounds[0]
    
    print(f"   Region span: {lat_span:.1f}° lat × {lon_span:.1f}° lon")
    print(f"   Node density: {len(regional_indices) / (lat_span * lon_span):.2f} nodes/deg²")
    
    # Check for edge clustering
    edge_lon_span = longitude_jumps.max() - longitude_jumps.min()
    edge_lat_span = np.ptp([edge['node1'][0] - edge['node2'][0] for edge in suspicious_edges] if suspicious_edges else [0])
    
    print(f"   Edge longitude span: {edge_lon_span:.1f}°")
    
    # Specific checks for Pacific region
    if lon_bounds[0] < 180 and lon_bounds[1] > 180:
        print(f"   📍 Pacific region detected (crosses 180° meridian)")
        
        # Check for connections across the date line
        dateline_crossings = 0
        for i in range(edge_index.shape[1]):
            node1_idx = edge_index[0, i]
            node2_idx = edge_index[1, i]
            
            orig_idx1 = regional_indices[node1_idx]
            orig_idx2 = regional_indices[node2_idx]
            
            lon1, lon2 = lon[orig_idx1], lon[orig_idx2]
            
            # Check if one node is near 180° and the other near 0°/360°
            if (lon1 > 170 and lon2 < 10) or (lon1 < 10 and lon2 > 170):
                dateline_crossings += 1
        
        if dateline_crossings > 0:
            print(f"   ⚠️  {dateline_crossings} edges cross the international date line")
            print(f"       This might indicate wraparound issues!")
        else:
            print(f"   ✅ No problematic date line crossings detected")
    
    return {
        'distances': distances,
        'longitude_jumps': longitude_jumps,
        'suspicious_edges': suspicious_edges,
        'regional_indices': regional_indices,
        'edge_index': edge_index,
        'regional_lat': regional_lat,
        'regional_lon': regional_lon
    }

def visualize_connectivity_issues(analysis_results: Dict):
    """
    Create visualizations of connectivity issues
    """
    
    print(f"\n📊 CREATING CONNECTIVITY VISUALIZATIONS")
    print("-" * 45)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distance distribution
    distances = analysis_results['distances']
    axes[0, 0].hist(distances, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Edge Distance (km)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Edge Distance Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add vertical line for max distance
    max_dist = distances.max()
    axes[0, 0].axvline(x=max_dist, color='red', linestyle='--', alpha=0.7, label=f'Max: {max_dist:.1f} km')
    axes[0, 0].legend()
    
    # 2. Longitude jump distribution
    longitude_jumps = analysis_results['longitude_jumps']
    axes[0, 1].hist(longitude_jumps, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Longitude Jump (degrees)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Longitude Jump Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add vertical line for 90° (suspicious threshold)
    axes[0, 1].axvline(x=90, color='red', linestyle='--', alpha=0.7, label='90° threshold')
    axes[0, 1].legend()
    
    # 3. Regional node distribution
    regional_lat = analysis_results['regional_lat']
    regional_lon = analysis_results['regional_lon']
    
    scatter = axes[1, 0].scatter(regional_lon, regional_lat, c=distances[:len(regional_lat)], 
                                cmap='viridis', alpha=0.6, s=1)
    axes[1, 0].set_xlabel('Longitude (degrees)')
    axes[1, 0].set_ylabel('Latitude (degrees)')
    axes[1, 0].set_title('Regional Node Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add colorbar
    plt.colorbar(scatter, ax=axes[1, 0], label='Distance (km)')
    
    # 4. Suspicious edges visualization
    suspicious_edges = analysis_results['suspicious_edges']
    
    if suspicious_edges:
        # Plot all nodes
        axes[1, 1].scatter(regional_lon, regional_lat, c='lightblue', alpha=0.5, s=1, label='All nodes')
        
        # Highlight suspicious connections
        for edge in suspicious_edges[:10]:  # Show top 10
            lat1, lon1 = edge['node1']
            lat2, lon2 = edge['node2']
            
            axes[1, 1].plot([lon1, lon2], [lat1, lat2], 'r-', alpha=0.7, linewidth=1)
            axes[1, 1].scatter([lon1, lon2], [lat1, lat2], c='red', s=20, zorder=5)
        
        axes[1, 1].set_xlabel('Longitude (degrees)')
        axes[1, 1].set_ylabel('Latitude (degrees)')
        axes[1, 1].set_title(f'Suspicious Edges (Top {min(10, len(suspicious_edges))})')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'No suspicious edges detected!', 
                       transform=axes[1, 1].transAxes, ha='center', va='center',
                       fontsize=14, bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.8))
        axes[1, 1].set_title('Suspicious Edges')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path("mesh_connectivity_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Visualizations saved to: {plot_path}")
    
    plt.show()

def main():
    """
    Main function to run connectivity analysis
    """
    
    print("🌊 MESH CONNECTIVITY ANALYZER")
    print("=" * 50)
    
    # Configuration
    mesh_config = MeshConfig(refinement_level=5)  # Use level 5 for faster testing
    data_config = DataConfig()
    
    print(f"Configuration:")
    print(f"   Mesh refinement level: {mesh_config.refinement_level}")
    print(f"   Max edge distance: {mesh_config.max_edge_distance_km} km")
    print(f"   Data region: {data_config.lat_bounds}, {data_config.lon_bounds}")
    
    # Run analysis
    try:
        analysis_results = analyze_mesh_connectivity(mesh_config, data_config)
        
        # Create visualizations
        visualize_connectivity_issues(analysis_results)
        
        # Summary recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 25)
        
        suspicious_count = len(analysis_results['suspicious_edges'])
        max_distance = analysis_results['distances'].max()
        max_lon_jump = analysis_results['longitude_jumps'].max()
        
        if suspicious_count > 0:
            print(f"   ⚠️  {suspicious_count} suspicious edges detected")
            print(f"   🔧 Consider implementing regional distance constraints")
            print(f"   🔧 Add longitude wraparound protection")
        
        if max_distance > mesh_config.max_edge_distance_km * 2:
            print(f"   ⚠️  Some edges are {max_distance:.1f} km (very long)")
            print(f"   🔧 Consider reducing max_edge_distance_km parameter")
        
        if max_lon_jump > 90:
            print(f"   ⚠️  Maximum longitude jump is {max_lon_jump:.1f}°")
            print(f"   🔧 This suggests potential wraparound issues")
        
        if suspicious_count == 0 and max_distance < mesh_config.max_edge_distance_km * 1.5:
            print(f"   ✅ Mesh connectivity looks reasonable!")
            print(f"   ✅ No major wraparound issues detected")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()