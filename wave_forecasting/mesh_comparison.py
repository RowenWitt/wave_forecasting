#!/usr/bin/env python3
"""
Compare mesh structures between North Pacific and Global models
"""

import pickle
import numpy as np
from pathlib import Path

def analyze_mesh_cache(cache_file_path: str):
    """Analyze contents of a mesh cache file"""
    
    print(f"\n📊 Analyzing: {Path(cache_file_path).name}")
    print("=" * 60)
    
    # Get file size
    file_size = Path(cache_file_path).stat().st_size / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")
    
    # Load cache
    with open(cache_file_path, 'rb') as f:
        mesh_data = pickle.load(f)
    
    print(f"\nCache contents:")
    for key in mesh_data.keys():
        print(f"  - {key}")
    
    # Analyze each component
    if 'vertices' in mesh_data:
        vertices = mesh_data['vertices']
        print(f"\nVertices:")
        print(f"  Shape: {vertices.shape}")
        print(f"  Dtype: {vertices.dtype}")
        print(f"  Size: {vertices.nbytes / (1024*1024):.2f} MB")
        print(f"  Range: [{vertices.min():.3f}, {vertices.max():.3f}]")
    
    if 'faces' in mesh_data:
        faces = mesh_data['faces']
        print(f"\nFaces:")
        print(f"  Shape: {faces.shape}")
        print(f"  Dtype: {faces.dtype}")
        print(f"  Size: {faces.nbytes / (1024*1024):.2f} MB")
        print(f"  Max vertex index: {faces.max()}")
    
    if 'edges' in mesh_data:
        edges = mesh_data['edges']
        print(f"\nEdges:")
        print(f"  Shape: {edges.shape}")
        print(f"  Dtype: {edges.dtype}")
        print(f"  Size: {edges.nbytes / (1024*1024):.2f} MB")
    
    # Check for extra components
    if 'multiscale_edges' in mesh_data:
        me = mesh_data['multiscale_edges']
        print(f"\nMultiscale edges:")
        for scale, edges in me.items():
            print(f"  {scale}: {edges.shape} ({edges.nbytes / (1024*1024):.2f} MB)")
    
    # Check for edge attributes
    if 'edge_attr' in mesh_data:
        edge_attr = mesh_data['edge_attr']
        print(f"\nEdge attributes:")
        print(f"  Shape: {edge_attr.shape}")
        print(f"  Dtype: {edge_attr.dtype}")
        print(f"  Size: {edge_attr.nbytes / (1024*1024):.2f} MB")
    
    # Total size breakdown
    total_size = 0
    print(f"\n💾 Size breakdown:")
    for key, value in mesh_data.items():
        if hasattr(value, 'nbytes'):
            size_mb = value.nbytes / (1024*1024)
            total_size += size_mb
            print(f"  {key}: {size_mb:.2f} MB")
        elif isinstance(value, dict):
            dict_size = sum(v.nbytes for v in value.values() if hasattr(v, 'nbytes'))
            total_size += dict_size / (1024*1024)
            print(f"  {key}: {dict_size / (1024*1024):.2f} MB")
    
    print(f"\nTotal data size: {total_size:.2f} MB")
    print(f"Pickle overhead: {file_size - total_size:.2f} MB")


def create_test_meshes():
    """Create test meshes to compare structure"""
    
    # Try to load both the North Pacific approach
    print("\n🔧 Testing mesh creation approaches...")
    
    # Approach 1: Current global model
    from global_wave_model_v1 import GlobalIcosahedralMesh
    
    global_mesh = GlobalIcosahedralMesh(refinement_level=5, cache_dir="test_cache/global")
    print(f"\nGlobal mesh (level 5):")
    print(f"  Vertices: {len(global_mesh.vertices)}")
    print(f"  Edges: {len(global_mesh.edges)}")
    
    # Check if North Pacific mesh module exists
    try:
        from mesh.icosahedral import IcosahedralMesh
        from config.base import MeshConfig
        
        config = MeshConfig(refinement_level=5)
        np_mesh = IcosahedralMesh(config, cache_dir="test_cache/north_pacific")
        
        print(f"\nNorth Pacific mesh (level 5):")
        print(f"  Vertices: {len(np_mesh.vertices)}")
        print(f"  Edges: {len(np_mesh.edges)}")
        print(f"  Has multiscale_edges: {'multiscale_edges' in np_mesh.__dict__}")
        
    except ImportError:
        print("\n⚠️  Cannot import North Pacific mesh module")


def compare_mesh_approaches():
    """Compare the two mesh building approaches"""
    
    print("\n📋 Key differences in mesh approaches:")
    print("\n1. NORTH PACIFIC approach likely includes:")
    print("   - Multiscale edge connectivity (local, medium, long-range)")
    print("   - Pre-computed edge attributes (distances, directions)")
    print("   - Regional edge filtering")
    print("   - Additional data structures for message passing")
    
    print("\n2. GLOBAL approach currently includes:")
    print("   - Basic vertices, faces, edges")
    print("   - Minimal edge connectivity")
    print("   - No multiscale edges")
    print("   - Edge attributes computed separately")
    
    print("\n💡 This explains the size difference!")
    print("   North Pacific: Comprehensive pre-computed connectivity")
    print("   Global: Minimal mesh structure")


def recommend_global_mesh_update():
    """Recommend updates to global mesh"""
    
    print("\n🚀 Recommended Global Mesh Updates:")
    print("""
    1. Add multiscale edge connectivity:
       - Local edges (nearest neighbors)
       - Medium edges (regional connections)
       - Long edges (teleconnections)
    
    2. Pre-compute edge attributes:
       - Great circle distances
       - Directional components
       - Regional identifiers
    
    3. Cache comprehensive connectivity:
       - All edge types
       - Attributes
       - Regional masks
    
    This will increase cache size but improve:
    - Training efficiency (no recomputation)
    - Model expressiveness (multiscale connections)
    - Physical realism (proper teleconnections)
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare mesh structures')
    parser.add_argument('--np_cache', type=str, help='Path to North Pacific mesh cache')
    parser.add_argument('--global_cache', type=str, help='Path to Global mesh cache')
    parser.add_argument('--test', action='store_true', help='Create test meshes')
    
    args = parser.parse_args()
    
    if args.np_cache:
        analyze_mesh_cache(args.np_cache)
    
    if args.global_cache:
        analyze_mesh_cache(args.global_cache)
    
    if args.test:
        create_test_meshes()
    
    # Always show comparison
    compare_mesh_approaches()
    recommend_global_mesh_update()