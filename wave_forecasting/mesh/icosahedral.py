# mesh/icosahedral.py
"""Clean icosahedral mesh implementation with caching"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import hashlib
import os
from typing import Tuple, Dict
from pathlib import Path

from config.base import MeshConfig

class IcosahedralMesh:
    """Create and manage icosahedral mesh for wave modeling with caching support"""
    
    def __init__(self, config: MeshConfig, cache_dir: str = "mesh_cache", use_cache: bool = True):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.vertices = None
        self.faces = None
        self.edges = None
        self.multiscale_edges = None
        
        # Create cache directory
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
        
        self._build_or_load_mesh()
    
    def _get_cache_key(self) -> str:
        """Generate unique cache key based on mesh configuration"""
        # Create hash from configuration parameters
        config_str = f"refinement_{self.config.refinement_level}_maxdist_{self.config.max_edge_distance_km}"
        cache_key = hashlib.md5(config_str.encode()).hexdigest()[:12]
        return f"icosahedral_mesh_{cache_key}.pkl"
    
    def _cache_exists(self) -> bool:
        """Check if cached mesh exists"""
        cache_file = self.cache_dir / self._get_cache_key()
        return cache_file.exists()
    
    def _save_mesh_to_cache(self):
        """Save current mesh to cache"""
        if not self.use_cache:
            return
            
        cache_file = self.cache_dir / self._get_cache_key()
        
        mesh_data = {
            'vertices': self.vertices,
            'faces': self.faces,
            'edges': self.edges,
            'multiscale_edges': self.multiscale_edges,
            'config': self.config,
            'version': '1.0'  # For future compatibility
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(mesh_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"💾 Mesh cached to: {cache_file}")
        except Exception as e:
            print(f"⚠️  Failed to cache mesh: {e}")
    
    def _load_mesh_from_cache(self) -> bool:
        """Load mesh from cache. Returns True if successful."""
        if not self.use_cache or not self._cache_exists():
            return False
        
        cache_file = self.cache_dir / self._get_cache_key()
        
        try:
            print(f"📂 Loading cached mesh from: {cache_file}")
            
            with open(cache_file, 'rb') as f:
                mesh_data = pickle.load(f)
            
            # Verify cache is compatible
            if mesh_data.get('version') != '1.0':
                print(f"⚠️  Cache version mismatch, rebuilding...")
                return False
            
            # Load mesh components
            self.vertices = mesh_data['vertices']
            self.faces = mesh_data['faces']
            self.edges = mesh_data['edges']
            self.multiscale_edges = mesh_data['multiscale_edges']
            
            print(f"✅ Mesh loaded from cache: {len(self.vertices)} vertices, {len(self.edges)} edges")
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to load cached mesh: {e}")
            print(f"    Rebuilding mesh from scratch...")
            return False
    
    def _build_or_load_mesh(self):
        """Build mesh or load from cache"""
        
        # Try loading from cache first
        if self._load_mesh_from_cache():
            return
        
        # Cache miss or disabled - build mesh
        print(f"🔨 Building icosahedral mesh (level {self.config.refinement_level})...")
        self._build_mesh()
        
        # Save to cache
        if self.use_cache:
            self._save_mesh_to_cache()
    
    def _build_mesh(self):
        """Build the icosahedral mesh"""
        
        # Start with basic icosahedron
        vertices, faces = self._create_base_icosahedron()
        
        # Refine iteratively
        for level in range(self.config.refinement_level):
            vertices, faces = self._subdivide_mesh(vertices, faces)
            print(f"  Level {level+1}: {len(vertices)} vertices, {len(faces)} faces")
        
        # Project to unit sphere and create connectivity
        self.vertices = self._normalize_to_sphere(vertices)
        self.faces = faces
        self.edges = self._create_edges_from_faces(faces)
        self.multiscale_edges = self._create_multiscale_edges(self.vertices, faces)
        
        print(f"✅ Mesh complete: {len(self.vertices)} vertices, {len(self.edges)} edges")
    
    def _create_base_icosahedron(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create the basic 12-vertex icosahedron"""
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        vertices = np.array([
            [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
        ], dtype=float)
        
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ])
        
        return vertices, faces
    
    def _subdivide_mesh(self, vertices, faces):
        """Subdivide each triangle into 4 smaller triangles"""
        new_vertices = list(vertices)
        new_faces = []
        edge_midpoints = {}
        
        def get_midpoint(v1_idx, v2_idx):
            edge = tuple(sorted([v1_idx, v2_idx]))
            if edge not in edge_midpoints:
                midpoint = (vertices[v1_idx] + vertices[v2_idx]) / 2
                edge_midpoints[edge] = len(new_vertices)
                new_vertices.append(midpoint)
            return edge_midpoints[edge]
        
        for face in faces:
            v1, v2, v3 = face
            m12 = get_midpoint(v1, v2)
            m23 = get_midpoint(v2, v3)
            m31 = get_midpoint(v3, v1)
            
            new_faces.extend([
                [v1, m12, m31], [v2, m23, m12],
                [v3, m31, m23], [m12, m23, m31]
            ])
        
        return np.array(new_vertices), np.array(new_faces)
    
    def _normalize_to_sphere(self, vertices):
        """Project vertices onto unit sphere"""
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        return vertices / norms
    
    def _create_edges_from_faces(self, faces):
        """Extract unique edges from faces"""
        edges = set()
        for face in faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i+1)%3]]))
                edges.add(edge)
        return np.array(list(edges))
    
    def _create_multiscale_edges(self, vertices, faces):
        """Create different types of edges for different physics scales"""
        local_edges = self._create_edges_from_faces(faces)
        medium_edges = self._create_extended_edges(vertices, max_distance=0.3)
        long_edges = self._create_extended_edges(vertices, max_distance=1.0, min_distance=0.5)
        
        return {
            'local': local_edges,
            'medium': medium_edges, 
            'long': long_edges
        }
    
    def _create_extended_edges(self, vertices, max_distance, min_distance=0.0):
        """Create edges between vertices within distance range"""
        edges = []
        n_vertices = len(vertices)
        
        for i in range(n_vertices):
            for j in range(i+1, n_vertices):
                dot_product = np.clip(np.dot(vertices[i], vertices[j]), -1, 1)
                distance = np.arccos(dot_product)
                
                if min_distance <= distance <= max_distance:
                    edges.append([i, j])
        
        return np.array(edges)
    
    def vertices_to_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert 3D vertices to lat/lon coordinates"""
        x, y, z = self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2]
        
        lat = np.arcsin(z) * 180 / np.pi
        lon = np.arctan2(y, x) * 180 / np.pi
        lon = np.where(lon < 0, lon + 360, lon)  # Ensure [0, 360)
        
        return lat, lon
    
    def filter_region(self, lat_bounds: Tuple[float, float], 
                     lon_bounds: Tuple[float, float]) -> np.ndarray:
        """Get node indices within geographic bounds"""
        lat, lon = self.vertices_to_lat_lon()
        mask = ((lat >= lat_bounds[0]) & (lat <= lat_bounds[1]) & 
                (lon >= lon_bounds[0]) & (lon <= lon_bounds[1]))
        return np.where(mask)[0]
    
    def clear_cache(self):
        """Clear mesh cache"""
        cache_file = self.cache_dir / self._get_cache_key()
        if cache_file.exists():
            cache_file.unlink()
            print(f"🗑️  Cleared mesh cache: {cache_file}")
    
    @classmethod
    def clear_all_cache(cls, cache_dir: str = "mesh_cache"):
        """Clear all cached meshes"""
        cache_path = Path(cache_dir)
        if cache_path.exists():
            for cache_file in cache_path.glob("icosahedral_mesh_*.pkl"):
                cache_file.unlink()
                print(f"🗑️  Cleared: {cache_file}")
    
    def get_cache_info(self) -> Dict[str, any]:
        """Get information about mesh cache"""
        cache_file = self.cache_dir / self._get_cache_key()
        
        info = {
            'cache_enabled': self.use_cache,
            'cache_file': str(cache_file),
            'cache_exists': cache_file.exists(),
            'cache_key': self._get_cache_key()
        }
        
        if cache_file.exists():
            info['cache_size_mb'] = cache_file.stat().st_size / (1024 * 1024)
            info['cache_modified'] = cache_file.stat().st_mtime
        
        return info

# Utility functions for mesh management
def create_mesh_with_caching(config: MeshConfig, use_cache: bool = True, cache_dir: str = "mesh_cache") -> IcosahedralMesh:
    """
    Convenience function to create mesh with caching enabled by default
    
    Args:
        config: Mesh configuration
        use_cache: Whether to use caching (default: True)
        cache_dir: Cache directory (default: "mesh_cache")
    
    Returns:
        IcosahedralMesh instance
    """
    return IcosahedralMesh(config, cache_dir=cache_dir, use_cache=use_cache)

def precompute_mesh_cache(refinement_levels: list = [4, 5, 6], cache_dir: str = "mesh_cache"):
    """
    Precompute and cache meshes for different refinement levels
    
    Args:
        refinement_levels: List of refinement levels to precompute
        cache_dir: Cache directory
    """
    print("🔄 Precomputing mesh cache...")
    
    for level in refinement_levels:
        print(f"\n📐 Precomputing refinement level {level}...")
        config = MeshConfig(refinement_level=level)
        mesh = IcosahedralMesh(config, cache_dir=cache_dir, use_cache=True)
        
        cache_info = mesh.get_cache_info()
        print(f"   ✅ Cached: {cache_info['cache_size_mb']:.2f} MB")
    
    print(f"\n🎉 Mesh cache precomputation complete!")

def show_cache_status(cache_dir: str = "mesh_cache"):
    """Show status of mesh cache"""
    cache_path = Path(cache_dir)
    
    print(f"📁 MESH CACHE STATUS")
    print(f"   Directory: {cache_path.absolute()}")
    print(f"   Exists: {cache_path.exists()}")
    
    if cache_path.exists():
        cache_files = list(cache_path.glob("icosahedral_mesh_*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        print(f"   Files: {len(cache_files)}")
        print(f"   Total size: {total_size / (1024 * 1024):.2f} MB")
        
        if cache_files:
            print(f"\n   📋 Cache files:")
            for cache_file in cache_files:
                size_mb = cache_file.stat().st_size / (1024 * 1024)
                print(f"      {cache_file.name}: {size_mb:.2f} MB")
    else:
        print(f"   No cache directory found")

# if __name__ == "__main__":
#     # Example usage and testing
#     print("🧪 Testing cached icosahedral mesh...")
    
#     # Show initial cache status
#     show_cache_status()
    
#     # Test mesh creation with caching
#     config = MeshConfig(refinement_level=4)  # Start with smaller mesh for testing
    
#     print(f"\n🔄 First mesh creation (should build and cache)...")
#     import time
#     start_time = time.time()
#     mesh1 = create_mesh_with_caching(config)
#     first_time = time.time() - start_time
#     print(f"   Time: {first_time:.2f} seconds")
    
#     print(f"\n🔄 Second mesh creation (should load from cache)...")
#     start_time = time.time()
#     mesh2 = create_mesh_with_caching(config)
#     second_time = time.time() - start_time
#     print(f"   Time: {second_time:.2f} seconds")
    
#     print(f"\n📊 Performance improvement: {first_time/second_time:.1f}x faster")
    
#     # Show cache info
#     cache_info = mesh1.get_cache_info()
#     print(f"\n📁 Cache info:")
#     for key, value in cache_info.items():
#         print(f"   {key}: {value}")
    
#     # Show final cache status
#     print(f"\n")
#     show_cache_status()