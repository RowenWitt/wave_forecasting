# mesh/icosahedral.py
"""Clean icosahedral mesh implementation"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple, Dict

from config.base import MeshConfig

class IcosahedralMesh:
    """Create and manage icosahedral mesh for wave modeling"""
    
    def __init__(self, config: MeshConfig):
        self.config = config
        self.vertices = None
        self.faces = None
        self.edges = None
        self.multiscale_edges = None
        
        self._build_mesh()
    
    def _build_mesh(self):
        """Build the icosahedral mesh"""
        print(f"Building icosahedral mesh (level {self.config.refinement_level})...")
        
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
