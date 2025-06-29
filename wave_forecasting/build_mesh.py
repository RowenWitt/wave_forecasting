import numpy as np
from scipy.spatial import SphericalVoronoi
import matplotlib.pyplot as plt

class IcosahedralMesh:
    """Create and manage icosahedral mesh for wave modeling"""
    
    def __init__(self, refinement_level=4):
        """
        refinement_level=4: ~10k nodes (~65km spacing)
        refinement_level=5: ~40k nodes (~28km spacing) 
        refinement_level=6: ~163k nodes (~14km spacing)
        """
        self.refinement_level = refinement_level
        self.vertices = None
        self.faces = None
        self.edges = None
        
        self._build_mesh()
    
    def _build_mesh(self):
        """Build the icosahedral mesh"""
        
        print(f"Building icosahedral mesh (refinement level {self.refinement_level})...")
        
        # Start with basic icosahedron
        vertices, faces = self._create_base_icosahedron()
        
        # Refine iteratively
        for level in range(self.refinement_level):
            vertices, faces = self._subdivide_mesh(vertices, faces)
            print(f"  Level {level+1}: {len(vertices)} vertices, {len(faces)} faces")
        
        # Project to unit sphere
        vertices = self._normalize_to_sphere(vertices)
        
        # Create edges from faces
        edges = self._create_edges_from_faces(faces)
        
        # Create multi-scale edges
        multiscale_edges = self._create_multiscale_edges(vertices, faces)
        
        self.vertices = vertices
        self.faces = faces
        self.edges = edges
        self.multiscale_edges = multiscale_edges
        
        print(f"✅ Mesh complete: {len(vertices)} vertices, {len(edges)} edges")
    
    def _create_base_icosahedron(self):
        """Create the basic 12-vertex icosahedron"""
        
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2
        
        # 12 vertices of icosahedron
        vertices = np.array([
            [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
        ], dtype=float)
        
        # 20 triangular faces
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
            """Get midpoint between two vertices, cache results"""
            edge = tuple(sorted([v1_idx, v2_idx]))
            if edge not in edge_midpoints:
                midpoint = (vertices[v1_idx] + vertices[v2_idx]) / 2
                edge_midpoints[edge] = len(new_vertices)
                new_vertices.append(midpoint)
            return edge_midpoints[edge]
        
        # Subdivide each face
        for face in faces:
            v1, v2, v3 = face
            
            # Get midpoints of each edge
            m12 = get_midpoint(v1, v2)
            m23 = get_midpoint(v2, v3)
            m31 = get_midpoint(v3, v1)
            
            # Create 4 new triangles
            new_faces.extend([
                [v1, m12, m31],
                [v2, m23, m12],
                [v3, m31, m23],
                [m12, m23, m31]
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
        
        # Local edges (from faces) - immediate neighbors
        local_edges = self._create_edges_from_faces(faces)
        
        # Medium edges - 2nd and 3rd neighbors
        medium_edges = self._create_extended_edges(vertices, max_distance=0.3)
        
        # Long edges - distant connections for basin-scale propagation
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
                # Compute great circle distance on unit sphere
                dot_product = np.clip(np.dot(vertices[i], vertices[j]), -1, 1)
                distance = np.arccos(dot_product)
                
                if min_distance <= distance <= max_distance:
                    edges.append([i, j])
        
        return np.array(edges)
    
    def vertices_to_lat_lon(self):
        """Convert 3D vertices to lat/lon coordinates"""
        
        x, y, z = self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2]
        
        # Convert to lat/lon
        lat = np.arcsin(z) * 180 / np.pi
        lon = np.arctan2(y, x) * 180 / np.pi
        
        # Ensure longitude is in [0, 360) to match your ERA5 data
        lon = np.where(lon < 0, lon + 360, lon)
        
        return lat, lon
    
    def plot_mesh(self, figsize=(12, 8)):
        """Visualize the mesh"""
        
        lat, lon = self.vertices_to_lat_lon()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Global view
        axes[0].scatter(lon, lat, s=1, alpha=0.6)
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        axes[0].set_title(f'Icosahedral Mesh (Level {self.refinement_level})')
        axes[0].grid(True, alpha=0.3)
        
        # North Pacific focus
        pacific_mask = (lat >= 10) & (lat <= 60) & (lon >= 120) & (lon <= 240)
        axes[1].scatter(lon[pacific_mask], lat[pacific_mask], s=2, alpha=0.8)
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].set_title('North Pacific Region')
        axes[1].set_xlim(120, 240)
        axes[1].set_ylim(10, 60)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Mesh statistics:")
        print(f"  Total vertices: {len(self.vertices)}")
        print(f"  North Pacific vertices: {pacific_mask.sum()}")
        print(f"  Local edges: {len(self.multiscale_edges['local'])}")
        print(f"  Medium edges: {len(self.multiscale_edges['medium'])}")
        print(f"  Long edges: {len(self.multiscale_edges['long'])}")

# Create the mesh
# mesh = IcosahedralMesh(refinement_level=4)  # Start with level 4 for POC
# mesh.plot_mesh()