# ============================================================
# ar/mesh_renderer.py — 3D Mesh to 2D Image Renderer
#
# Renders PyVista 3D mesh to 2D image buffer
# No threading - generates images on demand
# ============================================================

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
import time


class MeshRenderer:
    """
    Renders 3D hand mesh to 2D images.
    
    Uses PyVista off-screen rendering to generate mesh images
    from different angles. Images can then be composited onto
    camera frames.
    """
    
    def __init__(self, hand_mesh, render_size: int = 400):
        """
        Initialize mesh renderer.
        
        Parameters
        ----------
        hand_mesh : pv.MultiBlock
            PyVista mesh to render
        render_size : int
            Size of rendered images (square)
        """
        import pyvista as pv
        
        self.hand_mesh = hand_mesh
        self.render_size = render_size
        self.plotter = None
        self._cache = {}  # Cache rendered images
        
        print(f"[MeshRenderer] Initializing with size {render_size}x{render_size}")
        self._setup_plotter()
    
    def _setup_plotter(self):
        """Setup off-screen PyVista plotter."""
        import pyvista as pv
        
        self.plotter = pv.Plotter(
            off_screen=True,
            window_size=[self.render_size, self.render_size]
        )
        print("[MeshRenderer] Off-screen plotter created")
    
    def render_view(
        self,
        azimuth: float = 0,
        elevation: float = 30,
        distance: float = 4,
        cache: bool = True
    ) -> np.ndarray:
        """
        Render mesh from specified viewpoint.
        
        Parameters
        ----------
        azimuth : float
            Rotation around vertical axis (degrees)
        elevation : float
            Elevation angle (degrees)
        distance : float
            Camera distance from origin
        cache : bool
            Cache rendered images
        
        Returns
        -------
        np.ndarray
            RGBA image (H, W, 4)
        """
        # Check cache
        cache_key = (round(azimuth, 1), round(elevation, 1), round(distance, 1))
        if cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Clear and setup
        self.plotter.clear()
        self.plotter.set_background([0, 0, 0, 0])  # transparent
        
        # Add mesh blocks
        for i in range(self.hand_mesh.n_blocks):
            mesh_block = self.hand_mesh[i]
            name = self.hand_mesh.get_block_name(i) or ""
            
            if "skin" in name:
                self.plotter.add_mesh(
                    mesh_block,
                    color=(0.95, 0.85, 0.75),
                    smooth_shading=True,
                    specular=0.3,
                    opacity=0.9
                )
            elif "joint" in name:
                idx = int(name.split("_")[1])
                colors = [
                    (0.7, 0.3, 0.3), (0.9, 0.3, 0.3), (0.3, 0.9, 0.3),
                    (0.3, 0.6, 0.9), (0.9, 0.9, 0.3), (0.9, 0.5, 0.9)
                ]
                color = colors[min(idx // 4, 5)]
                self.plotter.add_mesh(
                    mesh_block,
                    color=color,
                    smooth_shading=True,
                    specular=0.5
                )
            elif "bone" in name:
                self.plotter.add_mesh(
                    mesh_block,
                    color=(0.85, 0.75, 0.65),
                    smooth_shading=True,
                    specular=0.2,
                    opacity=0.8
                )
        
        # Set camera position
        import math
        az_rad = math.radians(azimuth)
        el_rad = math.radians(elevation)
        
        x = distance * math.cos(el_rad) * math.sin(az_rad)
        y = distance * math.sin(el_rad)
        z = distance * math.cos(el_rad) * math.cos(az_rad)
        
        self.plotter.camera_position = [(x, y, z), (0, 0, 0), (0, 1, 0)]
        
        # Render
        img = self.plotter.screenshot(
            transparent_background=True,
            return_img=True
        )
        
        # Convert to RGBA
        import cv2
        img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        
        # Cache
        if cache and len(self._cache) < 100:  # Limit cache size
            self._cache[cache_key] = img_rgba
        
        return img_rgba
    
    def render_default(self) -> np.ndarray:
        """Render mesh from default viewpoint."""
        return self.render_view(azimuth=0, elevation=30, distance=4)
    
    def close(self):
        """Clean up resources."""
        if self.plotter:
            self.plotter.close()
            self.plotter = None
        self._cache.clear()
        print("[MeshRenderer] Closed")


def composite_mesh_on_frame(
    frame: np.ndarray,
    mesh_image: np.ndarray,
    center_x: int,
    center_y: int,
    alpha: float = 0.85
) -> np.ndarray:
    """
    Composite RGBA mesh image onto BGR frame.
    
    Parameters
    ----------
    frame : np.ndarray
        BGR frame (H, W, 3)
    mesh_image : np.ndarray
        RGBA mesh (H, W, 4)
    center_x, center_y : int
        Center position for mesh
    alpha : float
        Overall transparency
    
    Returns
    -------
    np.ndarray
        Frame with mesh composited
    """
    h, w = frame.shape[:2]
    mh, mw = mesh_image.shape[:2]
    
    # Calculate overlay region
    x1 = max(0, center_x - mw // 2)
    y1 = max(0, center_y - mh // 2)
    x2 = min(w, x1 + mw)
    y2 = min(h, y1 + mh)
    
    # Calculate mesh region
    mx1 = max(0, mw // 2 - center_x)
    my1 = max(0, mh // 2 - center_y)
    mx2 = mx1 + (x2 - x1)
    my2 = my1 + (y2 - y1)
    
    if x2 <= x1 or y2 <= y1:
        return frame  # Out of bounds
    
    # Extract regions
    frame_region = frame[y1:y2, x1:x2].copy()
    mesh_region = mesh_image[my1:my2, mx1:mx2]
    
    # Alpha blend
    mesh_rgb = mesh_region[:, :, :3]
    mesh_alpha = mesh_region[:, :, 3:4] / 255.0 * alpha
    
    blended = (mesh_rgb * mesh_alpha + frame_region * (1 - mesh_alpha)).astype(np.uint8)
    frame[y1:y2, x1:x2] = blended
    
    return frame