# ============================================================
# ar/renderer.py — AR overlay rendering (2D skeleton, future 3D mesh)
#
# ARCHITECTURE:
#   BaseRenderer   ← abstract interface
#       ↓
#   SkeletonRenderer (2D lines/joints) ← current implementation
#       ↓
#   MeshRenderer (3D model)           ← future swap-in replacement
#
# Both use same HandAligner for positioning/scaling.
# ============================================================

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
import numpy as np
import cv2


# MediaPipe hand topology
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]


class BaseRenderer(ABC):
    """
    Abstract base for AR hand renderers.
    
    Subclasses implement draw() to render reference hand overlay
    onto a frame. Position/scale is handled by HandAligner.
    """
    
    @abstractmethod
    def draw(
        self,
        frame: np.ndarray,
        landmarks_2d: np.ndarray,  # (21, 2) pixel coords
        alpha: float = 0.7,
    ) -> np.ndarray:
        """
        Render reference hand overlay onto frame.
        
        Parameters
        ----------
        frame : (H, W, 3) BGR image
        landmarks_2d : (21, 2) pixel coordinates
        alpha : transparency (0=invisible, 1=opaque)
        
        Returns
        -------
        frame with overlay drawn (in-place modification)
        """
        pass


class SkeletonRenderer(BaseRenderer):
    """
    2D skeleton renderer — draws lines and joints.
    
    Designed to be easily swappable with MeshRenderer later.
    """
    
    def __init__(
        self,
        line_color: Tuple[int, int, int] = (0, 255, 255),  # yellow
        joint_color: Tuple[int, int, int] = (255, 255, 0), # cyan
        line_thickness: int = 3,
        joint_radius: int = 5,
    ):
        self.line_color = line_color
        self.joint_color = joint_color
        self.line_thickness = line_thickness
        self.joint_radius = joint_radius
    
    def draw(
        self,
        frame: np.ndarray,
        landmarks_2d: np.ndarray,
        alpha: float = 0.7,
    ) -> np.ndarray:
        """Draw skeleton overlay with semi-transparency."""
        
        # Create overlay on separate canvas for alpha blending
        overlay = frame.copy()
        
        # Draw connections
        for (i, j) in HAND_CONNECTIONS:
            p1 = tuple(landmarks_2d[i].astype(int))
            p2 = tuple(landmarks_2d[j].astype(int))
            cv2.line(overlay, p1, p2, self.line_color, self.line_thickness, cv2.LINE_AA)
        
        # Draw joints on top
        for i, pt in enumerate(landmarks_2d):
            center = tuple(pt.astype(int))
            cv2.circle(overlay, center, self.joint_radius, self.joint_color, -1, cv2.LINE_AA)
            # Wrist is slightly larger
            if i == 0:
                cv2.circle(overlay, center, self.joint_radius + 2, self.joint_color, 2, cv2.LINE_AA)
        
        # Blend
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame


# ── Future 3D mesh renderer stub ──────────────────────────────
# Uncomment and implement when ready to swap in 3D

# class MeshRenderer(BaseRenderer):
#     """
#     3D mesh renderer using PyVista or OpenGL.
#     Same interface as SkeletonRenderer.
#     """
#     
#     def __init__(self, mesh_path: str):
#         # Load rigged hand mesh
#         pass
#     
#     def draw(self, frame, landmarks_2d, alpha=0.7):
#         # Project 3D mesh onto 2D frame using landmarks as bone positions
#         # Use OpenCV solvePnP for camera pose estimation
#         pass