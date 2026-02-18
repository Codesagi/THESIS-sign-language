# ============================================================
# ar/aligner.py — Hand alignment and similarity scoring
#
# RESPONSIBILITIES:
#   - Normalize both user and reference landmarks
#   - Compute alignment similarity
#   - Project normalized coords to screen space
#   - Handle left/right hand mirroring
# ============================================================

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


class HandAligner:
    """
    Aligns user hand with reference template and computes similarity.
    
    Normalization Pipeline
    ----------------------
    1. Translate to wrist origin
    2. Scale by palm distance (wrist → middle-MCP)
    3. Optional: mirror if left hand detected
    
    Usage
    -----
    aligner = HandAligner(reference_landmarks)
    similarity = aligner.compute_similarity(user_landmarks)
    ref_2d = aligner.project_to_screen(frame.shape, center=(320, 240))
    """
    
    def __init__(self, reference_landmarks_3d: np.ndarray):
        """
        Parameters
        ----------
        reference_landmarks_3d : (21, 3) normalized reference pose
        """
        self.reference = self._normalize(reference_landmarks_3d)
        
        # Smoothing for real-time similarity
        self._similarity_history = []
        self._history_size = 5  # moving average window
    
    @staticmethod
    def _normalize(landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks: wrist to origin + palm-scale.
        
        Parameters
        ----------
        landmarks : (21, 3) or (21, 2) array
        
        Returns
        -------
        (21, 3) normalized array
        """
        arr = np.asarray(landmarks, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(21, -1)
        
        # Ensure 3D (pad with zeros if only 2D)
        if arr.shape[1] == 2:
            arr = np.pad(arr, ((0, 0), (0, 1)), constant_values=0.0)
        
        # Wrist to origin
        arr = arr - arr[0]
        
        # Palm-scale (wrist [0] → middle-MCP [9])
        palm_dist = np.linalg.norm(arr[9])
        if palm_dist > 1e-6:
            arr = arr / palm_dist
        
        return arr
    
    def compute_similarity(
        self,
        user_landmarks: np.ndarray,
        smooth: bool = True,
    ) -> float:
        """
        Compute alignment similarity (0-100%).
        
        Uses vectorized Euclidean distance over all 21 joints.
        Lower distance = higher similarity.
        
        Parameters
        ----------
        user_landmarks : (21, 2) or (21, 3) user hand pose
        smooth : apply moving average filter
        
        Returns
        -------
        similarity percentage (0-100)
        """
        user_norm = self._normalize(user_landmarks)
        
        # Vectorized distance
        diff = user_norm - self.reference
        dist = np.linalg.norm(diff, axis=1).mean()  # mean distance per joint
        
        # Convert to similarity % (empirical sigmoid)
        # dist=0 → 100%, dist=0.5 → ~50%, dist=1.0 → ~10%
        similarity = 100.0 * np.exp(-2.0 * dist)
        
        if smooth:
            self._similarity_history.append(similarity)
            if len(self._similarity_history) > self._history_size:
                self._similarity_history.pop(0)
            similarity = np.mean(self._similarity_history)
        
        return float(np.clip(similarity, 0, 100))
    
    def project_to_screen(
        self,
        frame_shape: Tuple[int, int],
        center: Optional[Tuple[int, int]] = None,
        scale: float = 150.0,
    ) -> np.ndarray:
        """
        Project normalized 3D reference to 2D screen coordinates.
        
        Parameters
        ----------
        frame_shape : (height, width) of video frame
        center : (x, y) anchor point, default is frame center
        scale : pixel scale factor (default 150 px ≈ palm width)
        
        Returns
        -------
        (21, 2) pixel coordinates
        """
        h, w = frame_shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        
        cx, cy = center
        
        # Take X, Y from 3D (ignore Z for 2D projection)
        pts_2d = self.reference[:, :2] * scale
        
        # Translate to center
        pts_2d[:, 0] += cx
        pts_2d[:, 1] += cy
        
        return pts_2d
    
    def mirror_if_left_hand(
        self,
        user_landmarks: np.ndarray,
        is_left_hand: bool,
    ) -> np.ndarray:
        """
        Mirror user landmarks if left hand detected.
        
        This allows a single right-hand reference to match both hands.
        
        Parameters
        ----------
        user_landmarks : (21, 3) user pose
        is_left_hand : True if MediaPipe detected left hand
        
        Returns
        -------
        (21, 3) mirrored or original
        """
        if not is_left_hand:
            return user_landmarks
        
        mirrored = user_landmarks.copy()
        mirrored[:, 0] *= -1  # flip X axis
        return mirrored