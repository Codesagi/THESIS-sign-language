# ============================================================
# core/inference.py — Optimized real-time inference
#
# PERFORMANCE OPTIMIZATIONS:
#   - Models loaded lazily (only when predict() is called)
#   - Frame skipping (predict every N frames)
#   - Prediction caching (reuse between frames)
#   - No TensorFlow/sklearn imports at module level
# ============================================================

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np


class CachedPredictor:
    """
    Wraps InferenceEngine with frame-skip caching for real-time performance.
    
    Usage
    -----
    predictor = CachedPredictor(cnn_path, rf_path, frame_skip=3)
    
    for frame in video:
        landmarks = detect_hand(frame)
        predictions = predictor.predict(landmarks, frame_idx)
        # predictions are only recomputed every 3rd frame
    """
    
    def __init__(
        self,
        cnn_path: str,
        rf_path: str,
        frame_skip: int = 3,
        cnn_weight: float = 0.6,
        rf_weight: float = 0.4,
        use_cnn: bool = True,
        use_rf: bool = True,
    ):
        self.cnn_path = cnn_path
        self.rf_path = rf_path
        self.frame_skip = frame_skip
        self.cnn_weight = cnn_weight
        self.rf_weight = rf_weight
        self.use_cnn = use_cnn
        self.use_rf = use_rf
        
        # Lazy-loaded engine (imports TF/sklearn only when needed)
        self._engine = None
        
        # Cache
        self._last_prediction: List[Tuple[str, float]] = []
        self._last_frame_idx = -999
        self._frame_counter = 0
    
    def _get_engine(self):
        """Lazy load InferenceEngine (imports TF/sklearn here)."""
        if self._engine is None:
            from core.recognition import InferenceEngine
            self._engine = InferenceEngine.get(
                self.cnn_path, self.rf_path,
                self.cnn_weight, self.rf_weight
            )
        return self._engine
    
    @property
    def is_ready(self) -> bool:
        return self._get_engine().is_ready
    
    def predict(
        self,
        landmarks,
        frame_idx: Optional[int] = None,
        k: int = 5,
        margin_threshold: float = 0.10,
        force: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Predict with frame-skip caching.
        
        Parameters
        ----------
        landmarks : MediaPipe landmarks, list of dicts, or (21,3) array
        frame_idx : Optional frame index for cache checking
        k         : Number of top predictions
        margin_threshold : Confidence margin filter
        force     : Bypass cache and recompute
        
        Returns
        -------
        [(label, confidence_pct), ...] sorted descending
        """
        self._frame_counter += 1
        
        # Use frame_idx if provided, else use internal counter
        idx = frame_idx if frame_idx is not None else self._frame_counter
        
        # Check cache
        skip = self.frame_skip
        if not force and idx - self._last_frame_idx < skip and self._last_prediction:
            return self._last_prediction
        
        # Recompute
        engine = self._get_engine()
        preds = engine.predict_top_k(landmarks, k=k, margin_threshold=margin_threshold)
        
        # Update cache
        self._last_prediction = preds
        self._last_frame_idx = idx
        
        return preds