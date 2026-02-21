# ============================================================
# core/recognition.py  —  Feature extraction & inference
# ============================================================

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


# ──────────────────────────────────────────────────────────────
# 1. NORMALIZATION  (single source of truth)
# ──────────────────────────────────────────────────────────────

def normalize_landmarks(raw) -> np.ndarray:
    """
    Raw landmarks → normalized (21, 3) float32 array.

    Pipeline
    --------
    1. Parse MediaPipe objects, list-of-dicts, or ndarray.
    2. Wrist-relative translation  (landmark 0 → origin).
    3. Palm-scale normalization    (wrist→middle-MCP distance = 1.0).

    WHY palm-scale instead of max-abs
    ----------------------------------
    max-abs changes with which finger is most extended, so the
    *same* sign at two distances produces slightly different
    feature vectors → overlapping high-confidence predictions.
    Palm distance is invariant to finger pose and only tracks
    hand-to-camera distance, which is exactly what we want to
    remove.
    """
    if isinstance(raw, np.ndarray):
        arr = raw.astype(np.float32).reshape(21, 3)
    else:
        pts = []
        for lm in raw:
            if hasattr(lm, "x"):
                pts.append([lm.x, lm.y, lm.z])
            else:
                pts.append([lm["x"], lm["y"], lm["z"]])
        arr = np.array(pts, dtype=np.float32)

    # wrist to origin
    arr = arr - arr[0]

    # palm-scale
    palm_dist = float(np.linalg.norm(arr[9]))   # wrist → middle-MCP
    if palm_dist > 1e-6:
        arr = arr / palm_dist

    return arr   # (21, 3)


def mirror_landmarks(arr: np.ndarray) -> np.ndarray:
    """
    Horizontally mirror a normalized (21, 3) array.

    Used at TRAINING TIME only to augment right-hand data with
    synthetic left-hand samples, eliminating handedness bias.
    Never call this during live inference.
    """
    m = arr.copy()
    m[:, 0] *= -1
    return m


# ── backward-compatible wrappers ──────────────────────────────

def flatten_landmarks(raw) -> list:
    return normalize_landmarks(raw).flatten().tolist()

def landmarks_to_array(raw) -> np.ndarray:
    return normalize_landmarks(raw)


# ──────────────────────────────────────────────────────────────
# 2. COSINE-SIMILARITY FALLBACK
# ──────────────────────────────────────────────────────────────

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def rank_predictions(
    live_vector: np.ndarray,
    reference_bank: Dict[str, np.ndarray],
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    scores = [
        (lbl, max(0.0, cosine_similarity(live_vector, ref) * 100))
        for lbl, ref in reference_bank.items()
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


# ──────────────────────────────────────────────────────────────
# 3. INFERENCE ENGINE
# ──────────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Production inference wrapper:
      - Lazy load (models loaded once, on first predict call)
      - Class-level cache (never reloaded between frames)
      - Single normalization call per frame
      - Pre-allocated input arrays (no per-frame reshape)
      - Confidence-margin threshold (suppress ambiguous predictions)
    """

    _cache: Dict[Tuple[str, str], "InferenceEngine"] = {}

    @classmethod
    def get(
        cls,
        cnn_path: str,
        rf_path: str,
        cnn_weight: float = 0.6,
        rf_weight: float = 0.4,
    ) -> "InferenceEngine":
        """Return cached engine — models loaded only once per session."""
        key = (cnn_path, rf_path)
        if key not in cls._cache:
            cls._cache[key] = cls(cnn_path, rf_path, cnn_weight, rf_weight)
        return cls._cache[key]

    def __init__(
        self,
        cnn_path: str,
        rf_path: str,
        cnn_weight: float = 0.6,
        rf_weight: float = 0.4,
    ):
        self.cnn_path   = cnn_path
        self.rf_path    = rf_path
        self.cnn_weight = cnn_weight
        self.rf_weight  = rf_weight

        self._cnn   = None
        self._rf    = None
        self._enc   = None
        self._classes: np.ndarray | None = None
        self._loaded = False

        # Pre-allocated input buffers (reused every frame)
        self._buf_1x21x3 = np.zeros((1, 21, 3), dtype=np.float32)
        self._buf_1x63   = np.zeros((1, 63),    dtype=np.float32)

    # ── loading ───────────────────────────────────────────────

    def _load(self):
        if self._loaded:
            return
        import os, joblib

        if os.path.exists(self.cnn_path):
            try:
                import tensorflow as tf
                _ = tf.__version__   # guard against shadow file in project dir
                self._cnn = tf.keras.models.load_model(self.cnn_path)
                enc_p = self.cnn_path.replace(".h5", "_labels.pkl")
                if os.path.exists(enc_p):
                    self._enc = joblib.load(enc_p)
                print(f"[Engine] CNN  loaded ← {self.cnn_path}")
            except (ImportError, AttributeError, Exception) as e:
                print(f"[Engine] CNN  failed: {e}")

        if os.path.exists(self.rf_path):
            try:
                data = joblib.load(self.rf_path)
                self._rf  = data["model"]
                if self._enc is None:
                    self._enc = data["label_encoder"]
                print(f"[Engine] RF   loaded ← {self.rf_path}")
            except Exception as e:
                print(f"[Engine] RF   failed: {e}")

        if self._enc is not None:
            self._classes = self._enc.classes_

        self._loaded = True

    # ── inference ─────────────────────────────────────────────

    def predict_top_k(
        self,
        raw_landmarks,
        k: int = 5,
        margin_threshold: float = 0.10,
    ) -> List[Tuple[str, float]]:
        """
        Parameters
        ----------
        raw_landmarks     : MediaPipe list, list-of-dicts, or (21,3) array
        k                 : Number of top predictions to return
        margin_threshold  : Min gap (0-1) between top-1 and top-2 confidence.
                            If gap < threshold, top-1 confidence is dampened
                            to reflect genuine ambiguity (fixes overlapping
                            high-confidence predictions for similar signs
                            like M/N or S/E).

        Returns
        -------
        [(label, confidence_pct), ...] sorted descending
        """
        self._load()
        if self._classes is None:
            return []

        n_classes = len(self._classes)

        # ── normalize once, write into pre-allocated buffers ──
        arr = normalize_landmarks(raw_landmarks)          # (21,3)
        np.copyto(self._buf_1x21x3[0], arr)
        np.copyto(self._buf_1x63[0], arr.ravel())

        # ── accumulate weighted probabilities ──────────────────
        weighted = np.zeros(n_classes, dtype=np.float64)

        if self._cnn is not None:
            cnn_p = self._cnn.predict(self._buf_1x21x3, verbose=0)[0]
            weighted += cnn_p * self.cnn_weight

        if self._rf is not None:
            rf_p = self._rf.predict_proba(self._buf_1x63)[0]
            weighted += rf_p * self.rf_weight

        if weighted.sum() < 1e-9:
            return []

        # ── confidence-margin filter ───────────────────────────
        if margin_threshold > 0 and n_classes >= 2:
            top2 = np.argpartition(weighted, -2)[-2:]
            top2 = top2[np.argsort(weighted[top2])[::-1]]
            gap = weighted[top2[0]] - weighted[top2[1]]
            if gap < margin_threshold:
                dampen = 1.0 - (margin_threshold - gap)
                weighted[top2[0]] *= dampen

        # ── top-k ─────────────────────────────────────────────
        top_idx = np.argpartition(weighted, -min(k, n_classes))[-k:]
        top_idx = top_idx[np.argsort(weighted[top_idx])[::-1]]

        return [
            (str(self._classes[i]), round(float(weighted[i]) * 100, 2))
            for i in top_idx
        ]

    @property
    def is_ready(self) -> bool:
        self._load()
        return self._classes is not None and (
            self._cnn is not None or self._rf is not None
        )


# ── legacy shim ───────────────────────────────────────────────

def predict_with_ensemble(landmarks, ensemble_model, top_n: int = 5):
    arr = normalize_landmarks(landmarks)
    return ensemble_model.predict_top_k(arr, k=top_n)