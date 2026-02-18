# ============================================================
# modes/word_to_sign_ar.py — AR Word-to-Sign Mode
#
# User sees 2D reference skeleton overlaid on camera feed.
# Real-time similarity feedback as they align their hand.
# Click and drag to reposition overlay.
# ============================================================

from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Tuple

# Lazy imports — defer heavy deps until needed
def _lazy_mediapipe():
    import mediapipe as mp
    return mp.solutions.hands, mp.solutions.drawing_utils

import config


# Global state for mouse dragging
_overlay_center = None  # (x, y) pixel position
_dragging = False
_drag_start = None


def _mouse_callback(event, x, y, flags, param):
    """Handle mouse events for dragging overlay."""
    global _overlay_center, _dragging, _drag_start
    
    if _overlay_center is None:
        return
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is near overlay center (within 100px radius)
        cx, cy = _overlay_center
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        if dist < 100:
            _dragging = True
            _drag_start = (x - cx, y - cy)  # offset from center
    
    elif event == cv2.EVENT_MOUSEMOVE and _dragging:
        # Update overlay position
        _overlay_center = (x - _drag_start[0], y - _drag_start[1])
    
    elif event == cv2.EVENT_LBUTTONUP:
        _dragging = False
        _drag_start = None


def run_ar_word_to_sign(
    reference_landmarks: Dict[str, list],  # {label: [(x,y,z), ...]}
    language: str,
    label: str,
):
    """
    AR overlay mode with real-time similarity feedback.
    
    Click and drag the yellow skeleton to reposition it.
    
    Parameters
    ----------
    reference_landmarks : dict mapping label → 21-point landmark list
    language : "ASL" or "FSL"
    label : which sign to practice (e.g., "A")
    """
    global _overlay_center, _dragging
    
    from ar.renderer import SkeletonRenderer
    from core.recognition import normalize_landmarks  # use same as ML models
    
    if label not in reference_landmarks:
        print(f"[ERROR] No reference for '{label}'")
        return
    
    # Convert reference to normalized numpy array
    ref_lm = reference_landmarks[label]
    ref_arr = normalize_landmarks(ref_lm)  # (21, 3) normalized
    
    # Setup AR renderer
    renderer = SkeletonRenderer(
        line_color=(0, 255, 255),    # yellow
        joint_color=(255, 255, 0),   # cyan
        line_thickness=3,
        joint_radius=5,
    )
    
    # MediaPipe (lazy loaded)
    mp_hands, mp_drawing = _lazy_mediapipe()
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=config.MIN_DETECTION_CONF,
        min_tracking_confidence=config.MIN_TRACKING_CONF,
    )
    
    # Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    
    lang_color = config.COLOR_ASL if language == "ASL" else config.COLOR_FSL
    
    print(f"[AR] {language} - {label}")
    print("[AR] Align your hand with the yellow skeleton overlay")
    print("[AR] Click and drag to reposition overlay")
    print("[AR] ESC to exit")
    
    # Initialize overlay position at center
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        _overlay_center = (w // 2, h // 2)
    
    # Setup mouse callback
    window_name = f"AR Practice: {language} - {label}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _mouse_callback)
    
    # Smoothing for similarity
    similarity_history = []
    history_size = 5
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # ── Step 1: Draw reference skeleton overlay ──────────
            ref_2d = _project_to_screen(ref_arr, (h, w), _overlay_center, scale=150.0)
            renderer.draw(frame, ref_2d, alpha=0.5)
            
            # ── Step 2: Detect user hand ─────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            similarity = 0.0
            
            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]
                
                # Draw user hand (green)
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                )
                
                # ── Step 3: Compute similarity (same as ML models) ───
                user_arr = normalize_landmarks(hand_lm.landmark)  # (21, 3)
                
                # Cosine similarity (same metric as inference engine)
                ref_flat = ref_arr.ravel()
                user_flat = user_arr.ravel()
                
                # Handle left/right mirroring
                is_left = False
                if result.multi_handedness:
                    is_left = result.multi_handedness[0].classification[0].label == "Left"
                
                if is_left:
                    user_arr_mirrored = user_arr.copy()
                    user_arr_mirrored[:, 0] *= -1
                    user_flat = user_arr_mirrored.ravel()
                
                # Cosine similarity
                norm_ref = np.linalg.norm(ref_flat)
                norm_user = np.linalg.norm(user_flat)
                if norm_ref > 1e-9 and norm_user > 1e-9:
                    cos_sim = np.dot(ref_flat, user_flat) / (norm_ref * norm_user)
                    similarity = max(0.0, cos_sim * 100)  # convert to %
                
                # Smooth
                similarity_history.append(similarity)
                if len(similarity_history) > history_size:
                    similarity_history.pop(0)
                similarity = np.mean(similarity_history)
            
            # ── Step 4: Draw UI feedback ─────────────────────────
            _draw_accuracy_ui(frame, similarity, label, lang_color, _dragging)
            
            # ── Step 5: Show frame ───────────────────────────────
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        # Reset global state
        _overlay_center = None
        _dragging = False
        print(f"[AR] Session ended for '{label}'")


def _project_to_screen(
    normalized_3d: np.ndarray,  # (21, 3)
    frame_shape: Tuple[int, int],  # (height, width)
    center: Tuple[int, int],
    scale: float = 150.0,
) -> np.ndarray:
    """
    Project normalized 3D landmarks to 2D screen coordinates.
    
    Returns
    -------
    (21, 2) pixel coordinates
    """
    # Take X, Y from 3D (ignore Z for 2D projection)
    pts_2d = normalized_3d[:, :2] * scale
    
    # Translate to center
    cx, cy = center
    pts_2d[:, 0] += cx
    pts_2d[:, 1] += cy
    
    return pts_2d


def _draw_accuracy_ui(
    frame: np.ndarray,
    similarity: float,
    label: str,
    lang_color: tuple,
    is_dragging: bool,
):
    """Draw accuracy meter and feedback text."""
    h, w = frame.shape[:2]
    
    # ── Accuracy meter (top center) ──────────────────────────
    meter_w, meter_h = 300, 50
    meter_x = (w - meter_w) // 2
    meter_y = 20
    
    # Background
    cv2.rectangle(frame, (meter_x, meter_y),
                 (meter_x + meter_w, meter_y + meter_h),
                 (30, 30, 30), -1)
    
    # Fill bar
    fill_w = int(meter_w * similarity / 100)
    if similarity < 50:
        bar_color = (0, 0, 255)       # red
    elif similarity < 80:
        bar_color = (0, 165, 255)     # orange
    else:
        bar_color = (0, 255, 0)       # green
    
    cv2.rectangle(frame, (meter_x, meter_y),
                 (meter_x + fill_w, meter_y + meter_h),
                 bar_color, -1)
    
    # Border
    cv2.rectangle(frame, (meter_x, meter_y),
                 (meter_x + meter_w, meter_y + meter_h),
                 (200, 200, 200), 2)
    
    # Text inside meter
    text = f"Accuracy: {similarity:.1f}%"
    cv2.putText(frame, text, (meter_x + 10, meter_y + 33),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # ── Feedback message ──────────────────────────────────────
    if similarity >= 85:
        feedback, color = "Perfect! ✓", (0, 255, 0)
    elif similarity >= 70:
        feedback, color = "Good!", (0, 200, 255)
    elif similarity >= 50:
        feedback, color = "Keep trying...", (0, 165, 255)
    else:
        feedback, color = "Align your hand", (0, 0, 255)
    
    cv2.putText(frame, feedback, (meter_x, meter_y + meter_h + 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # ── Drag hint ────────────────────────────────────────────
    if is_dragging:
        hint = "Dragging overlay..."
        cv2.putText(frame, hint, (meter_x, meter_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        hint = "Click and drag yellow skeleton to move"
        cv2.putText(frame, hint, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # ── Target label (bottom left) ────────────────────────────
    cv2.rectangle(frame, (0, h - 50), (220, h), (20, 20, 20), -1)
    cv2.putText(frame, f"Target: {label}", (10, h - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, lang_color, 2)
    
    # ── Instructions (bottom right) ───────────────────────────
    cv2.putText(frame, "ESC to exit", (w - 130, h - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)