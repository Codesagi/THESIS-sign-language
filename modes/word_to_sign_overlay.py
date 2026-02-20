# ============================================================
# modes/word_to_sign_overlay.py — 3D Mesh AR Overlay
#
# Uses PyVista mesh rendering (same as 3D viewer) but:
# - Renders to image buffer (off-screen)
# - Composites onto camera feed as overlay
# - Draggable with mouse
# - Real-time accuracy feedback
# ============================================================

from __future__ import annotations
import cv2
import numpy as np
from typing import Dict
import time

# Lazy imports
def _lazy_imports():
    import pyvista as pv
    import mediapipe as mp
    return pv, mp.solutions.hands, mp.solutions.drawing_utils

import config
from core.recognition import normalize_landmarks


# Global state
_overlay_center = None
_dragging = False
_drag_start = None


def _mouse_callback(event, x, y, flags, param):
    """Handle dragging."""
    global _overlay_center, _dragging, _drag_start
    
    if _overlay_center is None:
        return
    
    if event == cv2.EVENT_LBUTTONDOWN:
        cx, cy = _overlay_center
        if np.sqrt((x - cx)**2 + (y - cy)**2) < 150:
            _dragging = True
            _drag_start = (x - cx, y - cy)
    
    elif event == cv2.EVENT_MOUSEMOVE and _dragging:
        _overlay_center = (x - _drag_start[0], y - _drag_start[1])
    
    elif event == cv2.EVENT_LBUTTONUP:
        _dragging = False


def run_word_to_sign_overlay(
    reference_landmarks: Dict[str, list],
    language: str,
    label: str,
):
    """
    3D mesh overlay on camera feed with hand detection.
    
    Renders PyVista mesh to buffer, then composites onto camera.
    """
    global _overlay_center, _dragging
    
    if label not in reference_landmarks:
        print(f"[ERROR] No reference for '{label}'")
        return
    
    # Lazy load
    pv, mp_hands, mp_drawing = _lazy_imports()
    
    # Normalize reference
    ref_lm = reference_landmarks[label]
    ref_arr = normalize_landmarks(ref_lm)
    
    # Create 3D mesh (same as viewer)
    from visualization.hand_3d_combined import create_hand_mesh_from_landmarks
    hand_mesh = create_hand_mesh_from_landmarks(ref_lm, scale=10.0)
    
    # Setup off-screen plotter
    mesh_size = 400  # render size for 3D mesh
    plotter = pv.Plotter(off_screen=True, window_size=[mesh_size, mesh_size])
    plotter.set_background([0, 0, 0, 0])  # transparent background
    
    # Add mesh parts
    for i in range(hand_mesh.n_blocks):
        mesh = hand_mesh[i]
        name = hand_mesh.get_block_name(i) or ""
        
        if "skin" in name:
            plotter.add_mesh(mesh, color=(0.95, 0.85, 0.75),
                           smooth_shading=True, specular=0.3, opacity=0.85)
        elif "joint" in name:
            idx = int(name.split("_")[1])
            colors = [(0.7,0.3,0.3), (0.9,0.3,0.3), (0.3,0.9,0.3),
                     (0.3,0.6,0.9), (0.9,0.9,0.3), (0.9,0.5,0.9)]
            color = colors[min(idx // 4, 5)]
            plotter.add_mesh(mesh, color=color, smooth_shading=True, specular=0.5)
        elif "bone" in name:
            plotter.add_mesh(mesh, color=(0.85, 0.75, 0.65),
                           smooth_shading=True, specular=0.2, opacity=0.8)
    
    plotter.camera_position = [(3, 3, 4), (0, 0, 0), (0, 1, 0)]
    
    # MediaPipe
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
    
    print(f"[3D Overlay] {language} - {label}")
    print("[3D Overlay] PyVista mesh rendered as camera overlay")
    print("[3D Overlay] Click and drag to reposition • ESC to exit")
    
    # Initialize position
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        _overlay_center = (w // 2, h // 2)
    
    window_name = f"3D Overlay: {language} - {label}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _mouse_callback)
    
    # Smoothing
    similarity_history = []
    
    try:
        # Render mesh once
        mesh_img = plotter.screenshot(transparent_background=True, return_img=True)
        mesh_img = cv2.cvtColor(mesh_img, cv2.COLOR_RGB2RGBA)  # add alpha
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Detect hand
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            similarity = 0.0
            
            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]
                
                # Draw user hand
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                )
                
                # Compute similarity
                user_arr = normalize_landmarks(hand_lm.landmark)
                
                is_left = False
                if result.multi_handedness:
                    is_left = result.multi_handedness[0].classification[0].label == "Left"
                
                if is_left:
                    user_arr_m = user_arr.copy()
                    user_arr_m[:, 0] *= -1
                    user_flat = user_arr_m.ravel()
                else:
                    user_flat = user_arr.ravel()
                
                ref_flat = ref_arr.ravel()
                norm_ref = np.linalg.norm(ref_flat)
                norm_user = np.linalg.norm(user_flat)
                
                if norm_ref > 1e-9 and norm_user > 1e-9:
                    cos_sim = np.dot(ref_flat, user_flat) / (norm_ref * norm_user)
                    similarity = max(0.0, cos_sim * 100)
                
                similarity_history.append(similarity)
                if len(similarity_history) > 5:
                    similarity_history.pop(0)
                similarity = np.mean(similarity_history)
            
            # Composite 3D mesh overlay
            cx, cy = _overlay_center
            _overlay_mesh(frame, mesh_img, cx, cy, alpha=0.6)
            
            # Draw UI
            _draw_ui(frame, similarity, label, lang_color, _dragging)
            
            cv2.imshow(window_name, frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    finally:
        plotter.close()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        _overlay_center = None
        _dragging = False
        print(f"[3D Overlay] Session ended")


def _overlay_mesh(frame, mesh_img, cx, cy, alpha=0.6):
    """Composite RGBA mesh image onto BGR frame at position (cx, cy)."""
    h, w = frame.shape[:2]
    mh, mw = mesh_img.shape[:2]
    
    # Calculate overlay region
    x1 = max(0, cx - mw // 2)
    y1 = max(0, cy - mh // 2)
    x2 = min(w, x1 + mw)
    y2 = min(h, y1 + mh)
    
    # Calculate mesh region
    mx1 = max(0, mw // 2 - cx)
    my1 = max(0, mh // 2 - cy)
    mx2 = mx1 + (x2 - x1)
    my2 = my1 + (y2 - y1)
    
    if x2 <= x1 or y2 <= y1:
        return  # out of bounds
    
    # Extract regions
    frame_region = frame[y1:y2, x1:x2]
    mesh_region = mesh_img[my1:my2, mx1:mx2]
    
    # Alpha blend using mesh alpha channel
    mesh_rgb = mesh_region[:, :, :3]
    mesh_alpha = mesh_region[:, :, 3:4] / 255.0 * alpha
    
    blended = (mesh_rgb * mesh_alpha + frame_region * (1 - mesh_alpha)).astype(np.uint8)
    frame[y1:y2, x1:x2] = blended


def _draw_ui(frame, similarity, label, lang_color, is_dragging):
    """Draw accuracy and hints."""
    h, w = frame.shape[:2]
    
    # Accuracy meter
    meter_w, meter_h = 300, 50
    meter_x = (w - meter_w) // 2
    meter_y = 20
    
    cv2.rectangle(frame, (meter_x, meter_y),
                 (meter_x + meter_w, meter_y + meter_h), (30, 30, 30), -1)
    
    fill_w = int(meter_w * similarity / 100)
    bar_color = (0, 0, 255) if similarity < 50 else (0, 165, 255) if similarity < 80 else (0, 255, 0)
    
    cv2.rectangle(frame, (meter_x, meter_y),
                 (meter_x + fill_w, meter_y + meter_h), bar_color, -1)
    cv2.rectangle(frame, (meter_x, meter_y),
                 (meter_x + meter_w, meter_y + meter_h), (200, 200, 200), 2)
    
    cv2.putText(frame, f"Accuracy: {similarity:.1f}%",
               (meter_x + 10, meter_y + 33),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Feedback
    feedback, color = (
        ("Perfect! ✓", (0, 255, 0)) if similarity >= 85 else
        ("Good!", (0, 200, 255)) if similarity >= 70 else
        ("Keep trying...", (0, 165, 255)) if similarity >= 50 else
        ("Align your hand", (0, 0, 255))
    )
    cv2.putText(frame, feedback, (meter_x, meter_y + meter_h + 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Hints
    hint = "Dragging 3D mesh..." if is_dragging else "Click and drag 3D mesh to move"
    cv2.putText(frame, hint, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Label
    cv2.rectangle(frame, (0, h - 50), (220, h), (20, 20, 20), -1)
    cv2.putText(frame, f"Target: {label}", (10, h - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, lang_color, 2)
    
    cv2.putText(frame, "ESC to exit", (w - 130, h - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)