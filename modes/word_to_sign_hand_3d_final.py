# ============================================================
# modes/word_to_sign_hand_3d_final.py — Enhanced 3D Hand AR with Phone Anchoring
#
# Features:
# - YOLO phone detection with smooth tracking
# - 3D mesh anchored above phone
# - WASD rotation controls + Q/E zoom
# - Hand tracking for accuracy
# - Visual feedback (phone box, anchor point, status)
# ============================================================

from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from ultralytics import YOLO

import config
from core.recognition import normalize_landmarks

# Global rotation state
azimuth = 0
elevation = 30
distance = 15

# Phone tracking with smoothing
phone_tracker = {
    'positions': [],      # History buffer
    'max_history': 5,     # Frames to average
    'last_valid': None,   # Last known good position
    'confidence': 0.0,    # Detection confidence
}


def run_hand_3d_ar_final(
    reference_landmarks: Dict[str, list],
    language: str,
    label: str,
):
    """
    Enhanced 3D mesh AR with robust phone anchoring.
    
    Improvements:
    - Smooth exponential moving average tracking
    - Visual indicators (phone box, anchor crosshair)
    - Enhanced controls (WASD + QE + R reset)
    - Better positioning (scales with phone size)
    - Status display
    """
    print(f"[3D Hand AR] Starting {language} - {label}")
    global azimuth, elevation, distance, phone_tracker

    if label not in reference_landmarks:
        print(f"[ERROR] No reference for '{label}'")
        return

    # -------------------------
    # Lazy imports
    # -------------------------
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        print("[DEBUG] MediaPipe loaded")
    except Exception as e:
        print(f"[ERROR] MediaPipe import failed: {e}")
        return

    # Load phone detection model
    try:
        phone_model = YOLO("yolov8n.pt")
        print("[DEBUG] YOLO phone detection ready")
    except Exception as e:
        print(f"[ERROR] YOLO model failed: {e}")
        print("[INFO] Install with: pip install ultralytics")
        return

    # -------------------------
    # Setup reference
    # -------------------------
    ref_lm = reference_landmarks[label]
    ref_arr = normalize_landmarks(ref_lm)

    # -------------------------
    # Create 3D mesh
    # -------------------------
    try:
        from visualization.hand_3d_combined import create_hand_mesh_from_landmarks
        hand_mesh = create_hand_mesh_from_landmarks(ref_lm, scale=10.0)
        print(f"[DEBUG] 3D mesh created: {hand_mesh.n_blocks} blocks")
    except Exception as e:
        print(f"[ERROR] Mesh creation failed: {e}")
        return

    # -------------------------
    # Setup mesh renderer
    # -------------------------
    try:
        from ar.mesh_renderer import MeshRenderer, composite_mesh_on_frame
        renderer = MeshRenderer(hand_mesh, render_size=250)
        print("[DEBUG] Mesh renderer initialized")
    except Exception as e:
        print(f"[ERROR] Renderer setup failed: {e}")
        return

    # -------------------------
    # MediaPipe hand tracking
    # -------------------------
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=config.MIN_DETECTION_CONF,
        min_tracking_confidence=config.MIN_TRACKING_CONF,
    )

    # -------------------------
    # Camera
    # -------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        renderer.close()
        return

    lang_color = config.COLOR_ASL if language == "ASL" else config.COLOR_FSL
    window_name = f"3D Hand AR: {language} - {label}"
    similarity_history = []

    print(f"[3D Hand AR] Place phone in camera view")
    print(f"[3D Hand AR] 3D mesh will anchor above phone")
    print(f"[3D Hand AR] Controls: WASD (rotate), QE (zoom), R (reset)")
    print(f"[3D Hand AR] ESC to exit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # -------------------------
            # Detect phone with smooth tracking
            # -------------------------
            phone_detected, phone_bbox, anchor_pos = _detect_and_track_phone(
                frame, phone_model, phone_tracker, w, h
            )

            # -------------------------
            # Render and composite 3D mesh
            # -------------------------
            if phone_detected and anchor_pos:
                # Render mesh with current view settings
                mesh_image = renderer.render_view(
                    azimuth=azimuth,
                    elevation=elevation,
                    distance=distance
                )

                # Ensure anchor is within safe bounds
                mh, mw = mesh_image.shape[:2]
                safe_x = np.clip(anchor_pos[0], mw // 2, w - mw // 2)
                safe_y = np.clip(anchor_pos[1], mh // 2, h - mh // 2)

                # Composite mesh onto frame
                composite_mesh_on_frame(
                    frame,
                    mesh_image,
                    safe_x,
                    safe_y,
                    alpha=0.85
                )

                # Draw visual anchoring indicators
                _draw_phone_indicator(frame, phone_bbox, (safe_x, safe_y))

            # -------------------------
            # Hand detection and accuracy
            # -------------------------
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            similarity = 0.0
            hand_detected = False

            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]
                hand_detected = True

                # Draw green skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                )

                # Compute similarity
                user_arr = normalize_landmarks(hand_lm.landmark)
                is_left = (result.multi_handedness and 
                          result.multi_handedness[0].classification[0].label == "Left")

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

            # -------------------------
            # Draw comprehensive UI
            # -------------------------
            _draw_ui(
                frame, 
                phone_detected, 
                hand_detected, 
                similarity, 
                label, 
                lang_color,
                phone_tracker['confidence']
            )

            cv2.imshow(window_name, frame)

            # -------------------------
            # Keyboard controls
            # -------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('a'):  # Rotate left
                azimuth -= 15
            elif key == ord('d'):  # Rotate right
                azimuth += 15
            elif key == ord('w'):  # Tilt up
                elevation = min(elevation + 15, 80)
            elif key == ord('s'):  # Tilt down
                elevation = max(elevation - 15, -80)
            elif key == ord('q'):  # Zoom in
                distance = max(distance - 1, 10)
            elif key == ord('e'):  # Zoom out
                distance = min(distance + 1, 25)
            elif key == ord('r'):  # Reset view
                azimuth = 0
                elevation = 30
                distance = 15
                print("[3D AR] View reset to defaults")

    finally:
        renderer.close()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        # Reset tracker
        phone_tracker['positions'].clear()
        phone_tracker['last_valid'] = None
        phone_tracker['confidence'] = 0.0
        print(f"[3D Hand AR] Session ended")


def _detect_and_track_phone(
    frame: np.ndarray,
    model: YOLO,
    tracker: dict,
    frame_w: int,
    frame_h: int
) -> Tuple[bool, Optional[Tuple], Optional[Tuple]]:
    """
    Detect phone and return smoothed anchor position.
    
    Returns
    -------
    detected : bool
        Phone found this frame
    bbox : tuple or None
        (x1, y1, x2, y2) phone bounding box
    anchor : tuple or None
        (x, y) smoothed anchor position for mesh
    """
    # Run YOLO detection
    results = model.predict(frame, conf=0.3, verbose=False)
    
    phone_bbox = None
    raw_anchor = None
    confidence = 0.0
    
    # Look for cell phone (class 67 in COCO dataset)
    if results and len(results[0].boxes) > 0:
        for box, cls_id, conf in zip(
            results[0].boxes.xyxy, 
            results[0].boxes.cls,
            results[0].boxes.conf
        ):
            if int(cls_id) == 67:  # Cell phone
                x1, y1, x2, y2 = map(int, box)
                phone_bbox = (x1, y1, x2, y2)
                confidence = float(conf)
                
                # Calculate raw anchor position
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                phone_height = y2 - y1
                
                # Position mesh above phone (120% of phone height)
                offset_y = int(phone_height * 1.2)
                raw_anchor = (center_x, center_y - offset_y)
                break
    
    # Apply smoothing
    if raw_anchor:
        # Add to history
        tracker['positions'].append(raw_anchor)
        if len(tracker['positions']) > tracker['max_history']:
            tracker['positions'].pop(0)
        
        # Exponential moving average
        if tracker['last_valid']:
            alpha = 0.3  # Smoothing factor (0.3 = 30% new, 70% old)
            smooth_x = int(alpha * raw_anchor[0] + (1 - alpha) * tracker['last_valid'][0])
            smooth_y = int(alpha * raw_anchor[1] + (1 - alpha) * tracker['last_valid'][1])
            smooth_anchor = (smooth_x, smooth_y)
        else:
            # First detection - use as-is
            smooth_anchor = raw_anchor
        
        tracker['last_valid'] = smooth_anchor
        tracker['confidence'] = confidence
        
        return True, phone_bbox, smooth_anchor
    
    else:
        # No detection - gradually reduce confidence
        tracker['confidence'] *= 0.9
        
        # Keep last position for short time (helps with brief occlusions)
        if len(tracker['positions']) > 0:
            tracker['positions'].pop(0)
        
        if len(tracker['positions']) == 0:
            tracker['last_valid'] = None
        
        # Return last valid position if confidence still reasonable
        if tracker['confidence'] > 0.1 and tracker['last_valid']:
            return False, None, tracker['last_valid']
        
        return False, None, None


def _draw_phone_indicator(frame: np.ndarray, bbox: Optional[Tuple], anchor: Tuple):
    """
    Draw visual indicators for phone and anchor point.
    
    Shows:
    - Yellow box around phone
    - Magenta crosshair at anchor
    - Connecting line
    - Label
    """
    # Draw phone bounding box
    if bbox:
        x1, y1, x2, y2 = bbox
        
        # Box outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Label with background
        label_text = "Phone"
        (text_w, text_h), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            frame, 
            (x1, y1 - text_h - 8), 
            (x1 + text_w + 4, y1), 
            (0, 255, 255), 
            -1
        )
        cv2.putText(
            frame, label_text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
        )
        
        # Draw connecting line to anchor
        phone_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.line(
            frame, 
            phone_center, 
            anchor, 
            (150, 150, 150), 
            1, 
            cv2.LINE_AA
        )
    
    # Draw anchor crosshair
    if anchor:
        ax, ay = anchor
        size = 15
        
        # Crosshair lines
        cv2.line(frame, (ax - size, ay), (ax + size, ay), (255, 0, 255), 2)
        cv2.line(frame, (ax, ay - size), (ax, ay + size), (255, 0, 255), 2)
        
        # Center dot
        cv2.circle(frame, (ax, ay), 5, (255, 0, 255), -1)
        cv2.circle(frame, (ax, ay), 7, (255, 255, 255), 1)


def _draw_ui(
    frame, 
    phone_detected, 
    hand_detected, 
    similarity, 
    label, 
    lang_color,
    phone_confidence
):
    """Draw comprehensive UI with status, accuracy, and controls."""
    h, w = frame.shape[:2]

    # -------------------------
    # Status indicators (top left)
    # -------------------------
    status_y = 30
    
    # Phone status
    if phone_detected:
        phone_text = f"Phone: ✓ ({phone_confidence*100:.0f}%)"
        phone_color = (0, 255, 0)
    else:
        phone_text = "Phone: ✗ Not detected"
        phone_color = (0, 0, 255)
    
    cv2.putText(
        frame, phone_text, (10, status_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, phone_color, 2
    )
    
    # Hand status
    hand_text = "Hand: ✓ Tracked" if hand_detected else "Hand: ✗ Not visible"
    hand_color = (0, 255, 0) if hand_detected else (100, 100, 100)
    cv2.putText(
        frame, hand_text, (10, status_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2
    )

    # -------------------------
    # Accuracy meter (top center)
    # -------------------------
    if similarity > 0:
        meter_w, meter_h = 300, 50
        meter_x = (w - meter_w) // 2
        meter_y = 20

        # Background
        cv2.rectangle(
            frame, 
            (meter_x, meter_y),
            (meter_x + meter_w, meter_y + meter_h), 
            (30, 30, 30), 
            -1
        )

        # Fill bar
        fill_w = int(meter_w * similarity / 100)
        if similarity < 50:
            bar_color = (0, 0, 255)
        elif similarity < 80:
            bar_color = (0, 165, 255)
        else:
            bar_color = (0, 255, 0)
        
        cv2.rectangle(
            frame, 
            (meter_x, meter_y),
            (meter_x + fill_w, meter_y + meter_h), 
            bar_color, 
            -1
        )
        
        # Border
        cv2.rectangle(
            frame, 
            (meter_x, meter_y),
            (meter_x + meter_w, meter_y + meter_h), 
            (200, 200, 200), 
            2
        )
        
        # Text
        cv2.putText(
            frame, 
            f"Accuracy: {similarity:.1f}%",
            (meter_x + 10, meter_y + 33),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )

        # Feedback message
        if similarity >= 85:
            feedback, color = "Perfect! ✓", (0, 255, 0)
        elif similarity >= 70:
            feedback, color = "Good!", (0, 200, 255)
        elif similarity >= 50:
            feedback, color = "Keep trying...", (0, 165, 255)
        else:
            feedback, color = "Match the reference", (0, 0, 255)

        cv2.putText(
            frame, 
            feedback, 
            (meter_x, meter_y + meter_h + 35),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            color, 
            2
        )

    # -------------------------
    # Controls guide (bottom left)
    # -------------------------
    controls = [
        "Controls:",
        "A/D: Rotate",
        "W/S: Tilt",
        "Q/E: Zoom",
        "R: Reset",
    ]
    
    controls_y = h - 140
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, controls_y - 10), (200, h - 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    for i, text in enumerate(controls):
        cv2.putText(
            frame, text, (10, controls_y + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

    # -------------------------
    # Target label (bottom center-left)
    # -------------------------
    cv2.rectangle(frame, (0, h - 50), (250, h), (20, 20, 20), -1)
    cv2.putText(
        frame, f"Target: {label}", (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, lang_color, 2
    )

    # -------------------------
    # Current view settings (top right)
    # -------------------------
    view_text = f"Az:{azimuth}° El:{elevation}° D:{distance}"
    cv2.putText(
        frame, view_text, (w - 250, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    
    # -------------------------
    # Exit hint (bottom right)
    # -------------------------
    cv2.putText(
        frame, "ESC to exit", (w - 130, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1
    )