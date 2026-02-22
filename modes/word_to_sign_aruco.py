# ============================================================
# modes/word_to_sign_aruco.py — ARuco Marker-Anchored 3D AR
#
# 3D mesh floats above ARuco marker
# More precise than phone detection
# Marker ID 0 (print from generated image)
# ============================================================

from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Optional, Tuple

import config
from core.recognition import normalize_landmarks

# Global rotation state
azimuth = 0
elevation = 30
distance = 15

# ARuco tracking state
aruco_tracker = {
    'positions': [],
    'max_history': 3,  # Less smoothing needed (more precise)
    'last_valid': None,
    'marker_size_mm': 100,  # 10cm marker
}


def run_aruco_ar(
    reference_landmarks: Dict[str, list],
    language: str,
    label: str,
):
    """
    ARuco marker-anchored 3D AR.
    
    Features:
    - Precise marker detection (sub-pixel accuracy)
    - 3D pose estimation (6DOF)
    - Stable tracking
    - Same controls as phone version
    """
    print(f"[ARuco AR] Starting {language} - {label}")
    global azimuth, elevation, distance, aruco_tracker

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

    # Setup ARuco detector
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        print("[DEBUG] ARuco detector initialized (DICT_6X6_250)")
    except Exception as e:
        print(f"[ERROR] ARuco setup failed: {e}")
        print("[INFO] OpenCV version may not support ARuco")
        return

    # Camera calibration (approximate)
    # For better results, calibrate your camera
    camera_matrix = None
    dist_coeffs = None

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

    # Get frame size for camera calibration
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        # Simple calibration (approximate)
        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1))

    lang_color = config.COLOR_ASL if language == "ASL" else config.COLOR_FSL
    window_name = f"ARuco AR: {language} - {label}"
    similarity_history = []

    print(f"[ARuco AR] Place ARuco marker (ID 0) in view")
    print(f"[ARuco AR] Generate marker: python -m tools.generate_aruco")
    print(f"[ARuco AR] Print at 10cm x 10cm size")
    print(f"[ARuco AR] Controls: WASD (rotate), QE (zoom), R (reset)")
    print(f"[ARuco AR] ESC to exit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # -------------------------
            # Detect ARuco marker
            # -------------------------
            marker_detected, marker_corners, marker_pose, anchor_pos = _detect_aruco_marker(
                frame, detector, camera_matrix, dist_coeffs, aruco_tracker, w, h
            )

            # -------------------------
            # Render and composite 3D mesh
            # -------------------------
            if marker_detected and anchor_pos:
                # Render mesh with current view
                mesh_image = renderer.render_view(
                    azimuth=azimuth,
                    elevation=elevation,
                    distance=distance
                )

                # Safe bounds
                mh, mw = mesh_image.shape[:2]
                safe_x = np.clip(anchor_pos[0], mw // 2, w - mw // 2)
                safe_y = np.clip(anchor_pos[1], mh // 2, h - mh // 2)

                # Composite mesh
                composite_mesh_on_frame(
                    frame,
                    mesh_image,
                    safe_x,
                    safe_y,
                    alpha=0.85
                )

                # Draw marker indicators
                _draw_marker_indicator(frame, marker_corners, marker_pose, (safe_x, safe_y))

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
            # Draw UI
            # -------------------------
            _draw_ui(
                frame, 
                marker_detected, 
                hand_detected, 
                similarity, 
                label, 
                lang_color
            )

            cv2.imshow(window_name, frame)

            # -------------------------
            # Keyboard controls
            # -------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('a'):
                azimuth -= 15
            elif key == ord('d'):
                azimuth += 15
            elif key == ord('w'):
                elevation = min(elevation + 15, 80)
            elif key == ord('s'):
                elevation = max(elevation - 15, -80)
            elif key == ord('q'):
                distance = max(distance - 1, 10)
            elif key == ord('e'):
                distance = min(distance + 1, 25)
            elif key == ord('r'):
                azimuth = 0
                elevation = 30
                distance = 15
                print("[ARuco AR] View reset")

    finally:
        renderer.close()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        # Reset tracker
        aruco_tracker['positions'].clear()
        aruco_tracker['last_valid'] = None
        print(f"[ARuco AR] Session ended")


def _detect_aruco_marker(
    frame: np.ndarray,
    detector,
    camera_matrix: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray],
    tracker: dict,
    frame_w: int,
    frame_h: int
) -> Tuple[bool, Optional[np.ndarray], Optional[Tuple], Optional[Tuple]]:
    """
    Detect ARuco marker and return smoothed anchor position.
    
    Returns
    -------
    detected : bool
        Marker found
    corners : np.ndarray or None
        Marker corner points
    pose : tuple or None
        (rvec, tvec) if pose estimated
    anchor : tuple or None
        (x, y) anchor position for mesh
    """
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(frame)
    
    marker_corners = None
    marker_pose = None
    raw_anchor = None
    
    if ids is not None and len(ids) > 0:
        # Look for marker ID 0
        marker_idx = None
        for i, marker_id in enumerate(ids):
            if marker_id[0] == 0:
                marker_idx = i
                break
        
        if marker_idx is not None:
            marker_corners = corners[marker_idx][0]
            
            # Calculate raw anchor (center of marker + offset)
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1]))
            
            # Estimate marker size in pixels
            side1 = np.linalg.norm(marker_corners[0] - marker_corners[1])
            side2 = np.linalg.norm(marker_corners[1] - marker_corners[2])
            marker_size_px = (side1 + side2) / 2
            
            # Position mesh above marker (1.5x marker size)
            offset_y = int(marker_size_px * 1.5)
            raw_anchor = (center_x, center_y - offset_y)
            
            # Estimate pose if camera calibrated
            if camera_matrix is not None:
                marker_size = tracker['marker_size_mm'] / 1000.0  # Convert to meters
                
                obj_points = np.array([
                    [-marker_size/2, marker_size/2, 0],
                    [marker_size/2, marker_size/2, 0],
                    [marker_size/2, -marker_size/2, 0],
                    [-marker_size/2, -marker_size/2, 0]
                ], dtype=np.float32)
                
                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    marker_corners,
                    camera_matrix,
                    dist_coeffs
                )
                
                if success:
                    marker_pose = (rvec, tvec)
    
    # Apply minimal smoothing (ARuco is already precise)
    if raw_anchor:
        tracker['positions'].append(raw_anchor)
        if len(tracker['positions']) > tracker['max_history']:
            tracker['positions'].pop(0)
        
        # Simple average (ARuco doesn't need heavy smoothing)
        if len(tracker['positions']) > 0:
            avg_x = int(np.mean([p[0] for p in tracker['positions']]))
            avg_y = int(np.mean([p[1] for p in tracker['positions']]))
            smooth_anchor = (avg_x, avg_y)
        else:
            smooth_anchor = raw_anchor
        
        tracker['last_valid'] = smooth_anchor
        return True, marker_corners, marker_pose, smooth_anchor
    
    else:
        # No detection
        if len(tracker['positions']) > 0:
            tracker['positions'].pop(0)
        
        if len(tracker['positions']) == 0:
            tracker['last_valid'] = None
        
        return False, None, None, tracker['last_valid']


def _draw_marker_indicator(
    frame: np.ndarray,
    corners: Optional[np.ndarray],
    pose: Optional[Tuple],
    anchor: Tuple
):
    """
    Draw ARuco marker indicators.
    
    Shows:
    - Green polygon around marker
    - 3D axes if pose estimated
    - Cyan crosshair at anchor
    """
    # Draw marker outline
    if corners is not None:
        corners_int = corners.astype(int)
        cv2.polylines(frame, [corners_int], True, (0, 255, 0), 3)
        
        # Label
        center = tuple(corners_int[0])
        cv2.putText(
            frame, "ARuco ID:0", 
            (center[0] - 40, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    
    # Draw 3D axes if pose available
    if pose is not None and corners is not None:
        rvec, tvec = pose
        # Note: Would need camera_matrix here to draw axes properly
        # Simplified for now
        pass
    
    # Draw anchor crosshair
    if anchor:
        ax, ay = anchor
        size = 15
        
        # Cyan crosshair
        cv2.line(frame, (ax - size, ay), (ax + size, ay), (255, 255, 0), 2)
        cv2.line(frame, (ax, ay - size), (ax, ay + size), (255, 255, 0), 2)
        cv2.circle(frame, (ax, ay), 5, (255, 255, 0), -1)
        cv2.circle(frame, (ax, ay), 7, (255, 255, 255), 1)


def _draw_ui(frame, marker_detected, hand_detected, similarity, label, lang_color):
    """Draw UI elements."""
    h, w = frame.shape[:2]

    # -------------------------
    # Status indicators
    # -------------------------
    status_y = 30
    
    # Marker status
    if marker_detected:
        marker_text = "Marker: ✓ ID:0 Detected"
        marker_color = (0, 255, 0)
    else:
        marker_text = "Marker: ✗ Place marker in view"
        marker_color = (0, 0, 255)
    
    cv2.putText(
        frame, marker_text, (10, status_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, marker_color, 2
    )
    
    # Hand status
    hand_text = "Hand: ✓ Tracked" if hand_detected else "Hand: ✗ Not visible"
    hand_color = (0, 255, 0) if hand_detected else (100, 100, 100)
    cv2.putText(
        frame, hand_text, (10, status_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2
    )

    # -------------------------
    # Accuracy meter
    # -------------------------
    if similarity > 0:
        meter_w, meter_h = 300, 50
        meter_x = (w - meter_w) // 2
        meter_y = 20

        cv2.rectangle(
            frame, 
            (meter_x, meter_y),
            (meter_x + meter_w, meter_y + meter_h), 
            (30, 30, 30), 
            -1
        )

        fill_w = int(meter_w * similarity / 100)
        bar_color = (
            (0, 0, 255) if similarity < 50 else
            (0, 165, 255) if similarity < 80 else
            (0, 255, 0)
        )
        
        cv2.rectangle(
            frame, 
            (meter_x, meter_y),
            (meter_x + fill_w, meter_y + meter_h), 
            bar_color, 
            -1
        )
        
        cv2.rectangle(
            frame, 
            (meter_x, meter_y),
            (meter_x + meter_w, meter_y + meter_h), 
            (200, 200, 200), 
            2
        )
        
        cv2.putText(
            frame, 
            f"Accuracy: {similarity:.1f}%",
            (meter_x + 10, meter_y + 33),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )

        # Feedback
        if similarity >= 85:
            feedback, color = "Perfect! ✓", (0, 255, 0)
        elif similarity >= 70:
            feedback, color = "Good!", (0, 200, 255)
        elif similarity >= 50:
            feedback, color = "Keep trying...", (0, 165, 255)
        else:
            feedback, color = "Match the reference", (0, 0, 255)

        cv2.putText(
            frame, feedback, (meter_x, meter_y + meter_h + 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

    # -------------------------
    # Controls guide
    # -------------------------
    controls = [
        "Controls:",
        "A/D: Rotate",
        "W/S: Tilt",
        "Q/E: Zoom",
        "R: Reset",
    ]
    
    controls_y = h - 140
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, controls_y - 10), (200, h - 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    for i, text in enumerate(controls):
        cv2.putText(
            frame, text, (10, controls_y + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

    # -------------------------
    # Target label
    # -------------------------
    cv2.rectangle(frame, (0, h - 50), (250, h), (20, 20, 20), -1)
    cv2.putText(
        frame, f"Target: {label}", (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, lang_color, 2
    )

    # -------------------------
    # View settings
    # -------------------------
    view_text = f"Az:{azimuth}° El:{elevation}° D:{distance}"
    cv2.putText(
        frame, view_text, (w - 250, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    
    # Exit hint
    cv2.putText(
        frame, "ESC to exit", (w - 130, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1
    )