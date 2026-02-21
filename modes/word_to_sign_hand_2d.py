# ============================================================
# modes/word_to_sign_hand_2d.py — 2D Skeleton AR
#
# 2D reference skeleton floats above your hand
# No PyVista threading — pure OpenCV
# Fast, stable, works on all platforms
# ============================================================

from __future__ import annotations
import cv2
import numpy as np
from typing import Dict

import config
from core.recognition import normalize_landmarks


# Hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]


def run_hand_2d_ar(
    reference_landmarks: Dict[str, list],
    language: str,
    label: str,
):
    """
    2D skeleton AR anchored to hand.
    
    Yellow skeleton floats above your hand.
    No threading issues - pure OpenCV.
    """
    print(f"[2D Hand AR] Starting {language} - {label}")
    
    if label not in reference_landmarks:
        print(f"[ERROR] No reference for '{label}'")
        return
    
    # Lazy import MediaPipe
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
    except Exception as e:
        print(f"[ERROR] MediaPipe import failed: {e}")
        return
    
    # Normalize reference
    ref_lm = reference_landmarks[label]
    ref_arr = normalize_landmarks(ref_lm)  # (21, 3)
    
    # MediaPipe setup
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
    
    print(f"[2D AR] Yellow skeleton will float above your hand")
    print(f"[2D AR] Match your hand (green) to reference (yellow)")
    print(f"[2D AR] ESC to exit")
    
    window_name = f"Hand AR: {language} - {label}"
    similarity_history = []
    
    try:
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
            hand_detected = False
            
            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]
                hand_detected = True
                
                # ══════════════════════════════════════════════════
                # Draw user hand (green) FIRST (underneath)
                # ══════════════════════════════════════════════════
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                )
                
                # ══════════════════════════════════════════════════
                # Get hand position for anchoring
                # ══════════════════════════════════════════════════
                wrist = hand_lm.landmark[0]
                wrist_x = int(wrist.x * w)
                wrist_y = int(wrist.y * h)
                
                # Calculate hand size (for scaling)
                # Use distance from wrist to middle finger MCP
                middle_mcp = hand_lm.landmark[9]
                middle_x = int(middle_mcp.x * w)
                middle_y = int(middle_mcp.y * h)
                hand_size = np.sqrt((middle_x - wrist_x)**2 + (middle_y - wrist_y)**2)
                
                # Scale reference to match user's hand size
                scale = hand_size * 1.5  # Make reference slightly larger
                
                # Position reference above hand
                offset_x = 0      # Directly above (no horizontal shift)
                offset_y = -int(hand_size * 2.5)  # Above hand
                
                ref_center_x = wrist_x + offset_x
                ref_center_y = wrist_y + offset_y
                
                # ══════════════════════════════════════════════════
                # Project reference landmarks to screen
                # ══════════════════════════════════════════════════
                ref_2d = []
                for i in range(21):
                    x = ref_arr[i, 0] * scale + ref_center_x
                    y = ref_arr[i, 1] * scale + ref_center_y
                    ref_2d.append((int(x), int(y)))
                
                # ══════════════════════════════════════════════════
                # Draw reference skeleton (yellow/cyan) ON TOP
                # ══════════════════════════════════════════════════
                _draw_reference_skeleton(frame, ref_2d, alpha=0.8)
                
                # Draw connecting line (shows reference is linked to hand)
                cv2.line(
                    frame,
                    (wrist_x, wrist_y),
                    (ref_center_x, ref_center_y),
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA
                )
                
                # ══════════════════════════════════════════════════
                # Compute similarity
                # ══════════════════════════════════════════════════
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
            
            # Draw UI
            _draw_ui(frame, hand_detected, similarity, label, lang_color)
            
            cv2.imshow(window_name, frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print(f"[2D Hand AR] Session ended")


def _draw_reference_skeleton(frame, ref_2d, alpha=0.8):
    """
    Draw reference skeleton with 3D depth effect.
    """
    overlay = frame.copy()
    
    # ── Draw connections (yellow with thickness variation) ────
    for (i, j) in HAND_CONNECTIONS:
        p1 = ref_2d[i]
        p2 = ref_2d[j]
        
        # Vary thickness based on connection type
        if i == 0:  # Palm connections
            thickness = 5
            color = (0, 200, 255)  # Orange
        else:
            thickness = 4
            color = (0, 255, 255)  # Yellow
        
        cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
    
    # ── Draw joints (gradient circles for 3D effect) ──────────
    for i, pt in enumerate(ref_2d):
        # Outer circle (darker)
        outer_radius = 9 if i == 0 else 7
        outer_color = (0, 200, 255) if i == 0 else (0, 220, 255)
        cv2.circle(overlay, pt, outer_radius, outer_color, -1, cv2.LINE_AA)
        
        # Inner highlight (lighter, offset for 3D)
        highlight_offset = (-2, -2)
        highlight_pt = (pt[0] + highlight_offset[0], pt[1] + highlight_offset[1])
        inner_radius = 4 if i == 0 else 3
        inner_color = (100, 255, 255)
        cv2.circle(overlay, highlight_pt, inner_radius, inner_color, -1, cv2.LINE_AA)
        
        # Wrist gets extra ring
        if i == 0:
            cv2.circle(overlay, pt, 13, (0, 180, 255), 2, cv2.LINE_AA)
    
    # Alpha blend
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_ui(frame, hand_detected, similarity, label, lang_color):
    """Draw UI elements."""
    h, w = frame.shape[:2]
    
    # Status
    if not hand_detected:
        cv2.putText(frame, "Show your hand to camera", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, "Yellow skeleton will appear above your hand", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "Hand Detected ✓", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Yellow = reference • Green = your hand", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Accuracy meter
    if similarity > 0:
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
        if similarity >= 85:
            feedback, color = "Perfect! ✓", (0, 255, 0)
        elif similarity >= 70:
            feedback, color = "Good!", (0, 200, 255)
        elif similarity >= 50:
            feedback, color = "Keep trying...", (0, 165, 255)
        else:
            feedback, color = "Match the yellow hand", (0, 0, 255)
        
        cv2.putText(frame, feedback, (meter_x, meter_y + meter_h + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Label
    cv2.rectangle(frame, (0, h - 50), (220, h), (20, 20, 20), -1)
    cv2.putText(frame, f"Target: {label}", (10, h - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, lang_color, 2)
    
    cv2.putText(frame, "ESC to exit", (w - 130, h - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)