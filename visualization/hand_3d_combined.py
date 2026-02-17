# ============================================================
# visualization/hand_3d_combined.py — 3D Hand Mesh + Live Camera
# ============================================================

import cv2
import mediapipe as mp
import numpy as np
import pyvista as pv
from tkinter import messagebox
import time

import config
from core.recognition import landmarks_to_array, cosine_similarity

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def create_hand_mesh_from_landmarks(landmarks, scale=10.0):
    """
    Create a 3D hand mesh with skin covering the skeleton.
    
    Returns both skeleton (lines/spheres) and skin mesh (surface).
    """
    # Convert landmarks to 3D points
    points = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks])
    
    # Center and scale
    center = points[0]
    points = (points - center) * scale
    points[:, 1] *= -1  # Flip Y
    points[:, 2] *= -1  # Flip Z
    
    # Create hand mesh collection
    hand_mesh = pv.MultiBlock()
    
    # Hand connections (MediaPipe topology)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm
    ]
    
    # Add joint spheres
    for i, point in enumerate(points):
        radius = 0.03 * scale if i == 0 else 0.02 * scale
        sphere = pv.Sphere(radius=radius, center=point, phi_resolution=16, theta_resolution=16)
        hand_mesh.append(sphere, f"joint_{i}")
    
    # Add bone cylinders
    for start_idx, end_idx in HAND_CONNECTIONS:
        start = points[start_idx]
        end = points[end_idx]
        line = pv.Line(start, end)
        tube = line.tube(radius=0.012 * scale, n_sides=12)
        hand_mesh.append(tube, f"bone_{start_idx}_{end_idx}")
    
    # Create skin mesh - finger tubes
    finger_groups = [
        [1, 2, 3, 4],     # Thumb
        [5, 6, 7, 8],     # Index
        [9, 10, 11, 12],  # Middle
        [13, 14, 15, 16], # Ring
        [17, 18, 19, 20]  # Pinky
    ]
    
    for finger in finger_groups:
        finger_pts = points[finger]
        # Create spline through finger
        spline = pv.Spline(finger_pts, n_points=20)
        # Tube around spline for skin
        skin_tube = spline.tube(radius=0.015 * scale, n_sides=16)
        hand_mesh.append(skin_tube, f"skin_finger_{finger[0]}")
    
    # Palm mesh (connecting base of fingers)
    palm_indices = [0, 5, 9, 13, 17]
    palm_points = points[palm_indices]
    try:
        palm_cloud = pv.PolyData(palm_points)
        palm_mesh = palm_cloud.delaunay_3d().extract_surface()
        hand_mesh.append(palm_mesh, "skin_palm")
    except:
        pass  # Skip palm if convex hull fails
    
    return hand_mesh


def _run_3d_window(hand_mesh, language, label, shared):
    """
    Runs PyVista 3D window in its own thread.
    shared["running"] = False signals this thread to stop.
    shared["closed"]  = True when this window has closed.
    """
    try:
        plotter = pv.Plotter(
            window_size=[720, 720],
            title=f"{language} — {label}  |  Drag to rotate"
        )
        plotter.set_background("#1a1a1a")

        # ── Add mesh parts ────────────────────────────────
        for i in range(hand_mesh.n_blocks):
            mesh = hand_mesh[i]
            name = hand_mesh.get_block_name(i) or ""

            if "skin" in name:
                plotter.add_mesh(mesh, color=(0.95, 0.85, 0.75),
                                 smooth_shading=True, specular=0.3, opacity=0.9)
            elif "joint" in name:
                idx = int(name.split("_")[1])
                color = (
                    (0.7, 0.3, 0.3) if idx == 0 else
                    (0.9, 0.3, 0.3) if idx <= 4 else
                    (0.3, 0.9, 0.3) if idx <= 8 else
                    (0.3, 0.6, 0.9) if idx <= 12 else
                    (0.9, 0.9, 0.3) if idx <= 16 else
                    (0.9, 0.5, 0.9)
                )
                plotter.add_mesh(mesh, color=color, smooth_shading=True, specular=0.5)
            elif "bone" in name:
                plotter.add_mesh(mesh, color=(0.85, 0.75, 0.65),
                                 smooth_shading=True, specular=0.2, opacity=0.8)

        try:
            plotter.show_grid(color="#444444", opacity=0.3)
        except TypeError:
            plotter.show_grid(color="#444444")

        plotter.add_text(
            f"{language}: {label}\n\n"
            "Left-drag  →  Rotate\n"
            "Right-drag →  Pan\n"
            "Scroll     →  Zoom\n"
            "Q / ESC    →  Close",
            position="upper_left", font_size=11, color="white"
        )

        plotter.camera_position = [(3, 3, 4), (0, 0, 0), (0, 1, 0)]
        plotter.enable_trackball_style()

        # Render loop — process events manually so the window stays
        # responsive while we check the shared["running"] flag.
        plotter.show(auto_close=False, interactive_update=True)

        while shared["running"]:
            plotter.update()          # keeps the window alive + processes mouse
            time.sleep(0.016)         # ~60 fps ceiling

        plotter.close()

    except Exception as e:
        print(f"[3D] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shared["closed"] = True


def show_combined_3d_camera_view(reference_samples: dict, language: str, label: str):
    """
    Show BOTH windows simultaneously:
      • LEFT  – PyVista 3D hand (draggable, runs in background thread)
      • RIGHT – OpenCV live camera with accuracy meter (runs on main thread)

    Closing either window stops both.
    """
    if label not in reference_samples:
        messagebox.showwarning("No Data", f"No reference sample found for '{label}'")
        return

    landmarks_ref = reference_samples[label]
    ref_vector    = np.array(landmarks_to_array(landmarks_ref)).flatten()
    lang_color    = config.COLOR_ASL if language == "ASL" else config.COLOR_FSL

    # ── Build mesh ────────────────────────────────────────
    print(f"[INFO] Building 3D hand mesh for {language}: {label} ...")
    try:
        hand_mesh = create_hand_mesh_from_landmarks(landmarks_ref, scale=10.0)
    except Exception as e:
        messagebox.showerror("3D Error", f"Failed to build hand mesh:\n{e}")
        return

    # Shared state between threads
    shared = {"running": True, "closed": False}

    # ── Launch 3D window in background thread ─────────────
    import threading
    t3d = threading.Thread(
        target=_run_3d_window,
        args=(hand_mesh, language, label, shared),
        daemon=True
    )
    t3d.start()

    # Give the 3D window time to open before camera starts
    time.sleep(0.8)

    # ── MediaPipe + Camera (main thread) ──────────────────
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=config.MIN_DETECTION_CONF,
        min_tracking_confidence=config.MIN_TRACKING_CONF,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        shared["running"] = False
        messagebox.showerror("Camera Error", "Cannot open camera.")
        hands.close()
        return

    print(f"[INFO] ✓ Both windows open")
    print(f"[INFO]   3D  window → drag to rotate")
    print(f"[INFO]   Camera     → shows accuracy")
    print(f"[INFO]   ESC in camera or close 3D to exit\n")

    try:
        while shared["running"]:

            # Stop if 3D window was closed by user
            if shared["closed"]:
                print("[INFO] 3D window closed — exiting")
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame      = cv2.flip(frame, 1)
            frame      = cv2.resize(frame, (640, 480))
            rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result     = hands.process(rgb_frame)

            accuracy_pct = 0.0

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                )

                lm_list     = [{"x": p.x, "y": p.y, "z": p.z} for p in lm.landmark]
                live_vector = np.array(landmarks_to_array(lm_list)).flatten()
                similarity  = cosine_similarity(live_vector, ref_vector)
                accuracy_pct = max(0.0, min(100.0, similarity * 100))

            # ── Accuracy meter ────────────────────────────
            mx, my, mw, mh = 10, 10, 300, 50
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (30, 30, 30), -1)

            fill = int(mw * accuracy_pct / 100)
            bar_color = (
                (0, 255, 0)   if accuracy_pct >= 80 else
                (0, 165, 255) if accuracy_pct >= 50 else
                (0, 0, 255)
            )
            cv2.rectangle(frame, (mx, my), (mx + fill, my + mh), bar_color, -1)
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (200, 200, 200), 2)
            cv2.putText(frame, f"Accuracy: {accuracy_pct:.1f}%",
                        (mx + 10, my + 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ── Feedback text ─────────────────────────────
            feedback, fb_color = (
                ("Perfect!",        (0, 255,   0)) if accuracy_pct >= 85 else
                ("Good!",           (0, 200, 255)) if accuracy_pct >= 70 else
                ("Keep trying...",  (0, 165, 255)) if accuracy_pct >= 50 else
                ("Adjust your hand",(0,   0, 255))
            )
            cv2.putText(frame, feedback,
                        (mx, my + mh + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, fb_color, 2)

            # ── Target label ──────────────────────────────
            cv2.rectangle(frame,
                          (0, frame.shape[0] - 40),
                          (220, frame.shape[0]), (20, 20, 20), -1)
            cv2.putText(frame, f"Target: {label}",
                        (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, lang_color, 2)

            cv2.putText(frame, "ESC to exit",
                        (frame.shape[1] - 115, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

            cv2.imshow(f"Your Hand  –  {language}: {label}", frame)

            if cv2.waitKey(1) & 0xFF == 27:   # ESC
                print("[INFO] ESC pressed — closing both windows")
                break

    finally:
        shared["running"] = False
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        t3d.join(timeout=3.0)
        print(f"[INFO] Session ended for '{label}'\n")