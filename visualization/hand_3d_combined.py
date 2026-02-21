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


# ============================================================
# LIVE 3D SKELETON VIEWER  (for Sign-to-Word mode)
# ============================================================

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(19,20),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

FINGER_COLORS = [
    (0.85, 0.35, 0.35),   # 0  wrist    — red
    (0.95, 0.40, 0.40),   # 1- thumb   — bright red
    (0.95, 0.40, 0.40),
    (0.95, 0.40, 0.40),
    (0.95, 0.40, 0.40),
    (0.35, 0.90, 0.35),   # 5- index   — green
    (0.35, 0.90, 0.35),
    (0.35, 0.90, 0.35),
    (0.35, 0.90, 0.35),
    (0.35, 0.60, 0.95),   # 9- middle  — blue
    (0.35, 0.60, 0.95),
    (0.35, 0.60, 0.95),
    (0.35, 0.60, 0.95),
    (0.95, 0.90, 0.30),   # 13- ring   — yellow
    (0.95, 0.90, 0.30),
    (0.95, 0.90, 0.30),
    (0.95, 0.90, 0.30),
    (0.90, 0.45, 0.90),   # 17- pinky  — pink
    (0.90, 0.45, 0.90),
    (0.90, 0.45, 0.90),
    (0.90, 0.45, 0.90),
]


def _landmarks_to_3d_points(landmarks, scale: float = 10.0) -> np.ndarray:
    """
    Convert a list of 21 landmark objects or dicts → (21, 3) world-space
    points suitable for PyVista (Y and Z flipped to match screen coords).
    """
    if hasattr(landmarks[0], "x"):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    else:
        pts = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks], dtype=np.float32)

    # Wrist to origin, palm-scale normalise
    pts -= pts[0]
    palm = float(np.linalg.norm(pts[9]))
    if palm > 1e-6:
        pts /= palm

    pts *= scale
    pts[:, 1] *= -1   # flip Y (screen → 3-D)
    pts[:, 2] *= -1   # flip Z

    return pts


def _build_live_skeleton(pts: np.ndarray, scale: float = 10.0):
    """
    Build a PyVista MultiBlock skeleton (joints + bones) from 21 points.
    Returns the MultiBlock and a flat list of sphere/tube actors in
    insertion order so we can update coordinates later.
    """
    mb = pv.MultiBlock()

    # joints
    for i, p in enumerate(pts):
        r = 0.025 * scale if i == 0 else 0.018 * scale
        s = pv.Sphere(radius=r, center=p, phi_resolution=10, theta_resolution=10)
        mb.append(s, f"j{i}")

    # bones
    for a, b in HAND_CONNECTIONS:
        line = pv.Line(pts[a], pts[b])
        tube = line.tube(radius=0.009 * scale, n_sides=8)
        mb.append(tube, f"b{a}_{b}")

    return mb


def _run_live_3d_skeleton(shared: dict, scale: float = 10.0):
    """
    Background thread that owns the PyVista window.

    shared["pts"]     — (21,3) ndarray, written by camera thread each frame
    shared["running"] — set False to close
    shared["closed"]  — set True when window exits
    shared["label"]   — current prediction label string
    shared["conf"]    — current confidence float
    """
    try:
        # Neutral hand for initial render (flat, spread fingers)
        init_pts = np.zeros((21, 3), dtype=np.float32)
        for i in range(5):
            for j in range(4):
                idx = 1 + i * 4 + j
                init_pts[idx] = [(-2 + i) * 0.4 * scale,
                                  -(j + 1) * 0.3 * scale,
                                  0]

        plotter = pv.Plotter(
            window_size=[640, 640],
            title="Live 3D Skeleton  |  Drag to rotate"
        )
        plotter.set_background("#111118")

        # Add skeleton actors — store references for in-place updates
        actors   = []
        mb       = _build_live_skeleton(init_pts, scale)

        for i in range(mb.n_blocks):
            mesh  = mb[i]
            name  = mb.get_block_name(i) or ""
            is_j  = name.startswith("j")

            if is_j:
                jidx  = int(name[1:])
                color = FINGER_COLORS[jidx]
                a = plotter.add_mesh(mesh, color=color,
                                     smooth_shading=True, specular=0.6,
                                     lighting=True)
            else:
                a = plotter.add_mesh(mesh, color=(0.70, 0.70, 0.70),
                                     smooth_shading=True, opacity=0.85)
            actors.append((name, mesh, a))

        # Prediction overlay text (top-left)
        txt_actor = plotter.add_text(
            "Detecting…",
            position="upper_left", font_size=14, color="white"
        )

        try:
            plotter.show_grid(color="#333344", opacity=0.25)
        except TypeError:
            plotter.show_grid(color="#333344")

        plotter.camera_position = [(0, -2, 6), (0, -2, 0), (0, 1, 0)]
        plotter.enable_trackball_style()
        plotter.show(auto_close=False, interactive_update=True)

        last_pts = init_pts.copy()

        while shared["running"]:
            new_pts = shared.get("pts")

            if new_pts is not None and not np.array_equal(new_pts, last_pts):
                # Rebuild skeleton with new joint positions
                # We update point coordinates in-place on each mesh
                ji = 0   # joint index
                bi = 0   # bone index

                for name, mesh, actor in actors:
                    if name.startswith("j"):
                        # Sphere — move by translating all its points
                        jidx    = int(name[1:])
                        delta   = new_pts[jidx] - last_pts[jidx]
                        mesh.points += delta

                    else:
                        # Tube — rebuild from new endpoints
                        parts   = name[1:].split("_")
                        a_idx, b_idx = int(parts[0]), int(parts[1])
                        new_line = pv.Line(new_pts[a_idx], new_pts[b_idx])
                        new_tube = new_line.tube(radius=0.009 * scale, n_sides=8)
                        mesh.copy_from(new_tube)

                last_pts = new_pts.copy()

            # Update prediction text
            label = shared.get("label", "")
            conf  = shared.get("conf",  0.0)
            if label:
                txt_actor.SetInput(
                    f"Sign: {label}  ({conf:.1f}%)\n"
                    "Drag to rotate  •  Scroll to zoom"
                )
            else:
                txt_actor.SetInput("No hand detected\nDrag to rotate")

            plotter.update()
            time.sleep(0.033)   # ~30 fps

        plotter.close()

    except Exception as e:
        import traceback
        print(f"[3D live] Error: {e}")
        traceback.print_exc()
    finally:
        shared["closed"] = True


def start_live_3d_skeleton(scale: float = 10.0) -> dict:
    """
    Launch the live 3D skeleton window in a background thread.

    Returns the shared-state dict.  The caller (sign_to_word camera loop)
    writes to it every frame:

        shared["pts"]   = <(21,3) ndarray>   # current landmark positions
        shared["label"] = "A"                 # top prediction
        shared["conf"]  = 87.3               # confidence %

    Call  shared["running"] = False  to close the window.
    """
    import threading

    shared = {
        "pts":     None,
        "label":   "",
        "conf":    0.0,
        "running": True,
        "closed":  False,
    }

    t = threading.Thread(
        target=_run_live_3d_skeleton,
        args=(shared, scale),
        daemon=True,
    )
    t.start()
    time.sleep(0.6)   # let the window initialise before camera starts
    return shared