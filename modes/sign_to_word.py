# ============================================================
# modes/sign_to_word.py — Live Sign Recognition Mode
# ============================================================

import cv2
import numpy as np
import threading

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    # Handle MediaPipe import differently if needed
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing

import config
from core.db import fetch_all_reference_vectors, get_or_create_sign
from core.recognition import normalize_landmarks, rank_predictions
from core.recording import create_recording_session
from core.caption import CaptionTracker
from ui.overlays import (
    draw_prediction_bars, draw_caption_history, draw_recording_indicator,
    draw_status, draw_controls_hint, draw_language_badge, set_status
)
from ui.dialogs import ask_recording_label

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Try to import 3D viewer (optional)
try:
    from visualization.hand_3d import show_3d_hand_interactive
    HAS_3D = True
except ImportError:
    HAS_3D = False


def run_sign_to_word_mode(db_client, language: str):
    """Live camera recognition mode with caption history."""
    
    # Initialize MediaPipe
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONF,
        min_tracking_confidence=config.MIN_TRACKING_CONF,
    )
    
    # State
    reference_bank = {}
    reference_samples = {}  # For 3D viewer
    predictions = []
    caption_tracker = CaptionTracker(hold_duration=2.0)
    recording_session = None
    
    # ── ML engine (lazy, cached — never reloads between frames) ──
    from pathlib import Path
    from core.recognition import InferenceEngine

    _cnn_path = str(Path("models") / f"cnn_{language.lower()}.h5")
    _rf_path  = str(Path("models") / f"rf_{language.lower()}.pkl")

    engine        = InferenceEngine.get(_cnn_path, _rf_path)
    use_ml_models = engine.is_ready

    if use_ml_models:
        print(f"[INFO] ✓ InferenceEngine ready  ({language}  CNN + RF)")
    else:
        print(f"[INFO] ML models not found — cosine-similarity fallback.")
        print(f"[INFO] Train: python -m tools.train_models --language {language}")
    
    # Load reference data in background
    def load_references():
        nonlocal reference_bank, reference_samples
        
        if not use_ml_models:
            set_status(f"Loading {language} reference data...")
            try:
                reference_bank = fetch_all_reference_vectors(db_client, language)
                n = len(reference_bank)
                set_status(f"✓ Loaded {n} signs" if n > 0 else "⚠ No samples found")
            except Exception as e:
                set_status(f"DB error: {e}")
        
        # Load samples for 3D viewer
        try:
            resp = db_client.table("landmark_samples").select("label, landmarks_json").eq("language", language).limit(1000).execute()
            samples_by_label = {}
            for row in resp.data or []:
                label = row["label"]
                if label not in samples_by_label:
                    samples_by_label[label] = row["landmarks_json"]
            reference_samples = samples_by_label
        except Exception as e:
            print(f"[WARN] Could not load samples for 3D viewer: {e}")
    
    t = threading.Thread(target=load_references, daemon=True)
    t.start()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return
    
    lang_color = config.COLOR_ASL if language == "ASL" else config.COLOR_FSL
    
    model_info = "CNN + Random Forest" if use_ml_models else "Cosine Similarity"
    print(f"[INFO] Sign-to-Word mode: {language} ({model_info})")
    print(f"[INFO] Hold a sign for 2s to add to caption")
    print(f"[INFO] Press R to record, 3 for 3D view, ESC to exit")
    
    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        camera_view = frame.copy()
        skeleton_view = np.zeros_like(frame)
        
        current_predictions = []
        
        # Process hands
        if result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                hand_side = "Unknown"
                if result.multi_handedness and idx < len(result.multi_handedness):
                    hand_side = result.multi_handedness[idx].classification[0].label
                
                # Draw landmarks
                mp_drawing.draw_landmarks(camera_view, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    skeleton_view, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
                )
                
                # Predict — single normalize call inside engine
                if use_ml_models:
                    preds = engine.predict_top_k(
                        hand_landmarks.landmark,
                        k=config.TOP_N_PREDICTIONS,
                        margin_threshold=0.10,
                    )
                elif reference_bank:
                    from core.recognition import normalize_landmarks, rank_predictions
                    live_vec = normalize_landmarks(hand_landmarks.landmark).ravel()
                    preds = rank_predictions(live_vec, reference_bank, config.TOP_N_PREDICTIONS)
                else:
                    preds = []
                current_predictions = [(l, p) for l, p in preds if p >= config.MIN_CONFIDENCE]
                
                # Recording
                if recording_session and recording_session.active:
                    recording_session.record_frame(camera_view.copy(), hand_landmarks, hand_side)
                    if recording_session.should_auto_stop():
                        frames = recording_session.stop()
                        set_status(f"✓ Saved {frames} frames for [{recording_session.label}]")
                        recording_session = None
                        # Reload references
                        t = threading.Thread(target=load_references, daemon=True)
                        t.start()
        
        predictions = current_predictions
        
        # Update caption
        top_pred = predictions[0][0] if predictions else ""
        caption_tracker.update(top_pred)
        
        # Draw UI
        draw_caption_history(camera_view, caption_tracker.get_text())
        draw_caption_history(skeleton_view, caption_tracker.get_text())
        
        if predictions:
            draw_prediction_bars(camera_view, predictions, language)
            draw_prediction_bars(skeleton_view, predictions, language)
        
        draw_language_badge(camera_view, language)
        
        if recording_session and recording_session.active:
            draw_recording_indicator(camera_view, recording_session.label, 
                                     recording_session.frame_count, config.FRAMES_PER_SAMPLE)
            draw_recording_indicator(skeleton_view, recording_session.label,
                                     recording_session.frame_count, config.FRAMES_PER_SAMPLE)
        
        draw_status(camera_view)
        
        hints = ["R: Record", "ESC: Exit"]
        if HAS_3D:
            hints.insert(0, "3: 3D View")
        draw_controls_hint(camera_view, hints)
        
        cv2.imshow("Camera View", camera_view)
        cv2.imshow("Skeleton View", skeleton_view)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            if recording_session:
                recording_session.stop()
            break
        
        elif key == ord("r") or key == ord("R"):
            if recording_session and recording_session.active:
                recording_session.stop()
                set_status("Recording stopped")
            else:
                label = ask_recording_label(language)
                if label:
                    sign_id = get_or_create_sign(db_client, language, label)
                    recording_session = create_recording_session(db_client, language, label, sign_id)
                    set_status(f"● Recording [{label}] — {config.FRAMES_PER_SAMPLE} frames")
        
        elif key == ord("3") and HAS_3D:
            if predictions and predictions[0][0] in reference_samples:
                top_label = predictions[0][0]
                landmarks = reference_samples[top_label]
                print(f"[3D] Opening for: {top_label}")
                t = threading.Thread(
                    target=show_3d_hand_interactive,
                    args=(landmarks,),
                    kwargs={"language": language, "label": top_label},
                    daemon=True
                )
                t.start()
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    if caption_tracker.history:
        print(f"\n[SESSION] Final spelling: {caption_tracker.get_text()}")