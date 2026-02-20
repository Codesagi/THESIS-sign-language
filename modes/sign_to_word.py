# ============================================================
# modes/sign_to_word.py — Live Sign Recognition Mode
# OPTIMIZED: Single view, frame skip, no redundant copies
# ============================================================

import cv2
import numpy as np
import threading

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing

import config
from core.db import fetch_all_reference_vectors, get_or_create_sign
from core.recording import create_recording_session
from core.caption import CaptionTracker
from ui.overlays import (
    draw_prediction_bars, draw_caption_history, draw_recording_indicator,
    draw_status, draw_controls_hint, draw_language_badge, set_status
)
from ui.dialogs import ask_recording_label

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def run_sign_to_word_mode(db_client, language: str):
    """Live camera recognition mode - OPTIMIZED."""
    
    # Initialize MediaPipe
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=config.MAX_NUM_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONF,
        min_tracking_confidence=config.MIN_TRACKING_CONF,
    )
    
    # State
    reference_bank = {}
    predictions = []
    caption_tracker = CaptionTracker(hold_duration=2.0)
    recording_session = None
    
    # ── Optimized ML inference (frame skip + lazy load) ──────
    from pathlib import Path
    from core.inference import CachedPredictor

    _cnn_path = str(Path("models") / f"cnn_{language.lower()}.h5")
    _rf_path  = str(Path("models") / f"rf_{language.lower()}.pkl")

    predictor = CachedPredictor(
        _cnn_path, _rf_path,
        frame_skip=3,  # KEY OPTIMIZATION: predict every 3rd frame
        cnn_weight=0.6,
        rf_weight=0.4,
    )
    use_ml_models = predictor.is_ready

    if use_ml_models:
        print(f"[INFO] ✓ CachedPredictor ready  ({language}  frame_skip=3)")
    else:
        print(f"[INFO] ML models not found — cosine-similarity fallback.")
        print(f"[INFO] Train: python -m tools.train_models --language {language}")
    
    # Load reference data in background (only if no ML models)
    def load_references():
        nonlocal reference_bank
        
        if not use_ml_models:
            set_status(f"Loading {language} reference data...")
            try:
                reference_bank = fetch_all_reference_vectors(db_client, language)
                n = len(reference_bank)
                set_status(f"✓ Loaded {n} signs" if n > 0 else "⚠ No samples found")
            except Exception as e:
                set_status(f"DB error: {e}")
    
    t = threading.Thread(target=load_references, daemon=True)
    t.start()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return
    
    lang_color = config.COLOR_ASL if language == "ASL" else config.COLOR_FSL
    
    model_info = "CNN + Random Forest (frame_skip=3)" if use_ml_models else "Cosine Similarity"
    print(f"[INFO] Sign-to-Word mode: {language} ({model_info})")
    print(f"[INFO] Hold a sign for 2s to add to caption")
    print(f"[INFO] Press R to record, ESC to exit")
    
    # Main loop - OPTIMIZED
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Process every frame with MediaPipe (fast)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        current_predictions = []
        
        # Process hands
        if result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                hand_side = "Unknown"
                if result.multi_handedness and idx < len(result.multi_handedness):
                    hand_side = result.multi_handedness[idx].classification[0].label
                
                # Draw landmarks ONCE (removed dual skeleton view - was causing lag)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Predict with frame skip (every 3rd frame)
                if use_ml_models:
                    preds = predictor.predict(
                        hand_landmarks.landmark,
                        frame_idx=frame_count,
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

                # Recording (no extra frame copy unless actively recording)
                if recording_session and recording_session.active:
                    recording_session.record_frame(frame, hand_landmarks, hand_side)
                    if recording_session.should_auto_stop():
                        frames = recording_session.stop()
                        set_status(f"✓ Saved {frames} frames for [{recording_session.label}]")
                        recording_session = None
                        t = threading.Thread(target=load_references, daemon=True)
                        t.start()
        
        predictions = current_predictions

        # Update caption
        top_pred = predictions[0][0] if predictions else ""
        caption_tracker.update(top_pred)
        
        # Draw UI
        draw_caption_history(frame, caption_tracker.get_text())
        
        if predictions:
            draw_prediction_bars(frame, predictions, language)
        
        if recording_session and recording_session.active:
            draw_recording_indicator(frame, recording_session.frames_captured, config.FRAMES_PER_SAMPLE)
        
        draw_status(frame)
        draw_controls_hint(frame, "R: Record  |  ESC: Exit")
        draw_language_badge(frame, language, lang_color)
        
        # Display single view (removed dual view - major performance gain)
        cv2.imshow("Sign-to-Word", frame)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            if recording_session:
                recording_session.stop()
            break
        
        elif key == ord("r") or key == ord("R"):
            if recording_session and recording_session.active:
                frames = recording_session.stop()
                set_status(f"✓ Saved {frames} frames for [{recording_session.label}]")
                recording_session = None
            else:
                label = ask_recording_label(db_client, language)
                if label:
                    sign_id = get_or_create_sign(db_client, language, label)
                    recording_session = create_recording_session(
                        db_client, sign_id, language, label,
                        max_frames=config.FRAMES_PER_SAMPLE
                    )
                    set_status(f"Recording [{label}]...")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"[INFO] Sign-to-Word mode ended.")