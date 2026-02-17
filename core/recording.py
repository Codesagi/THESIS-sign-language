# ============================================================
# core/recording.py — Recording Session Management
# ============================================================

import os
import cv2
import time
from datetime import datetime
from typing import Dict, Optional
import config


class RecordingSession:
    """Manages a single recording session."""
    
    def __init__(
        self,
        db_client,
        language: str,
        label: str,
        sign_id: str,
        session_id: str,
        video_path: str
    ):
        self.db_client = db_client
        self.language = language
        self.label = label
        self.sign_id = sign_id
        self.session_id = session_id
        self.video_path = video_path
        
        self.active = True
        self.frame_count = 0
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.start_time = time.time()
    
    def record_frame(self, frame, hand_landmarks, handedness_label: str):
        """Save one frame's landmarks and video."""
        from .db import insert_landmark_sample
        
        self.frame_count += 1
        
        # Extract landmarks
        lm_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]
        
        # Save to database
        insert_landmark_sample(
            self.db_client,
            sign_id=self.sign_id,
            language=self.language,
            label=self.label,
            landmarks=lm_list,
            source="live_recording",
            hand_side=handedness_label,
            source_file=self.video_path,
            frame_index=self.frame_count,
        )
        
        # Save video frame
        if self.video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 30.0, (w, h))
        
        self.video_writer.write(frame)
    
    def stop(self):
        """Stop recording and update database."""
        from .db import update_recording_session
        
        duration = time.time() - self.start_time
        
        if self.video_writer:
            self.video_writer.release()
        
        update_recording_session(
            self.db_client,
            session_id=self.session_id,
            frame_count=self.frame_count,
            duration_sec=round(duration, 2),
            video_filename=self.video_path,
        )
        
        self.active = False
        return self.frame_count
    
    def should_auto_stop(self) -> bool:
        """Check if we've recorded enough frames."""
        return self.frame_count >= config.FRAMES_PER_SAMPLE


def create_recording_session(db_client, language: str, label: str, sign_id: str) -> RecordingSession:
    """Create and initialize a new recording session."""
    from .db import create_recording_session as db_create_session
    
    os.makedirs(config.VIDEO_SAVE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(config.VIDEO_SAVE_DIR, f"{language}_{label}_{ts}.avi")
    
    session_id = db_create_session(
        db_client,
        sign_id=sign_id,
        language=language,
        label=label,
        session_type="static",
        video_filename=video_path,
    )
    
    return RecordingSession(
        db_client=db_client,
        language=language,
        label=label,
        sign_id=sign_id,
        session_id=session_id,
        video_path=video_path
    )