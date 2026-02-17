# ============================================================
# ui/overlays.py — On-screen UI Overlays
# ============================================================

import cv2
import time
import numpy as np
import config


# ── Status Message ────────────────────────────────────────────

_STATUS_MSG = ""
_STATUS_EXPIRE = 0.0


def set_status(msg: str, duration: float = 3.0):
    """Set status message shown at bottom of frame."""
    global _STATUS_MSG, _STATUS_EXPIRE
    _STATUS_MSG = msg
    _STATUS_EXPIRE = time.time() + duration


def draw_status(frame: np.ndarray):
    """Draw status message if active."""
    if _STATUS_MSG and time.time() < _STATUS_EXPIRE:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, _STATUS_MSG,
                    (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    config.COLOR_TEXT, 1, cv2.LINE_AA)


# ── Prediction Bars ────────────────────────────────────────────

def draw_prediction_bars(frame: np.ndarray, predictions: list, language: str):
    """Draw top-N prediction bars in bottom-left corner."""
    if not predictions:
        return

    bar_w, bar_h = 250, 18
    pad_x, pad_y = 10, 10
    row_h = bar_h + 6

    total_h = len(predictions) * row_h + 30
    start_y = frame.shape[0] - total_h - pad_y

    lang_color = config.COLOR_ASL if language == "ASL" else config.COLOR_FSL

    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (pad_x - 4, start_y - 20),
                  (pad_x + bar_w + 80, start_y + total_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"{language} Predictions",
                (pad_x + 2, start_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, lang_color, 1, cv2.LINE_AA)

    for i, (label, pct) in enumerate(predictions):
        y = start_y + i * row_h

        cv2.rectangle(frame,
                      (pad_x, y),
                      (pad_x + bar_w, y + bar_h),
                      config.COLOR_BAR_BG, -1)

        fill = int(bar_w * pct / 100)
        bar_color = lang_color if i == 0 else config.COLOR_BAR_FILL
        cv2.rectangle(frame,
                      (pad_x, y),
                      (pad_x + fill, y + bar_h),
                      bar_color, -1)

        cv2.putText(frame, label,
                    (pad_x + 4, y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    config.COLOR_TEXT, 1, cv2.LINE_AA)

        cv2.putText(frame, f"{pct:.1f}%",
                    (pad_x + bar_w + 6, y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    config.COLOR_TEXT, 1, cv2.LINE_AA)


# ── Caption History ────────────────────────────────────────────

def draw_caption_history(frame: np.ndarray, caption_text: str):
    """Draw caption history at top center."""
    if not caption_text:
        return

    h, w = frame.shape[:2]

    panel_h = 60
    cv2.rectangle(frame, (0, 0), (w, panel_h), (20, 20, 20), -1)

    font_scale = 1.2
    thickness = 2
    text_size = cv2.getTextSize(caption_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = 40

    cv2.putText(frame, caption_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.putText(frame, "Spelling:",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 180, 180), 1, cv2.LINE_AA)


# ── Recording Indicator ────────────────────────────────────────

def draw_recording_indicator(frame: np.ndarray, label: str, current_frame: int, total_frames: int):
    """Draw flashing REC indicator + progress bar."""
    h, w = frame.shape[:2]
    
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(frame, (w - 20, 20), 8, config.COLOR_RECORD, -1)
    
    cv2.putText(frame,
                f"REC [{label}]  {current_frame}/{total_frames}",
                (w - 280, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                config.COLOR_RECORD, 1, cv2.LINE_AA)

    prog = int((w - 20) * current_frame / max(total_frames, 1))
    cv2.rectangle(frame, (10, h - 8), (w - 10, h - 2), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, h - 8), (10 + prog, h - 2), config.COLOR_RECORD, -1)


# ── Controls Hint ──────────────────────────────────────────────

def draw_controls_hint(frame: np.ndarray, hints: list[str]):
    """Draw control hints in top-right corner."""
    h, w = frame.shape[:2]
    for i, hint in enumerate(hints):
        cv2.putText(frame, hint,
                    (w - 110, 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (160, 160, 160), 1, cv2.LINE_AA)


# ── Language Badge ─────────────────────────────────────────────

def draw_language_badge(frame: np.ndarray, language: str):
    """Draw language badge in top-left corner."""
    lang_color = config.COLOR_ASL if language == "ASL" else config.COLOR_FSL
    cv2.rectangle(frame, (0, 0), (100, 28), (20, 20, 20), -1)
    cv2.putText(frame, language,
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                lang_color, 2, cv2.LINE_AA)