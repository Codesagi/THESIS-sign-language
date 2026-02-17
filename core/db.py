# ============================================================
# core/db.py — Supabase database access layer
# ============================================================

import numpy as np
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY
from .recognition import flatten_landmarks, cosine_similarity, rank_predictions


def get_client() -> Client:
    """Return an authenticated Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ── Signs ─────────────────────────────────────────────────────

def get_or_create_sign(client: Client, language: str, label: str, is_gesture: bool = False) -> str:
    """Return sign UUID, creating it if it doesn't exist yet."""
    # Use .execute() without maybe_single() — it always returns a valid response object
    resp = (
        client.table("signs")
        .select("id")
        .eq("language", language)
        .eq("label", label)
        .limit(1)
        .execute()
    )
    if resp.data and len(resp.data) > 0:
        return resp.data[0]["id"]

    # Not found — insert it
    new = (
        client.table("signs")
        .insert({"language": language, "label": label, "is_gesture": is_gesture})
        .execute()
    )
    return new.data[0]["id"]


def get_all_signs(client: Client, language: str) -> list[dict]:
    """Return all sign rows for a language."""
    resp = (
        client.table("signs")
        .select("*")
        .eq("language", language)
        .execute()
    )
    return resp.data or []


# ── Landmark Samples ──────────────────────────────────────────

def insert_landmark_sample(
    client: Client,
    sign_id: str,
    language: str,
    label: str,
    landmarks: list[dict],          # [{x, y, z}, ...] × 21
    source: str = "live_recording",
    hand_side: str = "Unknown",
    confidence: float = None,
    source_file: str = None,
    frame_index: int = None,
) -> dict:
    """Insert one landmark sample into Supabase."""
    flat_vector = flatten_landmarks(landmarks)

    # pgvector requires the vector as a string like "[0.1,0.2,...]"
    # Passing a plain Python list causes a 500 from PostgREST
    vector_str = "[" + ",".join(f"{v:.6f}" for v in flat_vector) + "]"

    row = {
        "sign_id":          sign_id,
        "language":         language,
        "label":            label,
        "source":           source,
        "landmarks_json":   landmarks,
        "landmark_vector":  vector_str,
        "hand_side":        hand_side,
        "confidence":       confidence,
        "source_file":      source_file,
        "frame_index":      frame_index,
    }
    resp = client.table("landmark_samples").insert(row).execute()
    return resp.data[0] if resp.data else {}


def fetch_all_reference_vectors(client: Client, language: str) -> dict[str, np.ndarray]:
    """
    Fetch all landmark vectors for a language.
    Returns {label: np.ndarray of shape (N, 63)} — averaged per label.
    Used to build the in-memory reference bank for live comparison.
    
    Uses pagination to fetch all rows (Supabase default limit is 1000 per request).
    """
    all_rows = []
    page_size = 1000
    offset = 0
    
    while True:
        resp = (
            client.table("landmark_samples")
            .select("label, landmark_vector")
            .eq("language", language)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        
        rows = resp.data or []
        if not rows:
            break
            
        all_rows.extend(rows)
        
        # If we got fewer than page_size, we've reached the end
        if len(rows) < page_size:
            break
            
        offset += page_size

    label_vectors: dict[str, list] = {}
    for row in all_rows:
        vec = row.get("landmark_vector")
        if vec is None:
            continue
        # Supabase returns the vector as a string "[0.1,0.2,...]" — parse it
        if isinstance(vec, str):
            vec = [float(v) for v in vec.strip("[]").split(",")]
        label = row["label"]
        label_vectors.setdefault(label, []).append(vec)

    # Average all samples per label into one reference vector
    averaged: dict[str, np.ndarray] = {}
    for label, vecs in label_vectors.items():
        averaged[label] = np.mean(np.array(vecs, dtype=np.float32), axis=0)

    return averaged


# ── Recording Sessions ────────────────────────────────────────

def create_recording_session(
    client: Client,
    sign_id: str,
    language: str,
    label: str,
    session_type: str = "static",
    video_filename: str = None,
    recorded_by: str = "developer",
) -> str:
    """Create a recording session and return its UUID."""
    resp = (
        client.table("recording_sessions")
        .insert({
            "sign_id":        sign_id,
            "language":       language,
            "label":          label,
            "session_type":   session_type,
            "video_filename": video_filename,
            "recorded_by":    recorded_by,
        })
        .execute()
    )
    return resp.data[0]["id"] if resp.data else None


def update_recording_session(
    client: Client,
    session_id: str,
    frame_count: int,
    duration_sec: float,
    video_filename: str = None,
):
    """Update frame count, duration, and optional video path after recording."""
    update = {"frame_count": frame_count, "duration_sec": duration_sec}
    if video_filename:
        update["video_filename"] = video_filename
    client.table("recording_sessions").update(update).eq("id", session_id).execute()


# ── Gesture Sequences ─────────────────────────────────────────

def insert_gesture_frame(
    client: Client,
    session_id: str,
    sign_id: str,
    language: str,
    label: str,
    frame_index: int,
    landmarks: list[dict],
    timestamp_ms: float = None,
):
    """Insert one frame of a gesture sequence."""
    flat_vector = flatten_landmarks(landmarks)
    vector_str = "[" + ",".join(f"{v:.6f}" for v in flat_vector) + "]"
    client.table("gesture_sequences").insert({
        "session_id":       session_id,
        "sign_id":          sign_id,
        "language":         language,
        "label":            label,
        "frame_index":      frame_index,
        "landmarks_json":   landmarks,
        "landmark_vector":  vector_str,
        "timestamp_ms":     timestamp_ms,
    }).execute()


# ── Utilities ─────────────────────────────────────────────────