SUPABASE_URL = "a"
SUPABASE_KEY = "a"

TOP_N_PREDICTIONS   = 5      # How many top matches to show on screen
MIN_CONFIDENCE      = 0.0    # Minimum similarity % to show (0.0 = show all)
FRAMES_PER_SAMPLE   = 30     # Frames to capture per label in recording mode
VIDEO_SAVE_DIR      = "recordings"   # Local folder for raw video clips

# ── MediaPipe ────────────────────────────────────────────────
MAX_NUM_HANDS           = 2
MIN_DETECTION_CONF      = 0.7
MIN_TRACKING_CONF       = 0.7

# ── UI Colors (BGR) ──────────────────────────────────────────
COLOR_ASL       = (0, 200, 255)    # Orange-yellow for ASL
COLOR_FSL       = (255, 180, 0)    # Blue for FSL
COLOR_BAR_BG    = (50, 50, 50)
COLOR_BAR_FILL  = (0, 220, 100)
COLOR_TEXT      = (255, 255, 255)
COLOR_RECORD    = (0, 0, 220)  