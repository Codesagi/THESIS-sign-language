# ============================================================
# core/caption.py — Caption History Tracking
# ============================================================

import time


class CaptionTracker:
    """Tracks caption history for spelling mode."""
    
    def __init__(self, hold_duration: float = 2.0):
        self.history = []
        self.last_prediction = ""
        self.hold_start = 0.0
        self.hold_duration = hold_duration
    
    def update(self, current_prediction: str) -> bool:
        """
        Update with current top prediction.
        Returns True if a new letter was added to history.
        """
        if not current_prediction:
            self.last_prediction = ""
            self.hold_start = 0.0
            return False
        
        # Prediction changed - reset timer
        if current_prediction != self.last_prediction:
            self.last_prediction = current_prediction
            self.hold_start = time.time()
            return False
        
        # Same prediction - check hold time
        hold_time = time.time() - self.hold_start
        if hold_time >= self.hold_duration:
            # Add to history if not duplicate
            if not self.history or self.history[-1] != current_prediction:
                self.history.append(current_prediction)
                print(f"[CAPTION] Added: {current_prediction} → {self.get_text()}")
                # Reset timer to avoid repeated additions
                self.hold_start = time.time()
                return True
        
        return False
    
    def get_text(self) -> str:
        """Get caption as space-separated string."""
        return " ".join(self.history)
    
    def clear(self):
        """Clear caption history."""
        self.history = []
        self.last_prediction = ""
        self.hold_start = 0.0
    
    def get_progress(self) -> float:
        """Get hold progress as 0.0 to 1.0."""
        if not self.last_prediction:
            return 0.0
        elapsed = time.time() - self.hold_start
        return min(1.0, elapsed / self.hold_duration)