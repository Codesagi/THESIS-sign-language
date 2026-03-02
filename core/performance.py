# ============================================================
# core/performance.py — Performance Optimization Utilities
#
# Lazy imports, frame throttling, memory management
# ============================================================

import time
from functools import lru_cache
from typing import Any, Callable


class LazyImporter:
    """
    Lazy import manager - only imports modules when first accessed.
    
    Speeds up startup by deferring heavy imports (TensorFlow, PyVista).
    """
    
    def __init__(self):
        self._cache = {}
    
    def get(self, module_name: str, from_list: list = None):
        """
        Lazy import a module.
        
        Examples:
        --------
        tensorflow = lazy.get('tensorflow')
        pv = lazy.get('pyvista')
        """
        if module_name not in self._cache:
            if from_list:
                module = __import__(module_name, fromlist=from_list)
            else:
                module = __import__(module_name)
            self._cache[module_name] = module
        return self._cache[module_name]


# Global lazy importer
lazy_import = LazyImporter()


class FrameThrottler:
    """
    Throttle frame processing to reduce CPU usage.
    
    Processes every Nth frame, returns cached result for skipped frames.
    """
    
    def __init__(self, skip_frames: int = 3):
        """
        Parameters
        ----------
        skip_frames : int
            Process every Nth frame (default: 3 = 30 FPS -> 10 FPS processing)
        """
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.last_result = None
    
    def should_process(self) -> bool:
        """Check if current frame should be processed."""
        self.frame_count += 1
        return (self.frame_count % self.skip_frames) == 0
    
    def cache_result(self, result: Any):
        """Cache latest result."""
        self.last_result = result
    
    def get_result(self) -> Any:
        """Get cached result (for skipped frames)."""
        return self.last_result
    
    def reset(self):
        """Reset counter and cache."""
        self.frame_count = 0
        self.last_result = None


class PerformanceMonitor:
    """
    Monitor FPS and processing time.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Parameters
        ----------
        window_size : int
            Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = []
        self.last_frame_time = None
    
    def tick(self):
        """Call this at start of each frame."""
        current_time = time.time()
        
        if self.last_frame_time:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
            
            # Keep only recent frames
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)
        
        self.last_frame_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS (averaged over window)."""
        if not self.frame_times:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_avg_frame_time_ms(self) -> float:
        """Get average frame time in milliseconds."""
        if not self.frame_times:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return avg_frame_time * 1000


def optimize_opencv_window():
    """
    Apply OpenCV optimizations.
    
    Call once at startup.
    """
    import cv2
    
    # Use optimized code paths
    cv2.setUseOptimized(True)
    
    # Reduce window update overhead
    cv2.setNumThreads(2)  # Limit threads (more isn't always better)


@lru_cache(maxsize=128)
def cached_color_conversion(color_space: str):
    """
    Cache color conversion constants.
    
    Example:
    --------
    BGR2RGB = cached_color_conversion('BGR2RGB')
    frame_rgb = cv2.cvtColor(frame, BGR2RGB)
    """
    import cv2
    return getattr(cv2, f'COLOR_{color_space}')


class RenderOptimizer:
    """
    Optimize rendering by reducing draw calls.
    """
    
    @staticmethod
    def batch_draw_text(frame, texts: list, positions: list, font_params: dict):
        """
        Draw multiple text elements in one batch.
        
        Parameters
        ----------
        frame : np.ndarray
            Frame to draw on
        texts : list[str]
            Text strings
        positions : list[tuple]
            (x, y) positions
        font_params : dict
            OpenCV font parameters
        """
        import cv2
        
        for text, pos in zip(texts, positions):
            cv2.putText(frame, text, pos, **font_params)
    
    @staticmethod
    def reduce_quality_for_speed(frame, scale: float = 0.75):
        """
        Reduce frame resolution for processing.
        
        Process at lower res, display at full res.
        
        Parameters
        ----------
        frame : np.ndarray
            Input frame
        scale : float
            Scale factor (0.75 = 25% fewer pixels)
        
        Returns
        -------
        resized_frame : np.ndarray
            Smaller frame for processing
        original_size : tuple
            (width, height) of original
        """
        import cv2
        
        h, w = frame.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, (w, h)


# Performance tips for different components
PERFORMANCE_TIPS = {
    'mediapipe': [
        "Use static_image_mode=False for video",
        "Reduce max_num_hands to 1 if only tracking one hand",
        "Lower min_detection_confidence if struggling"
    ],
    'tensorflow': [
        "Use model.predict() in batch mode when possible",
        "Cache model predictions for identical inputs",
        "Consider quantized models for faster inference"
    ],
    'pyvista': [
        "Render off-screen at lower resolution",
        "Cache rendered views when camera doesn't move",
        "Use simplify() on complex meshes"
    ],
    'opencv': [
        "Process at lower resolution, display at full res",
        "Use cv2.setUseOptimized(True)",
        "Limit cv2.waitKey(1) calls - one per frame max"
    ]
}


def print_performance_tips(component: str = None):
    """Print performance tips for a component."""
    if component:
        print(f"\n[Performance Tips - {component}]")
        for tip in PERFORMANCE_TIPS.get(component, []):
            print(f"  • {tip}")
    else:
        print("\n[Performance Tips - All Components]")
        for comp, tips in PERFORMANCE_TIPS.items():
            print(f"\n{comp.upper()}:")
            for tip in tips:
                print(f"  • {tip}")