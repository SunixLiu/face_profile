import cv2
import numpy as np
from config import CAMERA_INDEX


def capture_photo(camera_index: int = CAMERA_INDEX) -> np.ndarray | None:
    """Capture a single frame from the webcam. Returns BGR image or None."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    
    # Skip a few frames to allow camera to adjust auto-exposure
    for _ in range(5):
        cap.read()
        
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame
