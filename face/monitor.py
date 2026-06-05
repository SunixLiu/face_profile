import threading
import time
from queue import Queue
from enum import Enum, auto

import cv2
import numpy as np

from config import CAMERA_INDEX, FACE_MONITOR_INTERVAL
from face.recognition import encode_face, compare_faces, get_known_face_encodings_from_db


class FaceEvent(Enum):
    USER_CHANGED = auto()
    UNKNOWN_USER = auto()
    USER_LEFT = auto()


class FaceMonitor:
    """Background thread that periodically checks the webcam for face changes.

    Events are pushed to a thread-safe Queue and consumed by the main thread.
    """

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        interval: float = FACE_MONITOR_INTERVAL,
    ):
        self.camera_index = camera_index
        self.interval = interval
        self.event_queue: Queue = Queue()
        self.known_encodings: list[np.ndarray] = []
        self.current_encoding: np.ndarray | None = None
        self.last_frame: np.ndarray | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._cap: cv2.VideoCapture | None = None

    def start(self):
        self._running = True
        self._cap = cv2.VideoCapture(self.camera_index)
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()

    def _monitor_loop(self):
        # Add a counter to reduce processing frequency for performance
        frame_counter = 0
        skip_frames = 2  # Process every 3rd frame to reduce CPU usage
        
        while self._running:
            ret, frame = self._cap.read() if self._cap else (False, None)
            if not ret:
                time.sleep(self.interval)
                continue

            # Skip frames to reduce CPU usage
            frame_counter += 1
            if frame_counter % skip_frames != 0:
                # Don't sleep here, just continue to read the next frame quickly
                # to clear the buffer
                continue

            self.last_frame = frame
            current_face = encode_face(frame)

            if current_face is None and self.current_encoding is not None:
                self.current_encoding = None
                self.event_queue.put(FaceEvent.USER_LEFT)

            elif current_face is not None and self.current_encoding is None:
                self.current_encoding = current_face
                self.event_queue.put(self._classify_face(current_face))

            elif current_face is not None and self.current_encoding is not None:
                matches = compare_faces([self.current_encoding], current_face)
                if not matches[0]:
                    self.current_encoding = current_face
                    self.event_queue.put(self._classify_face(current_face))

            time.sleep(self.interval)

    def _classify_face(self, encoding: np.ndarray) -> FaceEvent:
        # Use known encodings from the instance variable first
        if not self.known_encodings:
            # Fallback to getting encodings directly from database
            known_encodings = get_known_face_encodings_from_db()
            if not known_encodings:
                return FaceEvent.UNKNOWN_USER
            matches = compare_faces(known_encodings, encoding)
        else:
            matches = compare_faces(self.known_encodings, encoding)
        return FaceEvent.USER_CHANGED if any(matches) else FaceEvent.UNKNOWN_USER

    def get_event(self) -> FaceEvent | None:
        """Non-blocking poll for the next face event."""
        try:
            return self.event_queue.get_nowait()
        except Exception:
            return None