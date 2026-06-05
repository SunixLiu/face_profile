import cv2
import face_recognition
import numpy as np
from config import FACE_TOLERANCE


def cv2_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to RGB for face_recognition."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detect_faces(image: np.ndarray) -> list:
    """Detect all faces in an image. Returns list of face locations."""
    rgb = cv2_to_rgb(image)
    return face_recognition.face_locations(rgb)


def encode_face(image: np.ndarray) -> np.ndarray | None:
    """Get 128D face encoding for the first face found. Returns None if no face."""
    rgb = cv2_to_rgb(image)
    # Use 1 jitter to improve encoding quality slightly without too much overhead
    encodings = face_recognition.face_encodings(rgb, num_jitters=1)
    return encodings[0] if encodings else None


def compare_faces(
    known_encodings: list[np.ndarray],
    face_encoding: np.ndarray,
    tolerance: float = FACE_TOLERANCE,
) -> list[bool]:
    """Compare a face encoding against a list of known encodings."""
    return face_recognition.compare_faces(known_encodings, face_encoding, tolerance)


def face_distance(
    known_encodings: list[np.ndarray],
    face_encoding: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean distance between a face encoding and known encodings."""
    return face_recognition.face_distance(known_encodings, face_encoding)


def has_face_changed(image: np.ndarray, current_encoding: np.ndarray | None) -> bool:
    """Check if the face in the image differs from the current user's encoding."""
    if current_encoding is None:
        return bool(detect_faces(image))
    new_encoding = encode_face(image)
    if new_encoding is None:
        return True
    matches = compare_faces([current_encoding], new_encoding)
    return not matches[0]


def get_known_face_encodings_from_db() -> list[np.ndarray]:
    """Get all face encodings from the database for comparison."""
    from profile.manager import get_known_encodings
    return get_known_encodings()


def match_face_against_database(face_encoding: np.ndarray) -> tuple[str | None, float]:
    """Match a face against all known faces in the database and return profile ID and distance."""
    from profile.manager import get_profile_by_encoding
    profile, distance = get_profile_by_encoding(face_encoding)
    return (profile.id, distance) if profile else (None, float('inf'))