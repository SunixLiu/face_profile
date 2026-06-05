import numpy as np

from face.recognition import compare_faces, encode_face, face_distance
from config import FACE_TOLERANCE
from profile.models import Profile
from profile.storage import (
    create_profile as db_create_profile,
    get_all_profiles,
    update_profile,
)


def _encoding_to_ndarray(encoding_blob: bytes) -> np.ndarray:
    """Deserialize a face encoding stored as bytes back to np.ndarray."""
    return np.frombuffer(encoding_blob, dtype=np.float64)


def create_user_profile(name: str, face_image: np.ndarray) -> Profile | None:
    """Create a new user profile from a face image. Returns None if no face detected."""
    encoding = encode_face(face_image)
    if encoding is None:
        return None
    return db_create_profile(name, encoding)


def identify_user(face_encoding: np.ndarray) -> Profile | None:
    """Match a face encoding against all stored profiles. Returns best match or None."""
    profiles = get_all_profiles()
    if not profiles:
        return None
    known_encodings = [_encoding_to_ndarray(p.face_encoding) for p in profiles]
    distances = face_distance(known_encodings, face_encoding)
    best_idx = int(distances.argmin())
    if distances[best_idx] <= FACE_TOLERANCE:
        return profiles[best_idx]
    return None


def identify_and_update_last_seen(profile_id: str):
    """Update the last seen timestamp for a profile (future enhancement)."""
    # This is a placeholder for future functionality to track when users were last seen
    pass


def get_profile_by_encoding(face_encoding: np.ndarray) -> tuple[Profile | None, float]:
    """Match a face encoding against all stored profiles and return both profile and confidence score."""
    profiles = get_all_profiles()
    if not profiles:
        return None, float('inf')
    known_encodings = [_encoding_to_ndarray(p.face_encoding) for p in profiles]
    distances = face_distance(known_encodings, face_encoding)
    best_idx = int(distances.argmin())
    min_distance = distances[best_idx]
    if min_distance <= FACE_TOLERANCE:
        return profiles[best_idx], min_distance
    return None, min_distance


def get_all_profiles_with_encodings() -> tuple[list[Profile], list[np.ndarray]]:
    """Get all profiles and their corresponding encodings as separate lists."""
    profiles = get_all_profiles()
    encodings = [_encoding_to_ndarray(p.face_encoding) for p in profiles]
    return profiles, encodings


def identify_user_with_confidence(face_encoding: np.ndarray, threshold: float = None) -> tuple[Profile | None, float]:
    """Identify user with confidence score. Lower distance = higher confidence."""
    from config import FACE_TOLERANCE
    tolerance = threshold or FACE_TOLERANCE
    
    profiles, known_encodings = get_all_profiles_with_encodings()
    if not known_encodings:
        return None, float('inf')
    
    distances = face_distance(known_encodings, face_encoding)
    best_idx = int(distances.argmin())
    min_distance = distances[best_idx]
    
    if min_distance <= tolerance:
        return profiles[best_idx], min_distance
    else:
        # Return the closest match even if it's above threshold, with the distance
        return None, min_distance


def get_known_encodings() -> list[np.ndarray]:
    """Get all stored face encodings as numpy arrays (for the monitor)."""
    return [_encoding_to_ndarray(p.face_encoding) for p in get_all_profiles()]


def update_profile_preferences(profile_id: str, topics: list[str] | None = None, tone: str | None = None, language: str | None = None):
    """Update learned preferences for a profile."""
    kwargs = {}
    if topics is not None:
        kwargs["topics"] = topics
    if tone is not None:
        kwargs["tone"] = tone
    if language is not None:
        kwargs["language"] = language
    if kwargs:
        update_profile(profile_id, **kwargs)


def update_session_id(profile_id: str, session_id: str):
    """Update the Agno session ID for a profile."""
    update_profile(profile_id, agno_session_id=session_id)
