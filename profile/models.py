from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid

@dataclass
class Profile:
    name: str
    face_encoding: bytes  # numpy ndarray serialized via .tobytes()
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agno_session_id: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    topics: list[str] = field(default_factory=list)
    tone: str = "friendly"
    language: str = "zh"
