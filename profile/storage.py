import sqlite3
import json
from datetime import datetime, timezone

import numpy as np

from config import DB_PATH
from profile.models import Profile


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                face_encoding BLOB NOT NULL,
                encoding_shape TEXT NOT NULL DEFAULT '[128]',
                agno_session_id TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                topics TEXT DEFAULT '[]',
                tone TEXT DEFAULT 'friendly',
                language TEXT DEFAULT 'zh'
            )
            """
        )


def create_profile(name: str, encoding: np.ndarray, session_id: str = "") -> Profile:
    now = datetime.now(timezone.utc).isoformat()
    shape = json.dumps(list(encoding.shape))
    encoding_blob = encoding.tobytes()
    profile = Profile(
        name=name,
        face_encoding=encoding_blob,
        agno_session_id=session_id,
        created_at=now,
        updated_at=now,
    )
    with _conn() as db:
        db.execute(
            """INSERT INTO profiles (id, name, face_encoding, encoding_shape,
               agno_session_id, created_at, updated_at, topics, tone, language)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                profile.id,
                name,
                encoding_blob,
                shape,
                session_id,
                now,
                now,
                json.dumps(profile.topics),
                profile.tone,
                profile.language,
            ),
        )
    return profile


def get_profile(profile_id: str) -> Profile | None:
    with _conn() as db:
        row = db.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    return _row_to_profile(row) if row else None


def get_all_profiles() -> list[Profile]:
    with _conn() as db:
        rows = db.execute("SELECT * FROM profiles").fetchall()
    return [_row_to_profile(r) for r in rows]


def update_profile(profile_id: str, **kwargs):
    allowed = {"name", "agno_session_id", "topics", "tone", "language"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    if "topics" in updates:
        updates["topics"] = json.dumps(updates["topics"])
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [profile_id]
    with _conn() as db:
        db.execute(
            f"UPDATE profiles SET {set_clause} WHERE id = ?",
            values,
        )


def delete_profile(profile_id: str):
    with _conn() as db:
        db.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))


def _row_to_profile(row: sqlite3.Row) -> Profile:
    encoding_blob = bytes(row["face_encoding"])
    return Profile(
        id=row["id"],
        name=row["name"],
        face_encoding=encoding_blob,
        agno_session_id=row["agno_session_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        topics=json.loads(row["topics"]),
        tone=row["tone"],
        language=row["language"],
    )
