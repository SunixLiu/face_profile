# CLAUDE.md — Face Profile Chat

## Project Overview

CLI chat application with webcam-based face recognition for user identification, profile management, and personalized LLM-powered conversations via Agno + DeepSeek.

## Tech Stack

- **Face**: OpenCV (`cv2`) for capture, `face_recognition` (dlib) for 128D encoding
- **AI Agent**: `agno` framework with `OpenAIChat` model pointing to DeepSeek API
- **Storage**: SQLite for profiles + Agno's `SqliteDb` for sessions/memories
- **CLI**: `rich` for formatted terminal output
- **Config**: `python-dotenv` for `.env` file

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point, CLI loop, orchestrates all modules |
| `config.py` | All configuration constants (camera, tolerance, DB path) |
| `face/capture.py` | Single-frame webcam photo capture |
| `face/recognition.py` | Face detection, 128D encoding, comparison, distance |
| `face/monitor.py` | Background daemon thread for face change detection |
| `profile/models.py` | `Profile` dataclass |
| `profile/storage.py` | SQLite CRUD (init, create, read, update, delete) |
| `profile/manager.py` | Business logic (identify user, create profile, preferences) |
| `chat/agent.py` | Agno agent factory with DeepSeek + user memory |
| `chat/cli.py` | Rich-based display helpers and command parser |

## Face Encoding Pipeline

```
Webcam frame (BGR) → cv2_to_rgb() → face_recognition.face_encodings() → 128D np.ndarray
```

The 128D vector is the biometric key. Stored as `encoding.tobytes()` in SQLite BLOB. Deserialized with `np.frombuffer(blob, dtype=np.float64)`.

## Agno Integration Pattern

```python
Agent(
    model=OpenAIChat(id="deepseek-chat", api_key=..., base_url=...),
    user_id=profile.id,           # Per-user memory isolation
    session_id=profile.session,   # Conversation continuity
    db=SqliteDb(db_file="..."),   # Persist sessions + memories
    enable_user_memories=True,    # Auto-extract user facts
    add_memories_to_context=True, # Inject memories into prompts
    enable_session_summaries=True,# Summarize long conversations
    add_history_to_context=True,  # Include chat history
    learning=True,                # Enable learning from user interactions
    instructions=system_prompt,   # Personalized from profile
)
```

## Threading Model

- **Main thread**: CLI input loop, agent calls, UI rendering
- **Monitor thread** (daemon): Polls webcam, pushes `FaceEvent` to `Queue`
- **No shared mutable state**: Threads communicate only via the thread-safe queue
- Monitor is started/stopped in `App.start()` / `App._shutdown()`

## Commands

- `请建立我的个人档案` — Triggers profile creation flow (webcam capture + name prompt)
- `/help` — Show command list
- `/profile` — Show current profile details
- `/quit` or `/exit` — Graceful shutdown

## Environment

- `.env` file for `DEEPSEEK_API_KEY` and `DEEPSEEK_BASE_URL`
- Never commit `.env` to git
- Default base URL: `https://api.deepseek.com`

## Database Schema

Two separate SQLite databases:
1. `profiles.db` — User profiles (face encodings, preferences, session IDs)
2. `agno_sessions.db` — Agno-managed (sessions, memories, metrics)

The profiles table stores face encodings as BLOB with the encoding shape serialized as JSON for validation.
