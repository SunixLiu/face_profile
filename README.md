# Face Profile Chat

A CLI chat application that uses webcam-based face recognition to identify users, maintain personalized profiles, and provide LLM-powered conversations. The system learns user habits over time and automatically switches profiles when the person in front of the camera changes.

## Features

- **Face Recognition** — Detects and identifies users via webcam using 128D face encoding vectors
- **Automatic Profile Switching** — Background daemon thread monitors the camera and switches profiles when the user changes
- **Personalized Chat** — Agno Agent framework with DeepSeek LLM, personalized per user based on learned preferences
- **Continuous Learning** — Agno's built-in user memory automatically extracts preferences, interests, and communication style from each conversation
- **Profile Management** — Create, view, and manage user profiles with face biometrics stored in SQLite

## Architecture

```
main.py (CLI Loop + Orchestration)
   ├── face/          Webcam capture, face encoding, background monitoring
   ├── profile/       SQLite-backed profile CRUD with face biometric storage
   └── chat/          Agno Agent (DeepSeek) + Rich CLI interface
```

## Setup

### 1. Configure DeepSeek API

Edit `.env`:

```
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Note: `face-recognition` requires `dlib`. On macOS:

```bash
brew install cmake
pip install dlib face-recognition
```

### 3. Run

```bash
python main.py
```

## Usage

| Input | Action |
|-------|--------|
| Type anything | Chat with the AI agent |
| `请建立我的个人档案` | Create a new profile via webcam |
| `/profile` | Show current profile info |
| `/help` | Show available commands |
| `/quit` | Exit the program |

### Automatic Behavior

- **First run with no profiles**: Detects face → prompts for name → captures photo → creates profile → starts chat
- **Known user returns**: Auto-detects and switches to their profile, resumes conversation history
- **New face appears**: Prompts for name and creates a new profile
- **User leaves camera**: Pauses the active session

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `CAMERA_INDEX` | `0` | OpenCV camera device index |
| `FACE_MONITOR_INTERVAL` | `2.0` | Seconds between face checks |
| `FACE_TOLERANCE` | `0.4` | Face matching strictness (lower = stricter) |
| `DB_PATH` | `profiles.db` | SQLite database location |

## How It Works

### Face Recognition

The `face_recognition` library computes a 128-dimensional encoding vector for each detected face. This vector acts as a biometric fingerprint — the same person consistently produces similar vectors. User identity is determined by computing the Euclidean distance between a live face encoding and stored encodings; a match is confirmed when the distance falls below the configured tolerance (default 0.4).

### Profile Storage

Each profile stores:
- A UUID-based identifier
- The username
- The 128D face encoding (serialized as bytes)
- An Agno session ID for conversation continuity
- Learned preferences (topics of interest, tone, language)

### Learning & Personalization

Agno's `enable_user_memories` mode automatically:
1. Extracts facts and preferences from each conversation
2. Stores them as user memories in a dedicated SQLite database
3. Injects relevant memories into the system prompt for future conversations

This means the AI gradually learns each user's interests, preferred communication style, and recurring topics without explicit configuration.

### Threading Model

The face monitor runs in a background daemon thread, polling the webcam at the configured interval. Face change events are pushed to a thread-safe `Queue` and consumed by the main thread between CLI inputs. No shared mutable state exists between threads beyond the queue.
