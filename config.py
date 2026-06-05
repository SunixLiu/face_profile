import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

DB_PATH = BASE_DIR / "profiles.db"
CAMERA_INDEX = 0
FACE_MONITOR_INTERVAL = 2.0  # seconds between face checks
FACE_TOLERANCE = 0.5 # lower = stricter face matching
