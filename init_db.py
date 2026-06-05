#!/usr/bin/env python3
"""Initialize the database and ensure all tables are created."""

from profile.storage import init_db
from profile.manager import get_known_encodings
from chat.cli import show_system_message


def initialize_system():
    """Initialize the database and system components."""
    print("Initializing face profile system...")
    
    # Initialize database tables
    init_db()
    
    # Load known face encodings
    known_encodings = get_known_encodings()
    show_system_message(f"System initialized. Loaded {len(known_encodings)} known face profiles.")
    
    print("Initialization complete!")


if __name__ == "__main__":
    initialize_system()