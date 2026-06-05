import signal
import sys
import threading
import time
from typing import Optional

from face.capture import capture_photo
from face.monitor import FaceMonitor, FaceEvent
from face.recognition import encode_face
from profile.manager import (
    create_user_profile,
    identify_user,
    get_known_encodings,
    update_session_id,
    update_profile_preferences,
    identify_user_with_confidence,
)
from profile.storage import init_db, get_all_profiles
from chat.agent import create_agent, extract_user_preferences
from chat.cli import (
    Command,
    parse_command,
    show_welcome,
    show_help,
    show_profile_info,
    show_system_message,
    show_user_message,
    show_agent_message,
    show_error,
    stream_token,
    prompt_input,
    console,
)
from performance_monitor import start_performance_monitoring, get_performance_report


class App:
    def __init__(self):
        self.monitor = FaceMonitor()
        self.current_profile = None
        self.current_agent = None
        self.running = False
        self._lock = threading.RLock()
        self._new_user_detected = False
        self._is_handling_new_user = False

    def start(self):
        init_db()
        self._load_known_faces()
        self.monitor.known_encodings = get_known_encodings()
        self.monitor.start()

        # Start performance monitoring
        start_performance_monitoring()
        
        show_welcome()
        self.running = True

        # Start background face event processor
        event_thread = threading.Thread(target=self._event_loop, daemon=True)
        event_thread.start()

        # Check if a face is already present at startup
        self._check_initial_face()

        while self.running:
            # Handle new user creation if detected by background thread
            if self._new_user_detected and not self._is_handling_new_user:
                self._new_user_detected = False
                self._is_handling_new_user = True
                
                # Handle in a separate thread to avoid blocking main input loop
                def handle_new_user_async():
                    try:
                        self._handle_new_user_interactively()
                    finally:
                        self._is_handling_new_user = False
                
                new_user_thread = threading.Thread(target=handle_new_user_async, daemon=True)
                new_user_thread.start()
                # Give a brief moment for the thread to start
                time.sleep(0.05)
                continue  # Continue to next iteration to allow user input

            # Get user input
            try:
                user_input = prompt_input()
            except (EOFError, KeyboardInterrupt):
                break

            # Parse and handle commands
            cmd, text = parse_command(user_input)

            if cmd == Command.QUIT:
                break
            elif cmd == Command.HELP:
                show_help()
                continue
            elif cmd == Command.PROFILE:
                with self._lock:
                    if self.current_profile:
                        show_profile_info(self.current_profile)
                    else:
                        show_system_message("No profile active.")
                continue
            elif text.strip().lower() == "performance" or text.strip().lower() == "性能":
                from performance_monitor import get_performance_report
                get_performance_report()
                continue

            # Handle special phrase for profile creation
            if text == "请建立我的个人档案":
                self._handle_create_profile()
                continue

            # Chat with agent
            self._handle_chat(text)

        self._shutdown()

    def _event_loop(self):
        """Background loop to process face events immediately."""
        while self.running:
            event = self.monitor.get_event()
            if event is None:
                time.sleep(0.1)
                continue

            if event == FaceEvent.UNKNOWN_USER:
                with self._lock:
                    show_system_message("New face detected! Type '请建立我的个人档案' or wait for me to ask your name.")
                    self._new_user_detected = True

            elif event == FaceEvent.USER_CHANGED:
                encoding = self.monitor.current_encoding
                if encoding is not None:
                    # Do heavy identification OUTSIDE the lock
                    from profile.manager import identify_user_with_confidence
                    profile, confidence = identify_user_with_confidence(encoding)
                    
                    with self._lock:
                        if profile and profile.id != (self.current_profile.id if self.current_profile else None):
                            show_system_message(f"Welcome back, {profile.name}! (Confidence: {1-confidence:.2f})")
                            self._load_profile(profile)
                        elif profile:  # Same user, just ensure profile is loaded
                            if not self.current_profile or self.current_profile.id != profile.id:
                                self._load_profile(profile)

            elif event == FaceEvent.USER_LEFT:
                with self._lock:
                    show_system_message("User left the camera.")
                    self.current_agent = None
                    self.current_profile = None

    def _handle_new_user_interactively(self):
        """Called from main thread to handle profile creation when a new user is detected."""
        show_system_message("I noticed a new face. Would you like to create a profile? (yes/no)")
        choice = prompt_input().lower().strip()
        if choice in ('yes', 'y', '是', '好'):
            self._handle_create_profile()

    def _check_initial_face(self):
        """Check if someone is already in front of the camera."""
        # Try to get frame from monitor first
        photo = self.monitor.last_frame
        if photo is None:
            photo = capture_photo()
            
        if photo is None:
            return
        encoding = encode_face(photo)
        if encoding is None:
            show_system_message("No face detected. Point your face at the camera.")
            return
        profile = identify_user(encoding)
        if profile:
            self._load_profile(profile)
        else:
            show_system_message("Unknown face detected.")
            self._create_new_profile(photo)

    def _create_new_profile(self, photo):
        name = prompt_input("Enter your name: ")
        if not name.strip():
            show_error("Name cannot be empty.")
            return
        profile = create_user_profile(name.strip(), photo)
        if profile is None:
            show_error("No face found in the photo. Please look at the camera.")
            return
        show_system_message(f"Profile created for {profile.name}!")
        self._load_profile(profile)
        # Update monitor with new known faces
        self.monitor.known_encodings = get_known_encodings()

    def _handle_create_profile(self):
        show_system_message("Creating your profile. Look at the camera...")
        # Try to get frame from monitor first to avoid camera access conflict
        photo = self.monitor.last_frame
        if photo is None:
            photo = capture_photo()
            
        if photo is None:
            show_error("Cannot access webcam.")
            return
        self._create_new_profile(photo)

    def _load_profile(self, profile):
        with self._lock:
            self.current_profile = profile
            self.current_agent = create_agent(profile)
            if not profile.agno_session_id:
                # If there's no session yet, create one by sending an initial message
                # Agno will assign a session_id automatically
                pass
            show_system_message(f"Loaded profile: {profile.name}")

    def _handle_chat(self, text: str):
        if not text.strip():
            return
        
        with self._lock:
            if self.current_profile is None:
                show_system_message("No profile active. Type 请建立我的个人档案 to create one.")
                return
            if self.current_agent is None:
                self.current_agent = create_agent(self.current_profile)
            
            agent = self.current_agent
            profile = self.current_profile

        show_user_message(profile.name, text)

        try:
            response = agent.run(text, stream=True)
            show_agent_message("AI")
            
            # Process the response stream
            chunk_count = 0
            for chunk in response:
                # Check if user left during streaming - only check every 5 chunks to reduce lock contention
                chunk_count += 1
                if chunk_count % 5 == 0:
                    with self._lock:
                        if self.current_agent != agent:
                            show_system_message("\n[User changed or left, stopping response]")
                            break
                
                if hasattr(chunk, 'content') and chunk.content:
                    stream_token(chunk.content)
                elif isinstance(chunk, str):
                    stream_token(chunk)
            
            # Add a newline after response is complete
            console.print()  # newline after response

            # Schedule post-response tasks to run in background to avoid blocking
            def post_response_tasks():
                # Persist session_id after first successful chat
                with self._lock:
                    if self.current_profile and not self.current_profile.agno_session_id and hasattr(agent, 'session_id'):
                        sid = agent.session_id
                        if sid:
                            update_session_id(self.current_profile.id, sid)
                            self.current_profile.agno_session_id = sid
                
                # Update profile preferences based on agent learning
                self._update_profile_from_agent_learning(profile, agent)
            
            # Run post-response tasks in a separate thread to avoid blocking the UI
            post_task_thread = threading.Thread(target=post_response_tasks, daemon=True)
            post_task_thread.start()
        
        except Exception as e:
            show_error(str(e))
    
    def _update_profile_from_agent_learning(self, profile, agent):
        """Extract learned preferences from the agent and update the profile."""
        try:
            # Attempt to extract learned information from the agent
            # This could involve querying the agent's memory system for user-specific facts
            # For now, we'll implement a basic version that could be enhanced later
            
            # Update profile preferences based on agent learning
            # In a real implementation, this would extract specific user interests, topics, etc.
            # from the agent's memory system
            
            # Placeholder: we could extract topics from recent conversations
            # or other learned preferences
            
            # For demonstration purposes, let's imagine we extract some preferences
            # In a real system, this would interface with Agno's memory system
            
            # Only update if we have a meaningful agent object
            if agent and hasattr(agent, 'db'):
                # Future enhancement: extract actual learned preferences from agent memory
                pass
        except Exception as e:
            # Log the error but don't interrupt the main flow
            show_error(f"Error updating profile from agent learning: {str(e)}")

    def _load_known_faces(self):
        """Pre-load all profiles into the identifier cache."""
        profiles = get_all_profiles()
        if profiles:
            show_system_message(f"Loaded {len(profiles)} known profile(s).")

    def _shutdown(self):
        if not self.running:
            return  # already shutting down
        self.running = False
        self.monitor.stop()
        show_system_message("Goodbye!")
        sys.exit(0)


def main():
    app = App()
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda sig, frame: app._shutdown())
    app.start()


if __name__ == "__main__":
    main()