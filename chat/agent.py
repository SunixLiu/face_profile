from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAILike

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, BASE_DIR
from profile.models import Profile


def _build_system_prompt(profile: Profile) -> str:
    topics_hint = (
        f"You know they are interested in: {', '.join(profile.topics)}."
        if profile.topics
        else "You are still learning about their interests."
    )
    return (
        f"You are a friendly and helpful AI chat companion. "
        f"You are talking to {profile.name}. "
        f"{topics_hint} "
        f"Respond in {'Chinese' if profile.language == 'zh' else profile.language}. "
        f"Keep a {profile.tone} tone. "
        f"Learn from each conversation to better understand {profile.name}'s "
        f"preferences, interests, and communication style."
    )


def create_agent(profile: Profile, db_path: str | None = None) -> Agent:
    """Create an Agno agent for a user profile with DeepSeek as the LLM."""
    db_file = db_path or str(BASE_DIR / "agno_sessions.db")
    return Agent(
        name=f"chat_agent_{profile.id[:8]}",
        user_id=profile.id,
        session_id=profile.agno_session_id or None,
        model=OpenAILike(
            id="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        ),
        db=SqliteDb(db_file=db_file),
        instructions=_build_system_prompt(profile),
        add_history_to_context=True,
        num_history_runs=10,
        # Disable synchronous learning to avoid delays at the end of streaming
        learning=False,
        enable_user_memories=True,
        add_memories_to_context=True,
        # DeepSeek's OpenAI-compatible endpoint rejects the response_format
        # Agno uses internally for session summary generation.
        enable_session_summaries=False,
        markdown=True,
        stream=True,
    )


def extract_user_preferences(agent: Agent, profile: Profile):
    """Extract learned user preferences from the agent's memory system."""
    try:
        # Query the agent's memory for user-specific information
        # Access Agno's memory system to extract learned preferences
        if hasattr(agent, 'memory') and hasattr(agent.memory, 'search'):
            # Search for user-related memories
            user_memories = agent.memory.search(query=f"about {profile.name}", limit=20)
            
            # Extract topics from memories
            topics = set()  # Use a set to avoid duplicates
            tones = []  # Track communication style
            languages = []  # Track language preference
            
            for memory in user_memories:
                # Process memories to extract topics of interest
                if hasattr(memory, 'content'):
                    content = memory.content.lower()
                    
                    # Define common topics and interests
                    potential_topics = [
                        "technology", "programming", "software", "ai", "machine learning",
                        "sports", "football", "basketball", "tennis", "soccer",
                        "music", "jazz", "rock", "pop", "classical", "hip hop",
                        "travel", "tourism", "adventure", "exploration", "vacation",
                        "food", "cooking", "recipes", "cuisine", "restaurants",
                        "books", "reading", "literature", "novels", "fiction",
                        "movies", "cinema", "film", "hollywood", "bollywood",
                        "health", "fitness", "exercise", "nutrition", "wellness",
                        "business", "finance", "investment", "entrepreneurship",
                        "science", "research", "discovery", "innovation",
                        "art", "painting", "drawing", "sculpture", "design",
                        "gaming", "video games", "pc games", "mobile games",
                        "cars", "automotive", "driving", "racing",
                        "nature", "outdoors", "hiking", "wildlife", "animals",
                        "politics", "government", "elections", "policy"
                    ]
                    
                    # Extract topics
                    for topic in potential_topics:
                        if topic in content:
                            topics.add(topic)
                    
                    # Extract tone/style hints
                    formal_indicators = ["formal", "professional", "business", "academic"]
                    casual_indicators = ["casual", "informal", "friendly", "chill", "relaxed"]
                    
                    for indicator in formal_indicators:
                        if indicator in content:
                            tones.append("formal")
                    for indicator in casual_indicators:
                        if indicator in content:
                            tones.append("casual")
            
            # Determine most common tone
            final_tone = profile.tone
            if tones:
                # Count most common tone
                from collections import Counter
                tone_counts = Counter(tones)
                final_tone = tone_counts.most_common(1)[0][0]
            
            return {
                "topics": list(topics)[:10],  # Limit to top 10 topics
                "tone": final_tone,
                "language": profile.language
            }
        
        # If Agno's memory system is not accessible, return None
        return None
        
    except Exception as e:
        print(f"Error extracting user preferences: {e}")
        return None