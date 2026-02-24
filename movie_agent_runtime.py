"""
Movie Recommendation Agent for AgentCore Runtime

This agent uses AgentCore Memory to remember user movie preferences across sessions.
Deploy this to AgentCore Runtime with the following environment variables:
  - MEMORY_ID: Your pre-created memory resource ID
  - MODEL_ID: Bedrock model ID (e.g., us.anthropic.claude-sonnet-4-20250514-v1:0)
  - AWS_REGION: AWS region (e.g., us-west-2)
"""

import os
import logging
from datetime import datetime
from strands import Agent, tool
from strands.models import BedrockModel
from strands.hooks import (
    AgentInitializedEvent,
    AfterInvocationEvent,
    HookProvider,
    HookRegistry,
)
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("movie-agent")

# Initialize the AgentCore app
app = BedrockAgentCoreApp()

# Configuration
MEMORY_ID = "MovieAgentMemory-xyz"  # Give the Memory ID from Agentcore
MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
REGION = "us-east-1"

# Global agent instance
agent = None


class MovieMemoryHookProvider(HookProvider):
    """
    Memory hook provider for the movie agent.

    Handles:
    - Loading user movie preferences when agent initializes
    - Saving conversations after each interaction (triggers preference extraction)
    """

    def __init__(self, region_name: str):
        logger.info(f"Initializing MovieMemoryHookProvider with region {region_name}")
        self.memory_client = MemoryClient(region_name=region_name)

    def on_agent_initialized(self, event: AgentInitializedEvent):
        """Load movie preferences when agent starts"""
        logger.info("Agent initialization hook triggered")

        memory_id = event.agent.state.get("memory_id")
        actor_id = event.agent.state.get("actor_id")

        if not memory_id or not actor_id:
            logger.warning(
                f"Missing required state - memory_id: {memory_id}, actor_id: {actor_id}"
            )
            return

        try:
            # Retrieve stored movie preferences from long-term memory
            namespace = f"user/{actor_id}/movie_preferences"

            preferences = self.memory_client.retrieve_memories(
                memory_id=memory_id,
                namespace=namespace,
                query="movie preferences genres directors favorites dislikes actors",
                top_k=10,
            )

            if preferences:
                # Format preferences for context
                pref_texts = []
                for pref in preferences:
                    if isinstance(pref, dict):
                        content = pref.get("content", {})
                        if isinstance(content, dict):
                            text = content.get("text", "").strip()
                            if text:
                                pref_texts.append(f"- {text}")

                if pref_texts:
                    context = "\n".join(pref_texts)
                    event.agent.system_prompt += f"\n\n## User's Movie Preferences (from previous conversations):\n{context}"
                    logger.info(f"✅ Loaded {len(pref_texts)} movie preferences")
            else:
                logger.info("No previous movie preferences found - starting fresh!")

        except Exception as e:
            logger.error(f"Error loading preferences: {e}", exc_info=True)

    def on_after_invocation(self, event: AfterInvocationEvent):
        """Save conversation after each interaction"""
        logger.info("After invocation hook triggered")

        memory_id = event.agent.state.get("memory_id")
        actor_id = event.agent.state.get("actor_id")
        session_id = event.agent.state.get("session_id")

        if not memory_id or not actor_id or not session_id:
            logger.warning(
                f"Missing required state for saving - memory_id: {memory_id}, actor_id: {actor_id}, session_id: {session_id}"
            )
            return

        try:
            messages = event.agent.messages
            if len(messages) < 2:
                return

            # Get the last user message and assistant response
            user_msg = None
            assistant_msg = None

            for msg in reversed(messages):
                if msg["role"] == "assistant" and not assistant_msg:
                    content = msg.get("content", [])
                    if (
                        content
                        and isinstance(content[0], dict)
                        and "text" in content[0]
                    ):
                        assistant_msg = content[0]["text"]
                elif msg["role"] == "user" and not user_msg:
                    content = msg.get("content", [])
                    if (
                        content
                        and isinstance(content[0], dict)
                        and "text" in content[0]
                    ):
                        if "toolResult" not in content[0]:
                            user_msg = content[0]["text"]
                            break

            if user_msg and assistant_msg:
                # Save the conversation turn to memory
                # This triggers the USER_PREFERENCE strategy to extract preferences
                self.memory_client.create_event(
                    memory_id=memory_id,
                    actor_id=actor_id,
                    session_id=session_id,
                    messages=[(user_msg, "USER"), (assistant_msg, "ASSISTANT")],
                )
                logger.info("💾 Saved conversation to memory")

        except Exception as e:
            logger.error(f"Error saving conversation: {e}", exc_info=True)

    def register_hooks(self, registry: HookRegistry):
        """Register the memory hooks"""
        logger.info("Registering movie memory hooks")
        registry.add_callback(AgentInitializedEvent, self.on_agent_initialized)
        registry.add_callback(AfterInvocationEvent, self.on_after_invocation)


# =============================================================================
# TOOLS
# =============================================================================


@tool
def search_movies(query: str, max_results: int = 5) -> str:
    """
    Search for movie information, reviews, ratings, or recommendations.

    Args:
        query: Search query about movies (e.g., "best sci-fi movies 2024", "Inception reviews")
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Search results with movie information
    """
    try:
        results = DDGS().text(f"{query} movie", region="us-en", max_results=max_results)
        if not results:
            return "No results found."

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = r.get("body", "")
            formatted.append(f"{i}. {title}\n   {body}")

        return "\n\n".join(formatted)
    except RatelimitException:
        return "Rate limit reached. Please try again later."
    except DDGSException as e:
        return f"Search error: {e}"
    except Exception as e:
        return f"Search error: {str(e)}"


def get_system_prompt() -> str:
    """Generate the system prompt for the movie agent"""
    return f"""You are a friendly movie recommendation assistant with excellent taste in films.

Your role:
- Help users discover movies they'll love
- Remember their preferences (genres, directors, actors, specific movies)
- Give personalized recommendations based on their taste
- Discuss movies, share interesting facts, and engage in movie conversations

When chatting:
- Pay attention to movies and genres they mention liking or disliking
- Note any directors, actors, or styles they prefer
- Use their preferences to tailor recommendations
- Be enthusiastic about movies!

Today's date: {datetime.today().strftime('%Y-%m-%d')}
"""


def initialize_agent(actor_id: str, session_id: str):
    """Initialize the movie agent with memory hooks"""
    global agent

    logger.info(
        f"Initializing movie agent for actor_id={actor_id}, session_id={session_id}"
    )

    # Create model
    logger.info(f"Creating model with ID: {MODEL_ID}")
    model = BedrockModel(model_id=MODEL_ID)

    # Create memory hook
    logger.info(f"Creating memory hook with region: {REGION}")
    memory_hook = MovieMemoryHookProvider(region_name=REGION)

    # Create agent with tools
    agent = Agent(
        model=model,
        hooks=[memory_hook],
        tools=[search_movies],
        system_prompt=get_system_prompt(),
        state={"memory_id": MEMORY_ID, "actor_id": actor_id, "session_id": session_id},
    )

    logger.info(f"✅ Movie agent initialized with state: {agent.state.get()}")


@app.entrypoint
def movie_agent(payload: dict, context):
    """
    Main entry point for the movie recommendation agent.

    Expected payload:
    {
        "prompt": "User's message",
        "actor_id": "unique_user_id"  # Optional, defaults to "default_user"
    }

    The session_id comes from context.session_id (managed by AgentCore Runtime)
    """
    global agent

    logger.info(f"Received payload: {payload}")
    logger.info(f"Context session_id: {context.session_id}")
    print(f"Session ID: {context.session_id}")

    # Extract values from payload
    user_input = payload.get("prompt")
    actor_id = payload.get("actor_id", "default_user")
    session_id = context.session_id

    print(f"Actor ID: {actor_id}")
    print(f"Memory ID: {MEMORY_ID}")

    # Validate required fields
    if not user_input:
        error_msg = "❌ ERROR: Missing 'prompt' field in payload"
        logger.error(error_msg)
        return error_msg

    if not MEMORY_ID:
        error_msg = "❌ ERROR: MEMORY_ID environment variable not set"
        logger.error(error_msg)
        return error_msg

    # Initialize agent on first request or if session changed
    if agent is None:
        logger.info("First request - initializing agent")
        initialize_agent(actor_id, session_id)
    else:
        # Update state if actor_id or session_id changed
        current_session = agent.state.get("session_id")
        current_actor = agent.state.get("actor_id")

        if current_session != session_id or current_actor != actor_id:
            logger.info(f"Session or actor changed - reinitializing agent")
            initialize_agent(actor_id, session_id)

    # Invoke the agent
    logger.info(f"Invoking agent with input: {user_input}")
    response = agent(user_input)

    # Extract response text
    response_text = response.message["content"][0]["text"]
    logger.info(f"✅ Agent response: {response_text[:100]}...")

    return response_text


if __name__ == "__main__":
    logger.info("Starting Movie Agent on AgentCore Runtime")
    app.run()
