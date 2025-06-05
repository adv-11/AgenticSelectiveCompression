import os
import logging
from langgraph import Agent, Message
import openai

from memory.memory_tier_manager import get_memory_manager

logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")


class OrchestratorAgent(Agent):
    """
    Listens for 'retrieved' messages, composes the multi‐tier prompt,
    calls gpt4o-mini, and emits the assistant’s response.
    """

    def __init__(self, name: str = "orchestrator_agent"):
        super().__init__(name=name)
        self.memory = get_memory_manager()
        logger.info(f"[{self.name}] Initialized")

    @Agent.on_event("message")
    def handle_retrieved(self, msg: Message):
        if msg.metadata.get("stage") != "retrieved":
            return

        user_text = msg.content
        hot_items = msg.metadata["hot"]
        warm_items = msg.metadata["warm"]
        cold_items = msg.metadata["cold"]

        # Build sections
        hot_sec  = "\n".join(f"- {t}" for t in hot_items)
        warm_sec = "\n".join(f"- {w['summary']}" for w in warm_items)
        cold_sec = "\n".join(f"- {c['content'][:200]}… (score {c['score']:.2f})"
                             for c in cold_items)

        system_prompt = (
            "You are a helpful assistant.\n"
            "Use the following context from memory tiers:\n\n"
            "Hot Memory:\n" + hot_sec + "\n\n"
            "Warm Memory:\n" + warm_sec + "\n\n"
            "Cold Memory:\n" + cold_sec + "\n\n"
            "Now answer the user’s message."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_text}
        ]

        # Call LLM
        try:
            resp = openai.ChatCompletion.create(
                model="gpt4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=512
            )
            reply = resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[{self.name}] LLM error: {e}")
            reply = "Sorry, I encountered an error while thinking."

        # Emit final assistant message
        self.emit(
            Message(content=reply, metadata={"agent": self.name})
        )
