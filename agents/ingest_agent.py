import logging
from langgraph import Agent, Message

from memory.memory_tier_manager import get_memory_manager

logger = logging.getLogger(__name__)


class IngestAgent(Agent):
    """
    Listens to every raw user Message, persists it into memory,
    then re‐emits the message tagged as 'ingested'.
    """

    def __init__(self, name: str = "ingest_agent"):
        super().__init__(name=name)
        self.memory = get_memory_manager()
        logger.info(f"[{self.name}] Initialized")

    @Agent.on_event("message")
    def handle_message(self, msg: Message):
        # Only ingest true user messages (no agent‐emitted replies)
        if msg.metadata.get("agent"):
            return

        text = msg.content.strip()
        logger.info(f"[{self.name}] Ingesting: {text!r}")
        try:
            tier = self.memory.add(text)
            logger.info(f"[{self.name}] Stored in tier '{tier}'")
        except Exception as e:
            logger.error(f"[{self.name}] Error persisting: {e}")

        # Emit a new Message so next agent can pick it up
        self.emit(Message(content=text, metadata={"stage": "ingested"}))
