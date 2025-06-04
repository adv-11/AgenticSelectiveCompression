import logging
from langgraph import Agent, Message

from memory.memory_tier_manager import get_memory_manager

logger = logging.getLogger(__name__)


class RetrievalAgent(Agent):
    """
    Waits for 'ingested' Messages, pulls hot/warm/cold context
    and re‐emits tagged as 'retrieved'.
    """

    def __init__(self, name: str = "retrieval_agent"):
        super().__init__(name=name)
        self.memory = get_memory_manager()
        logger.info(f"[{self.name}] Initialized")

    @Agent.on_event("message")
    def handle_ingested(self, msg: Message):
        if msg.metadata.get("stage") != "ingested":
            return

        query = msg.content.strip()
        logger.info(f"[{self.name}] Retrieving context for: {query!r}")

        hot  = self.memory.get_hot(top_k=3)
        warm = self.memory.get_warm(query, top_k=2)
        cold = self.memory.get_cold(query, top_k=2)

        # Re‐emit with all contexts attached
        meta = {
            "stage": "retrieved",
            "hot": hot,
            "warm": warm,
            "cold": cold,
        }
        self.emit(Message(content=query, metadata=meta))
