"""
Ingest Agent - Handles incoming messages and stores them in memory
"""

import logging
import threading
from typing import Dict, Any
from memory.memory_tier_manager import get_memory_manager

logger = logging.getLogger(__name__)


class IngestAgent:
    """
    Handles incoming user messages and stores them in appropriate memory tiers
    """

    def __init__(self, name: str = "ingest_agent"):
        self.name = name
        self.memory = get_memory_manager()
        self.message_handlers = []
        self.lock = threading.Lock()
        logger.info(f"[{self.name}] Initialized")

    def register_handler(self, handler):
        """Register a message handler function"""
        with self.lock:
            self.message_handlers.append(handler)

    def process_message(self, message: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process incoming message and store in memory
        
        Args:
            message: The user message text
            metadata: Optional metadata about the message
            
        Returns:
            Dict containing processing results
        """
        metadata = metadata or {}
        
        # Only process user messages (not agent responses)
        if metadata.get("agent"):
            logger.debug(f"[{self.name}] Skipping agent message")
            return {"status": "skipped", "reason": "agent_message"}

        text = message.strip()
        logger.info(f"[{self.name}] Processing message: {text[:100]}...")
        
        try:
            # Store in memory
            tier = self.memory.add(text)
            logger.info(f"[{self.name}] Stored in '{tier}' tier")
            
            # Prepare result
            result = {
                "status": "processed",
                "tier": tier,
                "content": text,
                "metadata": {**metadata, "stage": "ingested"}
            }
            
            # Notify handlers
            with self.lock:
                for handler in self.message_handlers:
                    try:
                        handler(result)
                    except Exception as e:
                        logger.error(f"[{self.name}] Handler error: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"[{self.name}] Processing error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "content": text,
                "metadata": metadata
            }

    def handle_message(self, message: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Alias for process_message for backward compatibility
        """
        return self.process_message(message, metadata)