"""
Retrieval Agent - Handles context retrieval from memory tiers
"""

import logging
import threading
from typing import Dict, Any, List
from memory.memory_tier_manager import get_memory_manager

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    Retrieves context from hot, warm, and cold memory tiers
    """

    def __init__(self, name: str = "retrieval_agent"):
        self.name = name
        self.memory = get_memory_manager()
        self.message_handlers = []
        self.lock = threading.Lock()
        logger.info(f"[{self.name}] Initialized")

    def register_handler(self, handler):
        """Register a message handler function"""
        with self.lock:
            self.message_handlers.append(handler)

    def retrieve_context(self, query: str, hot_k: int = 3, warm_k: int = 2, cold_k: int = 2) -> Dict[str, Any]:
        """
        Retrieve context from all memory tiers
        
        Args:
            query: The search query
            hot_k: Number of hot memory items to retrieve
            warm_k: Number of warm memory items to retrieve  
            cold_k: Number of cold memory items to retrieve
            
        Returns:
            Dict containing retrieved context from all tiers
        """
        logger.info(f"[{self.name}] Retrieving context for: {query[:100]}...")
        
        try:
            # Retrieve from all tiers
            hot_items = self.memory.get_hot(top_k=hot_k)
            warm_items = self.memory.get_warm(query, top_k=warm_k)
            cold_items = self.memory.get_cold(query, top_k=cold_k)
            
            result = {
                "status": "retrieved",
                "query": query,
                "hot": hot_items,
                "warm": warm_items,
                "cold": cold_items,
                "metadata": {"stage": "retrieved"}
            }
            
            logger.info(f"[{self.name}] Retrieved {len(hot_items)} hot, {len(warm_items)} warm, {len(cold_items)} cold items")
            
            # Notify handlers
            with self.lock:
                for handler in self.message_handlers:
                    try:
                        handler(result)
                    except Exception as e:
                        logger.error(f"[{self.name}] Handler error: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"[{self.name}] Retrieval error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "query": query,
                "hot": [],
                "warm": [],
                "cold": []
            }

    def handle_ingested_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle messages that have been ingested and need context retrieval
        
        Args:
            message_data: Data from ingest agent
            
        Returns:
            Dict containing retrieved context
        """
        if message_data.get("metadata", {}).get("stage") != "ingested":
            return {"status": "skipped", "reason": "not_ingested"}
        
        query = message_data.get("content", "")
        return self.retrieve_context(query)

    def process_message(self, message: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process message and retrieve context
        
        Args:
            message: The message text
            metadata: Optional metadata
            
        Returns:
            Dict containing retrieved context
        """
        metadata = metadata or {}
        
        # Check if this is an ingested message
        if metadata.get("stage") == "ingested":
            return self.retrieve_context(message)
        
        # Otherwise, skip
        return {"status": "skipped", "reason": "not_ready_for_retrieval"}