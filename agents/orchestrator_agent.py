"""
Orchestrator Agent - Coordinates response generation using retrieved context
"""

import os
import logging
import threading
from typing import Dict, Any, List
from openai import OpenAI
from memory.memory_tier_manager import get_memory_manager

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Orchestrates response generation using context from all memory tiers
    """

    def __init__(self, name: str = "orchestrator_agent"):
        self.name = name
        self.memory = get_memory_manager()
        self.message_handlers = []
        self.lock = threading.Lock()
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)
        
        logger.info(f"[{self.name}] Initialized")

    def register_handler(self, handler):
        """Register a message handler function"""
        with self.lock:
            self.message_handlers.append(handler)

    def generate_response(self, user_message: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate response using retrieved context
        
        Args:
            user_message: The user's message
            context_data: Context retrieved from memory tiers
            
        Returns:
            Dict containing generated response
        """
        logger.info(f"[{self.name}] Generating response for: {user_message[:100]}...")
        
        try:
            # Extract context from different tiers
            hot_items = context_data.get("hot", [])
            warm_items = context_data.get("warm", [])
            cold_items = context_data.get("cold", [])
            
            # Build context sections
            context_sections = []
            
            if hot_items:
                hot_section = "Recent conversation:\n" + "\n".join(f"- {item}" for item in hot_items)
                context_sections.append(hot_section)
            
            if warm_items:
                warm_section = "Relevant summaries:\n" + "\n".join(
                    f"- {item.get('summary', item.get('content', '')[:100])}" for item in warm_items
                )
                context_sections.append(warm_section)
            
            if cold_items:
                cold_section = "Related archived content:\n" + "\n".join(
                    f"- {item.get('content', '')[:200]}... (relevance: {item.get('score', 0):.2f})" 
                    for item in cold_items
                )
                context_sections.append(cold_section)
            
            # Build system prompt
            system_prompt = "You are a helpful assistant."
            if context_sections:
                system_prompt += "\n\nContext from memory tiers:\n" + "\n\n".join(context_sections)
            
            # Generate response
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=512
            )
            
            ai_response = response.choices[0].message.content.strip()
            logger.info(f"[{self.name}] Generated response: {ai_response[:100]}...")
            
            # Store AI response in memory
            try:
                self.memory.add(f"Assistant: {ai_response}")
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to store AI response: {e}")
            
            result = {
                "status": "generated",
                "response": ai_response,
                "user_message": user_message,
                "context_used": {
                    "hot_count": len(hot_items),
                    "warm_count": len(warm_items),
                    "cold_count": len(cold_items)
                },
                "metadata": {"agent": self.name}
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
            logger.error(f"[{self.name}] Response generation error: {e}")
            error_response = "I'm sorry, I encountered an error while processing your message."
            
            return {
                "status": "error",
                "error": str(e),
                "response": error_response,
                "user_message": user_message,
                "metadata": {"agent": self.name}
            }

    def handle_retrieved_message(self, retrieved_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle messages with retrieved context
        
        Args:
            retrieved_data: Data from retrieval agent
            
        Returns:
            Dict containing generated response
        """
        if retrieved_data.get("metadata", {}).get("stage") != "retrieved":
            return {"status": "skipped", "reason": "not_retrieved"}
        
        user_message = retrieved_data.get("query", "")
        return self.generate_response(user_message, retrieved_data)

    def process_message(self, message: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process message and generate response if context is available
        
        Args:
            message: The message text
            metadata: Optional metadata
            
        Returns:
            Dict containing response or skip status
        """
        metadata = metadata or {}
        
        # Check if this has retrieved context
        if metadata.get("stage") == "retrieved":
            # Extract context from metadata
            context_data = {
                "hot": metadata.get("hot", []),
                "warm": metadata.get("warm", []),
                "cold": metadata.get("cold", [])
            }
            return self.generate_response(message, context_data)
        
        # Otherwise, skip
        return {"status": "skipped", "reason": "no_context_available"}