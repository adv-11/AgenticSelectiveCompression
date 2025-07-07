"""
Main Agent System - Coordinates all agents without LangGraph
"""

import logging
import time
from typing import Dict, Any, Optional

from .ingest_agent import IngestAgent
from .retrieval_agent import RetrievalAgent
from .orchestrator_agent import OrchestratorAgent
from .cleanup_agent import CleanupAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)


class AgentSystem:
    """
    Coordinates all agents in the memory management system
    """

    def __init__(self):
        self.ingest_agent = IngestAgent()
        self.retrieval_agent = RetrievalAgent()
        self.orchestrator_agent = OrchestratorAgent()
        self.cleanup_agent = CleanupAgent()
        
        # Set up agent chain
        self._setup_agent_chain()
        
        # Start cleanup agent
        self.cleanup_agent.start()
        
        logger.info("AgentSystem initialized successfully")

    def _setup_agent_chain(self):
        """Set up the message flow between agents"""
        
        # Ingest -> Retrieval
        self.ingest_agent.register_handler(self._handle_ingested_message)
        
        # Retrieval -> Orchestrator
        self.retrieval_agent.register_handler(self._handle_retrieved_message)
        
        # Orchestrator -> Final response
        self.orchestrator_agent.register_handler(self._handle_generated_response)

    def _handle_ingested_message(self, message_data: Dict[str, Any]):
        """Handle messages from ingest agent"""
        if message_data.get("status") == "processed":
            self.retrieval_agent.handle_ingested_message(message_data)

    def _handle_retrieved_message(self, retrieved_data: Dict[str, Any]):
        """Handle messages from retrieval agent"""
        if retrieved_data.get("status") == "retrieved":
            self.orchestrator_agent.handle_retrieved_message(retrieved_data)

    def _handle_generated_response(self, response_data: Dict[str, Any]):
        """Handle messages from orchestrator agent"""
        if response_data.get("status") == "generated":
            # This is where the final response would be output
            # For now, we'll just log it
            response = response_data.get("response", "")
            logger.info(f"[AgentSystem] Final response: {response[:100]}...")

    def process_user_message(self, user_message: str) -> str:
        """
        Process a user message through the entire agent chain
        
        Args:
            user_message: The user's input message
            
        Returns:
            The assistant's response
        """
        logger.info(f"[AgentSystem] Processing user message: {user_message[:100]}...")
        
        try:
            # Step 1: Ingest the message
            ingest_result = self.ingest_agent.process_message(user_message)
            
            if ingest_result.get("status") != "processed":
                return "Sorry, I couldn't process your message."
            
            # Step 2: Retrieve context
            retrieval_result = self.retrieval_agent.retrieve_context(user_message)
            
            if retrieval_result.get("status") != "retrieved":
                return "Sorry, I couldn't retrieve the necessary context."
            
            # Step 3: Generate response
            response_result = self.orchestrator_agent.generate_response(user_message, retrieval_result)
            
            if response_result.get("status") != "generated":
                return response_result.get("response", "Sorry, I couldn't generate a response.")
            
            return response_result.get("response", "")
            
        except Exception as e:
            logger.error(f"[AgentSystem] Error processing message: {e}")
            return "Sorry, I encountered an error while processing your message."

    def chat_loop(self):
        """Interactive chat loop"""
        print("\nAgentic Selective Compression System ready!")
        print("Type 'quit' to exit, 'stats' for memory statistics, 'cleanup' for garbage collection.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() == 'stats':
                    stats = self.cleanup_agent.get_memory_stats()
                    print(f"Memory Stats: {stats}")
                    continue
                
                if user_input.lower() == 'cleanup':
                    result = self.cleanup_agent.force_cleanup()
                    print(f"Cleanup Result: {result}")
                    continue
                
                # Process normal message
                response = self.process_user_message(user_input)
                print(f"Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"[AgentSystem] Chat loop error: {e}")
                print("Sorry, something went wrong. Please try again.")

    def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("[AgentSystem] Shutting down...")
        self.cleanup_agent.stop()
        logger.info("[AgentSystem] Shutdown complete")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


def main():
    """Main entry point"""
    try:
        with AgentSystem() as system:
            system.chat_loop()
    except Exception as e:
        logger.error(f"Failed to start agent system: {e}")
        print("Failed to start the agent system. Please check your configuration.")


if __name__ == "__main__":
    main()