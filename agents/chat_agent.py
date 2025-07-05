"""
Simple Chat Agent - Standalone implementation without LangGraph
"""

import os
import logging
import time
from typing import Dict, List
from openai import OpenAI
from memory.memory_tier_manager import get_memory_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

class SimpleChatAgent:
    """
    A simple chat agent that:
    1. Stores user messages in memory tiers
    2. Retrieves relevant context from all tiers
    3. Generates responses using OpenAI
    """

    def __init__(self, name: str = "simple_chat_agent"):
        self.name = name
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)
        
        # Initialize memory manager
        self.memory = get_memory_manager()
        
        logger.info(f"Initialized {self.name}")

    def process_message(self, user_message: str) -> str:
        """
        Process a user message and return AI response
        """
        logger.info(f"Processing message: {user_message}")
        
        # Store user message in memory
        try:
            tier = self.memory.add(user_message)
            logger.info(f"Stored message in '{tier}' tier")
        except Exception as e:
            logger.error(f"Failed to store message: {e}")
        
        # Retrieve context from all tiers
        hot_context = self.memory.get_hot(top_k=3)
        warm_context = self.memory.get_warm(user_message, top_k=2)
        cold_context = self.memory.get_cold(user_message, top_k=2)
        
        # Build context sections
        context_sections = []
        
        if hot_context:
            hot_section = "Recent conversation:\n" + "\n".join(f"- {item}" for item in hot_context)
            context_sections.append(hot_section)
        
        if warm_context:
            warm_section = "Relevant summaries:\n" + "\n".join(
                f"- {item['summary']}" for item in warm_context
            )
            context_sections.append(warm_section)
        
        if cold_context:
            cold_section = "Related archived content:\n" + "\n".join(
                f"- {item['content'][:200]}..." for item in cold_context
            )
            context_sections.append(cold_section)
        
        # Build system prompt with context
        system_prompt = "You are a helpful assistant."
        if context_sections:
            system_prompt += "\n\nContext from memory:\n" + "\n\n".join(context_sections)
        
        # Generate response
        try:
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
            logger.info(f"Generated response: {ai_response}")
            
            # Store AI response in memory as well
            self.memory.add(f"Assistant: {ai_response}")
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I'm sorry, I encountered an error while processing your message."

    def chat_loop(self):
        """
        Interactive chat loop for testing
        """
        print(f"\n{self.name} ready! Type 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                response = self.process_message(user_input)
                print(f"Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print("Sorry, something went wrong. Please try again.")

def main():
    """
    Main entry point for the chat agent
    """
    try:
        agent = SimpleChatAgent()
        agent.chat_loop()
    except Exception as e:
        logger.error(f"Failed to start chat agent: {e}")
        print("Failed to start the chat agent. Please check your configuration.")

if __name__ == "__main__":
    main()