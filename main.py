#!/usr/bin/env python3
"""
Main entry point for the Agentic Selective Compression System
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set the OPENAI_API_KEY environment variable.")
        print("You can:")
        print("1. Set it in your environment: export OPENAI_API_KEY=your_key_here")
        print("2. Create a .env file with: OPENAI_API_KEY=your_key_here")
        sys.exit(1)
    
    try:
        # Import and run the agent system
        from agents.main import AgentSystem
        
        print("Starting Agentic Selective Compression System...")
        
        with AgentSystem() as system:
            system.chat_loop()
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()