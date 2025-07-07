"""
Cleanup Agent - Handles periodic memory management and garbage collection
"""

import os
import time
import threading
import logging
from typing import Dict, Any
from memory.memory_tier_manager import get_memory_manager

logger = logging.getLogger(__name__)


class CleanupAgent:
    """
    Handles periodic garbage collection and memory tier management
    """

    def __init__(self, name: str = "cleanup_agent"):
        self.name = name
        self.memory = get_memory_manager()
        self.message_handlers = []
        self.lock = threading.Lock()
        
        # Configuration
        self.interval = float(os.getenv("GC_INTERVAL", "3600"))  # Default 1 hour
        self.running = False
        self.gc_thread = None
        
        logger.info(f"[{self.name}] Initialized with {self.interval}s interval")

    def register_handler(self, handler):
        """Register a message handler function"""
        with self.lock:
            self.message_handlers.append(handler)

    def start(self):
        """Start the cleanup background thread"""
        if self.running:
            logger.warning(f"[{self.name}] Already running")
            return
        
        self.running = True
        self.gc_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.gc_thread.start()
        logger.info(f"[{self.name}] Started cleanup thread")

    def stop(self):
        """Stop the cleanup background thread"""
        if not self.running:
            return
        
        self.running = False
        if self.gc_thread:
            self.gc_thread.join(timeout=5)
        logger.info(f"[{self.name}] Stopped cleanup thread")

    def _cleanup_loop(self):
        """Main cleanup loop running in background thread"""
        while self.running:
            try:
                time.sleep(self.interval)
                if not self.running:
                    break
                
                logger.info(f"[{self.name}] Running garbage collection")
                self.run_garbage_collection()
                
            except Exception as e:
                logger.error(f"[{self.name}] Cleanup loop error: {e}")

    def run_garbage_collection(self) -> Dict[str, Any]:
        """
        Run garbage collection and tier migration
        
        Returns:
            Dict containing garbage collection results
        """
        logger.info(f"[{self.name}] Starting garbage collection")
        
        try:
            # Run memory garbage collection
            self.memory.garbage_collect()
            
            result = {
                "status": "completed",
                "timestamp": time.time(),
                "action": "garbage_collection"
            }
            
            logger.info(f"[{self.name}] Garbage collection completed")
            
            # Notify handlers
            with self.lock:
                for handler in self.message_handlers:
                    try:
                        handler(result)
                    except Exception as e:
                        logger.error(f"[{self.name}] Handler error: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"[{self.name}] Garbage collection error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
                "action": "garbage_collection"
            }

    def force_cleanup(self) -> Dict[str, Any]:
        """
        Force immediate cleanup (useful for testing)
        
        Returns:
            Dict containing cleanup results
        """
        logger.info(f"[{self.name}] Force cleanup requested")
        return self.run_garbage_collection()

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics
        
        Returns:
            Dict containing memory statistics
        """
        try:
            # Get basic counts from each tier
            hot_items = self.memory.get_hot(top_k=1000)  # Get all hot items
            warm_items = self.memory.get_warm("", top_k=1000)  # Get all warm items
            cold_items = self.memory.get_cold("", top_k=1000)  # Get all cold items
            
            stats = {
                "hot_count": len(hot_items),
                "warm_count": len(warm_items),
                "cold_count": len(cold_items),
                "total_count": len(hot_items) + len(warm_items) + len(cold_items),
                "timestamp": time.time()
            }
            
            logger.info(f"[{self.name}] Memory stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"[{self.name}] Failed to get memory stats: {e}")
            return {
                "error": str(e),
                "timestamp": time.time()
            }

    def process_message(self, message: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process cleanup-related messages
        
        Args:
            message: The message text
            metadata: Optional metadata
            
        Returns:
            Dict containing processing results
        """
        metadata = metadata or {}
        
        # Handle cleanup commands
        if message.lower().strip() in ["cleanup", "gc", "garbage_collect"]:
            return self.force_cleanup()
        elif message.lower().strip() in ["stats", "memory_stats"]:
            return self.get_memory_stats()
        
        # Otherwise, skip
        return {"status": "skipped", "reason": "not_cleanup_command"}

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()