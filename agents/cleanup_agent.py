import os
import time
import threading
import logging
from langgraph import Agent

from memory.memory_tier_manager import get_memory_manager

logger = logging.getLogger(__name__)


class CleanupAgent(Agent):
    """
    Spawns a background thread to run periodic garbage collection
    and tier migration on the shared MemoryTierManager.
    """

    def __init__(self, name: str = "cleanup_agent"):
        super().__init__(name=name)
        self.memory = get_memory_manager()
        # Interval in seconds (default 1h)
        self.interval = float(os.getenv("GC_INTERVAL", 3600))
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        logger.info(f"[{self.name}] GC thread every {self.interval}s started")

    def _loop(self):
        while True:
            logger.info(f"[{self.name}] Running garbage_collect()")
            try:
                self.memory.garbage_collect()
            except Exception as e:
                logger.error(f"[{self.name}] GC error: {e}")
            time.sleep(self.interval)
