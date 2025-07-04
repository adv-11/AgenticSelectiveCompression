"""
MemoryTierManager - Core memory management system
"""

import os
import sqlite3
import json
import logging
import time
import threading
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
from openai import OpenAI

try:
    import faiss
except ImportError:
    faiss = None
    logging.warning("FAISS not available, using basic similarity search")

from .adaptive_control_system import AdaptiveControlSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

class MemoryTierManager:
    """
    Manages hot, warm, and cold memory tiers with SQLite and optional FAISS indexing
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        adaptive_control: Optional[AdaptiveControlSystem] = None,
        embed_dim: int = 1536,
        stats_window: int = 20
    ):
        self.db_path = db_path
        self.adaptive_control = adaptive_control or AdaptiveControlSystem()
        self.embed_dim = embed_dim
        self.stats_window = stats_window
        self.lock = threading.Lock()

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)

        # Usage stats for instrumentation
        self.usage_stats: Dict[str, List[float]] = {
            "hot": [], "warm": [], "cold": []
        }

        # Initialize database
        self._init_database()
        
        # Initialize FAISS indices if available
        if faiss:
            self._init_faiss_indices()
        else:
            self.cold_index = None
            self.warm_index = None

    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Hot memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hot_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Warm memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS warm_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary TEXT NOT NULL,
                    original_content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Cold memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cold_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    embedding_json TEXT NOT NULL,
                    original_content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")

    def _init_faiss_indices(self):
        """Initialize FAISS indices for warm and cold memory"""
        if not faiss:
            return
            
        # Cold memory index
        self.cold_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embed_dim))
        
        # Warm memory index
        self.warm_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embed_dim))
        
        # Load existing embeddings
        self._load_existing_embeddings()

    def _load_existing_embeddings(self):
        """Load existing embeddings into FAISS indices"""
        if not faiss:
            return
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Load cold embeddings
            cursor.execute("SELECT id, embedding_json FROM cold_memory")
            for row_id, emb_json in cursor.fetchall():
                try:
                    vec = np.array(json.loads(emb_json), dtype=np.float32)
                    vec = vec / np.linalg.norm(vec)  # Normalize
                    self.cold_index.add_with_ids(
                        vec.reshape(1, -1),
                        np.array([row_id], dtype=np.int64)
                    )
                except Exception as e:
                    logger.warning(f"Failed to load cold embedding {row_id}: {e}")
            
            # Load warm embeddings
            cursor.execute("SELECT id, embedding_json FROM warm_memory")
            for row_id, emb_json in cursor.fetchall():
                try:
                    vec = np.array(json.loads(emb_json), dtype=np.float32)
                    vec = vec / np.linalg.norm(vec)  # Normalize
                    self.warm_index.add_with_ids(
                        vec.reshape(1, -1),
                        np.array([row_id], dtype=np.int64)
                    )
                except Exception as e:
                    logger.warning(f"Failed to load warm embedding {row_id}: {e}")
                    
        logger.info(f"Loaded {self.cold_index.ntotal} cold and {self.warm_index.ntotal} warm embeddings")

    def add(self, text: str) -> str:
        """Add text to appropriate memory tier"""
        with self.lock:
            tier = self.adaptive_control.classify(text)
            logger.info(f"Storing text in tier '{tier}'")
            
            try:
                if tier == "hot":
                    self._add_hot(text)
                elif tier == "warm":
                    summary = self._summarize(text)
                    emb_json = self._embed(summary)
                    self._add_warm(summary, text, emb_json)
                else:  # cold
                    emb_json = self._embed(text)
                    self._add_cold(emb_json, text)
                    
                return tier
            except Exception as e:
                logger.error(f"Failed to add text to {tier} tier: {e}")
                return "error"

    def _add_hot(self, text: str):
        """Add text to hot memory"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO hot_memory (content) VALUES (?)", (text,))
            conn.commit()

    def _summarize(self, text: str) -> str:
        """Generate summary using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful summarizer. Provide a concise summary."},
                    {"role": "user", "content": f"Summarize the following text in one sentence:\n{text}"}
                ],
                max_tokens=60,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return text[:100] + "..." if len(text) > 100 else text

    def _embed(self, text: str) -> str:
        """Generate embedding using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return json.dumps(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            # Return a dummy embedding
            return json.dumps([0.0] * self.embed_dim)

    def _add_warm(self, summary: str, original: str, emb_json: str):
        """Add to warm memory with embedding"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO warm_memory (summary, original_content, embedding_json) VALUES (?, ?, ?)",
                (summary, original, emb_json)
            )
            row_id = cursor.lastrowid
            conn.commit()
            
            # Add to FAISS index
            if faiss and self.warm_index:
                try:
                    vec = np.array(json.loads(emb_json), dtype=np.float32)
                    vec = vec / np.linalg.norm(vec)
                    self.warm_index.add_with_ids(
                        vec.reshape(1, -1),
                        np.array([row_id], dtype=np.int64)
                    )
                except Exception as e:
                    logger.warning(f"Failed to add warm embedding to FAISS: {e}")

    def _add_cold(self, emb_json: str, original: str):
        """Add to cold memory with embedding"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO cold_memory (embedding_json, original_content) VALUES (?, ?)",
                (emb_json, original)
            )
            row_id = cursor.lastrowid
            conn.commit()
            
            # Add to FAISS index
            if faiss and self.cold_index:
                try:
                    vec = np.array(json.loads(emb_json), dtype=np.float32)
                    vec = vec / np.linalg.norm(vec)
                    self.cold_index.add_with_ids(
                        vec.reshape(1, -1),
                        np.array([row_id], dtype=np.int64)
                    )
                except Exception as e:
                    logger.warning(f"Failed to add cold embedding to FAISS: {e}")

    def _record_latency(self, tier: str, ms: float):
        """Record retrieval latency for adaptive control"""
        stats = self.usage_stats[tier]
        stats.append(ms)
        
        if len(stats) >= self.stats_window:
            self.adaptive_control.adjust_thresholds(self.usage_stats)
            # Clear stats after adjustment
            for k in self.usage_stats:
                self.usage_stats[k].clear()

    def get_hot(self, top_k: int = 5) -> List[str]:
        """Retrieve from hot memory"""
        start_time = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content FROM hot_memory ORDER BY timestamp DESC LIMIT ?",
                (top_k,)
            )
            results = [row[0] for row in cursor.fetchall()]
        
        elapsed = (time.time() - start_time) * 1000
        self._record_latency("hot", elapsed)
        return results

    def get_warm(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve from warm memory using semantic search"""
        start_time = time.time()
        
        if not faiss or not self.warm_index:
            return self._get_warm_fallback(query, top_k)
        
        try:
            # Get query embedding
            query_emb = self._embed(query)
            query_vec = np.array(json.loads(query_emb), dtype=np.float32)
            query_vec = query_vec / np.linalg.norm(query_vec)
            
            # Search FAISS index
            scores, indices = self.warm_index.search(query_vec.reshape(1, -1), top_k)
            
            # Retrieve full records
            results = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for idx, score in zip(indices[0], scores[0]):
                    if idx == -1:
                        continue
                    cursor.execute(
                        "SELECT summary, original_content, timestamp FROM warm_memory WHERE id = ?",
                        (int(idx),)
                    )
                    row = cursor.fetchone()
                    if row:
                        results.append({
                            "id": int(idx),
                            "summary": row[0],
                            "content": row[1],
                            "timestamp": row[2],
                            "score": float(score)
                        })
            
            elapsed = (time.time() - start_time) * 1000
            self._record_latency("warm", elapsed)
            return results
            
        except Exception as e:
            logger.error(f"Warm retrieval failed: {e}")
            return self._get_warm_fallback(query, top_k)

    def _get_warm_fallback(self, query: str, top_k: int) -> List[Dict]:
        """Fallback warm retrieval without FAISS"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, summary, original_content, timestamp FROM warm_memory "
                "ORDER BY timestamp DESC LIMIT ?",
                (top_k,)
            )
            return [
                {
                    "id": row[0],
                    "summary": row[1],
                    "content": row[2],
                    "timestamp": row[3],
                    "score": 1.0
                }
                for row in cursor.fetchall()
            ]

    def get_cold(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve from cold memory using semantic search"""
        start_time = time.time()
        
        if not faiss or not self.cold_index:
            return self._get_cold_fallback(query, top_k)
        
        try:
            # Get query embedding
            query_emb = self._embed(query)
            query_vec = np.array(json.loads(query_emb), dtype=np.float32)
            query_vec = query_vec / np.linalg.norm(query_vec)
            
            # Search FAISS index
            scores, indices = self.cold_index.search(query_vec.reshape(1, -1), top_k)
            
            # Retrieve full records
            results = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for idx, score in zip(indices[0], scores[0]):
                    if idx == -1:
                        continue
                    cursor.execute(
                        "SELECT original_content, timestamp FROM cold_memory WHERE id = ?",
                        (int(idx),)
                    )
                    row = cursor.fetchone()
                    if row:
                        results.append({
                            "id": int(idx),
                            "content": row[0],
                            "timestamp": row[1],
                            "score": float(score)
                        })
            
            elapsed = (time.time() - start_time) * 1000
            self._record_latency("cold", elapsed)
            return results
            
        except Exception as e:
            logger.error(f"Cold retrieval failed: {e}")
            return self._get_cold_fallback(query, top_k)

    def _get_cold_fallback(self, query: str, top_k: int) -> List[Dict]:
        """Fallback cold retrieval without FAISS"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, original_content, timestamp FROM cold_memory "
                "ORDER BY timestamp DESC LIMIT ?",
                (top_k,)
            )
            return [
                {
                    "id": row[0],
                    "content": row[1],
                    "timestamp": row[2],
                    "score": 1.0
                }
                for row in cursor.fetchall()
            ]

    def garbage_collect(
        self,
        hot_to_warm_age: float = 3600,        # 1 hour
        warm_to_cold_age: float = 86400,      # 1 day
        cold_prune_age: float = 7 * 86400     # 1 week
    ):
        """
        Migrate and prune memory tiers based on age
        """
        with self.lock:
            logger.info("Starting garbage collection")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Hot → Warm migration
                hot_threshold_days = hot_to_warm_age / 86400
                cursor.execute(
                    "SELECT id, content FROM hot_memory "
                    "WHERE (julianday('now') - julianday(timestamp)) > ?",
                    (hot_threshold_days,)
                )
                
                for row_id, content in cursor.fetchall():
                    try:
                        summary = self._summarize(content)
                        emb_json = self._embed(summary)
                        self._add_warm(summary, content, emb_json)
                        cursor.execute("DELETE FROM hot_memory WHERE id = ?", (row_id,))
                        logger.info(f"Migrated hot entry {row_id} to warm")
                    except Exception as e:
                        logger.error(f"Failed to migrate hot entry {row_id}: {e}")
                
                # Warm → Cold migration
                warm_threshold_days = warm_to_cold_age / 86400
                cursor.execute(
                    "SELECT id, original_content FROM warm_memory "
                    "WHERE (julianday('now') - julianday(timestamp)) > ?",
                    (warm_threshold_days,)
                )
                
                for row_id, original in cursor.fetchall():
                    try:
                        emb_json = self._embed(original)
                        self._add_cold(emb_json, original)
                        cursor.execute("DELETE FROM warm_memory WHERE id = ?", (row_id,))
                        logger.info(f"Migrated warm entry {row_id} to cold")
                    except Exception as e:
                        logger.error(f"Failed to migrate warm entry {row_id}: {e}")
                
                # Cold pruning
                cold_threshold_days = cold_prune_age / 86400
                cursor.execute(
                    "SELECT id FROM cold_memory "
                    "WHERE (julianday('now') - julianday(timestamp)) > ?",
                    (cold_threshold_days,)
                )
                
                for (row_id,) in cursor.fetchall():
                    try:
                        if faiss and self.cold_index:
                            self.cold_index.remove_ids(np.array([row_id], dtype=np.int64))
                        cursor.execute("DELETE FROM cold_memory WHERE id = ?", (row_id,))
                        logger.info(f"Pruned cold entry {row_id}")
                    except Exception as e:
                        logger.error(f"Failed to prune cold entry {row_id}: {e}")
                
                conn.commit()
            
            logger.info("Garbage collection completed")


# Singleton pattern for shared manager
_shared_manager: Optional[MemoryTierManager] = None
_manager_lock = threading.Lock()

def get_memory_manager() -> MemoryTierManager:
    """Get or create the shared MemoryTierManager instance"""
    global _shared_manager
    
    with _manager_lock:
        if _shared_manager is None:
            _shared_manager = MemoryTierManager()
        return _shared_manager