"""
Persistent Memory System — vector search, cross-session memory.
Supports multiple backends: in-memory, SQLite, vector DB.
"""
import json
import time
import hashlib
import sqlite3
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    content: str
    category: str = "general"       # user | system | fact | preference | procedure
    importance: float = 0.5         # 0.0 - 1.0
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    ttl: Optional[float] = None     # Seconds until expiry, None = forever
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl

    @property
    def relevance_score(self) -> float:
        """Combined score: importance * recency * frequency."""
        age_hours = (time.time() - self.last_accessed) / 3600
        recency = 1.0 / (1.0 + age_hours * 0.1)
        frequency = min(self.access_count / 10, 1.0)
        return self.importance * 0.5 + recency * 0.3 + frequency * 0.2


class VectorIndex:
    """Simple in-memory vector index with cosine similarity."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._vectors: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def add(self, id: str, vector: List[float]):
        with self._lock:
            self._vectors[id] = vector

    def remove(self, id: str):
        with self._lock:
            self._vectors.pop(id, None)

    def search(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Search by cosine similarity. Returns (id, score) pairs."""
        results = []
        q_norm = self._norm(query_vector)
        if q_norm == 0:
            return []

        with self._lock:
            for id, vec in self._vectors.items():
                v_norm = self._norm(vec)
                if v_norm == 0:
                    continue
                dot = sum(a * b for a, b in zip(query_vector, vec))
                similarity = dot / (q_norm * v_norm)
                results.append((id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @staticmethod
    def _norm(vec: List[float]) -> float:
        return sum(x * x for x in vec) ** 0.5


class MemoryStore:
    """
    Multi-backend memory store with:
    - Categorized memories (user, system, fact, preference, procedure)
    - Importance scoring
    - TTL support
    - Vector similarity search
    - SQLite persistence
    - Thread-safe operations
    - Auto-cleanup of expired entries
    """

    def __init__(self, db_path: Optional[str] = None, vector_dim: int = 384):
        self._memories: Dict[str, MemoryEntry] = {}
        self._vector_index = VectorIndex(vector_dim)
        self._lock = threading.Lock()
        self._db_path = db_path
        self._db: Optional[sqlite3.Connection] = None

        if db_path:
            self._init_db(db_path)

    def _init_db(self, path: str):
        """Initialize SQLite backend."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                created_at REAL,
                last_accessed REAL,
                ttl REAL,
                tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                embedding TEXT
            )
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON memories(category)
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance DESC)
        """)
        self._db.commit()
        self._load_from_db()

    def _load_from_db(self):
        """Load all memories from SQLite."""
        if not self._db:
            return
        cursor = self._db.execute("SELECT * FROM memories")
        for row in cursor.fetchall():
            entry = MemoryEntry(
                id=row[0], content=row[1], category=row[2],
                importance=row[3], access_count=row[4],
                created_at=row[5], last_accessed=row[6],
                ttl=row[7], tags=json.loads(row[8] or "[]"),
                metadata=json.loads(row[9] or "{}"),
            )
            if not entry.is_expired:
                self._memories[entry.id] = entry

    def add(self, content: str, category: str = "general", importance: float = 0.5,
            ttl: Optional[float] = None, tags: List[str] = None,
            metadata: Dict = None, embedding: List[float] = None) -> str:
        """Add a memory entry. Returns the memory ID."""
        mid = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:12]
        entry = MemoryEntry(
            id=mid, content=content, category=category,
            importance=importance, ttl=ttl, tags=tags or [],
            metadata=metadata or {}, embedding=embedding
        )

        with self._lock:
            self._memories[mid] = entry
            if embedding:
                self._vector_index.add(mid, embedding)

        if self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (mid, content, category, importance, 0, entry.created_at,
                 entry.last_accessed, ttl, json.dumps(tags or []),
                 json.dumps(metadata or {}), json.dumps(embedding) if embedding else None)
            )
            self._db.commit()

        return mid

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID."""
        entry = self._memories.get(memory_id)
        if entry and not entry.is_expired:
            entry.access_count += 1
            entry.last_accessed = time.time()
            return entry
        return None

    def search(self, query: str = "", category: str = None,
               tags: List[str] = None, top_k: int = 10,
               min_importance: float = 0) -> List[MemoryEntry]:
        """Search memories by text, category, tags, or importance."""
        results = []
        with self._lock:
            for entry in self._memories.values():
                if entry.is_expired:
                    continue
                if category and entry.category != category:
                    continue
                if tags and not set(tags).intersection(entry.tags):
                    continue
                if entry.importance < min_importance:
                    continue
                if query and query.lower() not in entry.content.lower():
                    continue
                results.append(entry)

        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results[:top_k]

    def search_vector(self, embedding: List[float], top_k: int = 10) -> List[Tuple[MemoryEntry, float]]:
        """Search by vector similarity."""
        hits = self._vector_index.search(embedding, top_k)
        results = []
        for mid, score in hits:
            entry = self._memories.get(mid)
            if entry and not entry.is_expired:
                entry.access_count += 1
                entry.last_accessed = time.time()
                results.append((entry, score))
        return results

    def update(self, memory_id: str, **kwargs) -> bool:
        """Update a memory entry."""
        entry = self._memories.get(memory_id)
        if not entry:
            return False
        for k, v in kwargs.items():
            if hasattr(entry, k):
                setattr(entry, k, v)
        if self._db:
            self._db.execute(
                "UPDATE memories SET content=?, category=?, importance=?, tags=?, metadata=? WHERE id=?",
                (entry.content, entry.category, entry.importance,
                 json.dumps(entry.tags), json.dumps(entry.metadata), memory_id)
            )
            self._db.commit()
        return True

    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        with self._lock:
            entry = self._memories.pop(memory_id, None)
            if entry:
                self._vector_index.remove(memory_id)
                if self._db:
                    self._db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
                    self._db.commit()
                return True
        return False

    def cleanup(self) -> int:
        """Remove expired entries. Returns count removed."""
        expired = [mid for mid, e in self._memories.items() if e.is_expired]
        for mid in expired:
            self.delete(mid)
        return len(expired)

    def get_context(self, max_tokens: int = 2000) -> str:
        """Get relevant memories as context string for LLM."""
        entries = sorted(
            [e for e in self._memories.values() if not e.is_expired],
            key=lambda e: e.relevance_score,
            reverse=True
        )
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # rough estimate

        for entry in entries:
            line = f"[{entry.category}] {entry.content}"
            if total_chars + len(line) > max_chars:
                break
            context_parts.append(line)
            total_chars += len(line)

        return "\n".join(context_parts)

    def stats(self) -> Dict:
        """Get memory store statistics."""
        by_category = defaultdict(int)
        for e in self._memories.values():
            if not e.is_expired:
                by_category[e.category] += 1
        return {
            "total": len(self._memories),
            "by_category": dict(by_category),
            "vector_index_size": len(self._vector_index._vectors),
            "expired": sum(1 for e in self._memories.values() if e.is_expired),
        }
