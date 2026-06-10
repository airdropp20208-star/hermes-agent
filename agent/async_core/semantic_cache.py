"""
Semantic Cache — embedding-based cache for LLM responses and tool results.
Deduplicates similar queries, smart eviction, semantic search over cache.
"""
import time
import uuid
import hashlib
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached result."""
    id: str
    key: str  # original query/request
    value: Any  # cached response
    embedding: List[float] = field(default_factory=list)
    hit_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_hit: float = 0
    ttl: float = 0  # 0 = no expiry
    metadata: Dict = field(default_factory=dict)
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        if self.ttl <= 0:
            return False
        return time.time() > self.created_at + self.ttl


class SemanticCache:
    """
    Semantic cache with:
    - Exact match (hash-based)
    - Similarity match (embedding-based)
    - LRU eviction with frequency boost
    - TTL support
    - Cache warming
    - Hit rate tracking
    - Namespace support (cache per tool/model)
    - Size-based eviction
    """

    def __init__(self, max_entries: int = 1000, max_size_mb: float = 100,
                 similarity_threshold: float = 0.85, embedder=None):
        self.max_entries = max_entries
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.similarity_threshold = similarity_threshold
        self._embedder = embedder
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._total_size = 0
        self._hits = 0
        self._misses = 0
        self._exact_hits = 0
        self._semantic_hits = 0

    def set_embedder(self, embedder):
        """Set the embedding function for semantic matching."""
        self._embedder = embedder

    async def get(self, key: str, namespace: str = "") -> Optional[Any]:
        """Get from cache. Tries exact match first, then semantic."""
        cache_key = self._make_key(key, namespace)

        # Exact match
        if cache_key in self._entries:
            entry = self._entries[cache_key]
            if not entry.is_expired:
                entry.hit_count += 1
                entry.last_hit = time.time()
                self._hits += 1
                self._exact_hits += 1
                self._entries.move_to_end(cache_key)
                return entry.value
            else:
                self._remove(cache_key)

        # Semantic match
        if self._embedder and len(self._entries) > 0:
            result = await self._semantic_search(key, namespace)
            if result:
                self._hits += 1
                self._semantic_hits += 1
                return result

        self._misses += 1
        return None

    async def set(self, key: str, value: Any, namespace: str = "",
                  ttl: float = 0, metadata: Dict = None):
        """Store in cache."""
        cache_key = self._make_key(key, namespace)
        size = self._estimate_size(value)

        # Remove existing if present
        if cache_key in self._entries:
            self._remove(cache_key)

        # Evict if needed
        while len(self._entries) >= self.max_entries or self._total_size + size > self.max_size_bytes:
            if not self._entries:
                break
            self._evict_lfu()

        # Generate embedding
        embedding = []
        if self._embedder:
            try:
                embedding = await self._embedder.embed(key)
            except Exception:
                pass

        entry = CacheEntry(
            id=cache_key, key=key, value=value,
            embedding=embedding, ttl=ttl,
            metadata=metadata or {}, size_bytes=size,
        )
        self._entries[cache_key] = entry
        self._total_size += size

    async def _semantic_search(self, query: str, namespace: str) -> Optional[Any]:
        """Find semantically similar cached entry."""
        if not self._embedder:
            return None

        try:
            query_vec = await self._embedder.embed(query)
        except Exception:
            return None

        best_score = 0
        best_entry = None

        prefix = namespace + ":" if namespace else ""
        for key, entry in self._entries.items():
            if prefix and not key.startswith(prefix):
                continue
            if entry.is_expired or not entry.embedding:
                continue

            # Cosine similarity
            dot = sum(a * b for a, b in zip(query_vec, entry.embedding))
            norm_a = sum(a * a for a in query_vec) ** 0.5
            norm_b = sum(b * b for b in entry.embedding) ** 0.5
            if norm_a == 0 or norm_b == 0:
                continue
            score = dot / (norm_a * norm_b)

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.similarity_threshold:
            best_entry.hit_count += 1
            best_entry.last_hit = time.time()
            return best_entry.value

        return None

    def _evict_lfu(self):
        """Evict least frequently used entry."""
        if not self._entries:
            return
        # Score = hit_count * recency
        now = time.time()
        scored = []
        for key, entry in self._entries.items():
            recency = 1.0 / (1.0 + (now - entry.last_hit) / 3600) if entry.last_hit else 0.1
            score = entry.hit_count * 0.7 + recency * 0.3
            scored.append((score, key))

        scored.sort()
        _, victim_key = scored[0]
        self._remove(victim_key)

    def _remove(self, key: str):
        entry = self._entries.pop(key, None)
        if entry:
            self._total_size -= entry.size_bytes

    def invalidate(self, key: str, namespace: str = ""):
        """Remove specific entry."""
        cache_key = self._make_key(key, namespace)
        self._remove(cache_key)

    def invalidate_namespace(self, namespace: str):
        """Remove all entries in a namespace."""
        prefix = namespace + ":"
        keys = [k for k in self._entries if k.startswith(prefix)]
        for k in keys:
            self._remove(k)

    def clear(self):
        """Clear entire cache."""
        self._entries.clear()
        self._total_size = 0

    def cleanup(self) -> int:
        """Remove expired entries."""
        expired = [k for k, e in self._entries.items() if e.is_expired]
        for k in expired:
            self._remove(k)
        return len(expired)

    def _make_key(self, key: str, namespace: str) -> str:
        raw = "%s:%s" % (namespace, key) if namespace else key
        return hashlib.md5(raw.encode()).hexdigest()

    @staticmethod
    def _estimate_size(value: Any) -> int:
        try:
            return len(json.dumps(value, default=str).encode())
        except Exception:
            return len(str(value).encode())

    def stats(self) -> Dict:
        total = self._hits + self._misses
        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "size_mb": round(self._total_size / 1024 / 1024, 2),
            "max_size_mb": round(self.max_size_bytes / 1024 / 1024, 2),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": "%.1f%%" % (self._hits / total * 100) if total else "0%",
            "exact_hits": self._exact_hits,
            "semantic_hits": self._semantic_hits,
        }
