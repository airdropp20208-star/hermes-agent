"""
Embeddings — local TF-IDF + API-based for vector search.
Provides embedding generation without requiring heavy ML deps.
"""
import math
import re
import hashlib
import logging
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class TFIDFEmbedder:
    """
    Lightweight TF-IDF embedder — no ML dependencies.
    Generates sparse vectors that can be used for similarity search.
    Good enough for memory search; use API embedder for production.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._idf: Dict[str, float] = {}
        self._doc_count = 0
        self._vocab: Dict[str, int] = {}
        self._vocab_size = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, split, remove short words."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]

    def _hash_to_dim(self, token: str) -> int:
        """Hash token to a dimension index."""
        h = hashlib.md5(token.encode()).hexdigest()
        return int(h, 16) % self.dimension

    def fit(self, documents: List[str]):
        """Fit TF-IDF on a corpus."""
        doc_freq: Dict[str, int] = Counter()
        self._doc_count = len(documents)

        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] += 1

        # Compute IDF
        for token, freq in doc_freq.items():
            self._idf[token] = math.log((self._doc_count + 1) / (freq + 1)) + 1

    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.dimension

        # TF
        tf = Counter(tokens)
        total = len(tokens)

        # Build vector using random projection (hashing trick)
        vector = [0.0] * self.dimension
        for token, count in tf.items():
            tf_score = count / total
            idf_score = self._idf.get(token, math.log(self._doc_count + 2) + 1)
            score = tf_score * idf_score

            idx = self._hash_to_dim(token)
            # Use sign hash for better distribution
            sign = 1 if int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % 2 == 0 else -1
            vector[idx] += sign * score

        # Normalize
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        return [self.embed(t) for t in texts]


class APIEmbedder:
    """
    API-based embedder using OpenAI-compatible embedding endpoints.
    Higher quality than TF-IDF but requires API key.
    """

    def __init__(self, api_key: str = "", base_url: str = "",
                 model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.model = model
        self._client = None

    async def _get_client(self):
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(timeout=30)
            except ImportError:
                pass
        return self._client

    async def embed(self, text: str) -> List[float]:
        """Generate embedding via API."""
        results = await self.embed_batch([text])
        return results[0] if results else []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        client = await self._get_client()
        if not client:
            logger.warning("httpx not available, falling back to TF-IDF")
            fallback = TFIDFEmbedder()
            return fallback.embed_batch(texts)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "input": texts,
        }

        resp = await client.post(
            f"{self.base_url}/embeddings",
            json=body, headers=headers
        )
        resp.raise_for_status()
        data = resp.json()

        # Sort by index to maintain order
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        return [e["embedding"] for e in embeddings]

    async def close(self):
        if self._client:
            await self._client.aclose()


class HybridEmbedder:
    """
    Combines TF-IDF (fast, local) with API (high quality).
    Uses TF-IDF for quick pre-filtering, API for final ranking.
    """

    def __init__(self, api_key: str = "", base_url: str = "",
                 model: str = "text-embedding-3-small",
                 dimension: int = 384):
        self.tfidf = TFIDFEmbedder(dimension)
        self.api = APIEmbedder(api_key, base_url, model) if api_key else None
        self.dimension = dimension

    def fit(self, documents: List[str]):
        """Fit TF-IDF on corpus."""
        self.tfidf.fit(documents)

    async def embed(self, text: str) -> List[float]:
        """Generate embedding — prefers API, falls back to TF-IDF."""
        if self.api and self.api.api_key:
            try:
                return await self.api.embed(text)
            except Exception as e:
                logger.warning(f"API embedding failed, using TF-IDF: {e}")
        return self.tfidf.embed(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        if self.api and self.api.api_key:
            try:
                return await self.api.embed_batch(texts)
            except Exception as e:
                logger.warning(f"API batch embedding failed, using TF-IDF: {e}")
        return self.tfidf.embed_batch(texts)

    async def close(self):
        if self.api:
            await self.api.close()
