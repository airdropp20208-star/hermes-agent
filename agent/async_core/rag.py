"""
RAG Pipeline — Retrieval-Augmented Generation.
Document loading, chunking, embedding, retrieval, context injection.
"""
import os
import re
import hashlib
import logging
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A loaded document."""
    id: str
    source: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    loaded_at: float = field(default_factory=time.time)
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.content)


@dataclass
class Chunk:
    """A document chunk for embedding and retrieval."""
    id: str
    doc_id: str
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)
    score: float = 0

    @property
    def token_estimate(self) -> int:
        return len(self.content) // 4


class DocumentLoader:
    """Multi-format document loader."""

    SUPPORTED_EXTENSIONS = {
        '.txt', '.md', '.py', '.js', '.ts', '.json', '.csv', '.yaml', '.yml',
        '.html', '.htm', '.xml', '.css', '.sh', '.bash', '.toml', '.ini',
        '.cfg', '.conf', '.log', '.rst', '.tex',
    }

    @staticmethod
    def load_file(path: str) -> Document:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError("File not found: %s" % path)
        content = p.read_text(encoding='utf-8', errors='replace')
        doc_id = hashlib.md5(("%s:%s" % (path, p.stat().st_mtime)).encode()).hexdigest()[:12]
        return Document(
            id=doc_id, source=str(p.absolute()), content=content,
            metadata={"filename": p.name, "extension": p.suffix, "size": p.stat().st_size},
        )

    @staticmethod
    def load_directory(path: str, recursive: bool = True) -> List[Document]:
        p = Path(path)
        if not p.is_dir():
            raise NotADirectoryError("Not a directory: %s" % path)
        docs = []
        pattern = "**/*" if recursive else "*"
        for f in p.glob(pattern):
            if f.is_file() and f.suffix.lower() in DocumentLoader.SUPPORTED_EXTENSIONS:
                try:
                    docs.append(DocumentLoader.load_file(str(f)))
                except Exception as e:
                    logger.warning("Failed to load %s: %s" % (f, e))
        return docs

    @staticmethod
    def load_text(text: str, source: str = "inline") -> Document:
        doc_id = hashlib.md5(("%s:%s" % (source, time.time())).encode()).hexdigest()[:12]
        return Document(id=doc_id, source=source, content=text)


class TextChunker:
    """Smart text chunker with sentence/paragraph/code-aware splitting."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200,
                 split_strategy: str = "sentence"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.split_strategy = split_strategy

    def chunk_document(self, doc: Document) -> List[Chunk]:
        if self.split_strategy == "paragraph":
            return self._chunk_by_paragraph(doc)
        elif self.split_strategy == "code":
            return self._chunk_by_code(doc)
        elif self.split_strategy == "fixed":
            return self._chunk_fixed(doc)
        return self._chunk_by_sentence(doc)

    def _chunk_by_sentence(self, doc: Document) -> List[Chunk]:
        sentences = re.split(r'(?<=[.!?])\s+', doc.content)
        chunks = []
        current = ""
        current_start = 0
        char_pos = 0
        for sentence in sentences:
            if len(current) + len(sentence) > self.chunk_size and current:
                chunk_id = hashlib.md5(("%s:%d" % (doc.id, len(chunks))).encode()).hexdigest()[:10]
                chunks.append(Chunk(
                    id=chunk_id, doc_id=doc.id, content=current.strip(),
                    index=len(chunks), start_char=current_start,
                    end_char=current_start + len(current),
                    metadata=dict(list(doc.metadata.items()) + [("chunk_method", "sentence")]),
                ))
                overlap_text = current[-self.overlap:] if self.overlap else ""
                current_start = char_pos - len(overlap_text)
                current = overlap_text
            current += sentence + " "
            char_pos += len(sentence) + 1
        if current.strip():
            chunk_id = hashlib.md5(("%s:%d" % (doc.id, len(chunks))).encode()).hexdigest()[:10]
            chunks.append(Chunk(
                id=chunk_id, doc_id=doc.id, content=current.strip(),
                index=len(chunks), start_char=current_start,
                end_char=current_start + len(current),
                metadata=dict(list(doc.metadata.items()) + [("chunk_method", "sentence")]),
            ))
        return chunks

    def _chunk_by_paragraph(self, doc: Document) -> List[Chunk]:
        paragraphs = doc.content.split("\n\n")
        chunks = []
        current = ""
        current_start = 0
        char_pos = 0
        for para in paragraphs:
            if len(current) + len(para) > self.chunk_size and current:
                chunk_id = hashlib.md5(("%s:%d" % (doc.id, len(chunks))).encode()).hexdigest()[:10]
                chunks.append(Chunk(
                    id=chunk_id, doc_id=doc.id, content=current.strip(),
                    index=len(chunks), start_char=current_start,
                    end_char=current_start + len(current),
                    metadata=dict(list(doc.metadata.items()) + [("chunk_method", "paragraph")]),
                ))
                current_start = char_pos
                current = ""
            current += para + "\n\n"
            char_pos += len(para) + 2
        if current.strip():
            chunk_id = hashlib.md5(("%s:%d" % (doc.id, len(chunks))).encode()).hexdigest()[:10]
            chunks.append(Chunk(
                id=chunk_id, doc_id=doc.id, content=current.strip(),
                index=len(chunks), start_char=current_start,
                end_char=current_start + len(current),
                metadata=dict(list(doc.metadata.items()) + [("chunk_method", "paragraph")]),
            ))
        return chunks

    def _chunk_by_code(self, doc: Document) -> List[Chunk]:
        boundaries = re.finditer(r'^(def |class |async def |\w+\s*=)', doc.content, re.MULTILINE)
        positions = [0] + [m.start() for m in boundaries] + [len(doc.content)]
        chunks = []
        for i in range(len(positions) - 1):
            segment = doc.content[positions[i]:positions[i+1]].strip()
            if not segment:
                continue
            if len(segment) > self.chunk_size * 2:
                sub_doc = Document(id=doc.id, source=doc.source, content=segment)
                sub_chunks = self._chunk_by_sentence(sub_doc)
                chunks.extend(sub_chunks)
            else:
                chunk_id = hashlib.md5(("%s:%d" % (doc.id, len(chunks))).encode()).hexdigest()[:10]
                chunks.append(Chunk(
                    id=chunk_id, doc_id=doc.id, content=segment,
                    index=len(chunks), start_char=positions[i], end_char=positions[i+1],
                    metadata=dict(list(doc.metadata.items()) + [("chunk_method", "code")]),
                ))
        return chunks

    def _chunk_fixed(self, doc: Document) -> List[Chunk]:
        chunks = []
        pos = 0
        while pos < len(doc.content):
            end = min(pos + self.chunk_size, len(doc.content))
            chunk_id = hashlib.md5(("%s:%d" % (doc.id, len(chunks))).encode()).hexdigest()[:10]
            chunks.append(Chunk(
                id=chunk_id, doc_id=doc.id, content=doc.content[pos:end],
                index=len(chunks), start_char=pos, end_char=end,
                metadata=dict(list(doc.metadata.items()) + [("chunk_method", "fixed")]),
            ))
            pos += self.chunk_size - self.overlap
        return chunks


class RAGRetriever:
    """RAG retriever with vector + keyword + hybrid search."""

    def __init__(self, embedder=None):
        self._chunks: Dict[str, Chunk] = {}
        self._doc_chunks: Dict[str, List[str]] = {}
        self._embedder = embedder
        self._chunker = TextChunker()

    async def add_document(self, doc: Document) -> int:
        chunks = self._chunker.chunk_document(doc)
        self._doc_chunks[doc.id] = []
        for chunk in chunks:
            if self._embedder:
                chunk.embedding = await self._embedder.embed(chunk.content)
            self._chunks[chunk.id] = chunk
            self._doc_chunks[doc.id].append(chunk.id)
        return len(chunks)

    async def add_directory(self, path: str, recursive: bool = True) -> int:
        docs = DocumentLoader.load_directory(path, recursive)
        total = 0
        for doc in docs:
            total += await self.add_document(doc)
        return total

    async def retrieve(self, query: str, top_k: int = 5,
                       strategy: str = "hybrid") -> List[Chunk]:
        if strategy == "vector":
            return await self._vector_search(query, top_k)
        elif strategy == "keyword":
            return self._keyword_search(query, top_k)
        elif strategy == "hybrid":
            vector_results = await self._vector_search(query, top_k * 2)
            keyword_results = self._keyword_search(query, top_k * 2)
            return self._merge_results(vector_results, keyword_results, top_k)
        return []

    async def _vector_search(self, query: str, top_k: int) -> List[Chunk]:
        if not self._embedder:
            return []
        query_vec = await self._embedder.embed(query)
        scored = []
        for chunk in self._chunks.values():
            if chunk.embedding:
                sim = sum(a * b for a, b in zip(query_vec, chunk.embedding))
                chunk.score = sim
                scored.append(chunk)
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    def _keyword_search(self, query: str, top_k: int) -> List[Chunk]:
        query_terms = set(query.lower().split())
        scored = []
        for chunk in self._chunks.values():
            content_lower = chunk.content.lower()
            score = sum(1 for term in query_terms if term in content_lower)
            if query.lower() in content_lower:
                score += 5
            if score > 0:
                chunk.score = score / len(query_terms) if query_terms else 0
                scored.append(chunk)
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    def _merge_results(self, vector: List[Chunk], keyword: List[Chunk],
                       top_k: int) -> List[Chunk]:
        k = 60
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}
        for rank, chunk in enumerate(vector):
            scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (k + rank + 1)
            chunk_map[chunk.id] = chunk
        for rank, chunk in enumerate(keyword):
            scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (k + rank + 1)
            chunk_map[chunk.id] = chunk
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        results = []
        for cid in sorted_ids[:top_k]:
            chunk = chunk_map[cid]
            chunk.score = scores[cid]
            results.append(chunk)
        return results

    def assemble_context(self, chunks: List[Chunk], max_tokens: int = 3000,
                         separator: str = "\n---\n") -> str:
        parts = []
        total = 0
        for chunk in chunks:
            tokens = chunk.token_estimate
            if total + tokens > max_tokens:
                break
            source = chunk.metadata.get("filename", chunk.doc_id)
            parts.append("[Source: %s]\n%s" % (source, chunk.content))
            total += tokens
        return separator.join(parts)

    def stats(self) -> Dict:
        return {
            "total_chunks": len(self._chunks),
            "total_documents": len(self._doc_chunks),
            "total_chars": sum(len(c.content) for c in self._chunks.values()),
        }
