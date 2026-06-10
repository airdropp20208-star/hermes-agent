"""
Context Compressor — smart window management for long conversations.
Handles context window limits with summarization, sliding window, and priority-based pruning.
"""
import time
import logging
import hashlib
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    SLIDING_WINDOW = "sliding_window"    # Keep last N messages
    SUMMARIZE_OLD = "summarize_old"      # Summarize old messages
    PRIORITY_KEEP = "priority_keep"      # Keep high-priority messages
    SMART = "smart"                       # Combination strategy


@dataclass
class Message:
    """A conversation message with metadata."""
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    priority: float = 0.5  # 0=low, 1=high
    token_count: int = 0
    is_summary: bool = False
    summarized_ids: List[str] = field(default_factory=list)
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(
                ("%s:%s:%s" % (self.role, self.content[:100], self.timestamp)).encode()
            ).hexdigest()[:10]


class ContextCompressor:
    """
    Context window management with:
    - Sliding window (keep last N messages)
    - Summarization of old messages
    - Priority-based message retention
    - Token counting and budget management
    - System message anchoring (never remove system prompt)
    - Tool result deduplication
    - Message merging (combine consecutive short messages)
    """

    def __init__(self, max_tokens: int = 8000, strategy: CompressionStrategy = CompressionStrategy.SMART,
                 system_tokens: int = 1000):
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.system_tokens = system_tokens
        self._messages: List[Message] = []
        self._total_tokens = 0
        self._compressions = 0
        self._summarizer = None

    def set_summarizer(self, summarizer_fn):
        """Set the summarization function (async fn(messages) -> str)."""
        self._summarizer = summarizer_fn

    def add(self, role: str, content: str, priority: float = 0.5) -> Message:
        """Add a message to the context."""
        token_count = self._estimate_tokens(content)
        msg = Message(
            role=role, content=content,
            priority=priority, token_count=token_count,
        )
        self._messages.append(msg)
        self._total_tokens += token_count

        # Auto-compress if over budget
        if self._total_tokens > self.max_tokens:
            self._compress()

        return msg

    def get_messages(self) -> List[Dict]:
        """Get messages in API format."""
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def get_token_count(self) -> int:
        return self._total_tokens

    def _compress(self):
        """Compress messages based on strategy."""
        if self.strategy == CompressionStrategy.SLIDING_WINDOW:
            self._sliding_window()
        elif self.strategy == CompressionStrategy.SUMMARIZE_OLD:
            self._summarize_old()
        elif self.strategy == CompressionStrategy.PRIORITY_KEEP:
            self._priority_keep()
        elif self.strategy == CompressionStrategy.SMART:
            self._smart_compress()
        self._compressions += 1

    def _sliding_window(self):
        """Keep system message + last N messages that fit in budget."""
        system_msgs = [m for m in self._messages if m.role == "system"]
        other_msgs = [m for m in self._messages if m.role != "system"]

        budget = self.max_tokens - sum(m.token_count for m in system_msgs)
        kept = []
        used = 0

        for msg in reversed(other_msgs):
            if used + msg.token_count > budget:
                break
            kept.insert(0, msg)
            used += msg.token_count

        self._messages = system_msgs + kept
        self._total_tokens = sum(m.token_count for m in self._messages)

    def _priority_keep(self):
        """Keep system + highest priority messages within budget."""
        system_msgs = [m for m in self._messages if m.role == "system"]
        other_msgs = [m for m in self._messages if m.role != "system"]

        # Always keep last 4 messages (recent context)
        recent = other_msgs[-4:] if len(other_msgs) > 4 else other_msgs
        older = other_msgs[:-4] if len(other_msgs) > 4 else []

        # Sort older by priority
        older.sort(key=lambda m: m.priority, reverse=True)

        budget = self.max_tokens - sum(m.token_count for m in system_msgs) - sum(m.token_count for m in recent)
        kept_older = []
        used = 0

        for msg in older:
            if used + msg.token_count > budget:
                break
            kept_older.append(msg)
            used += msg.token_count

        # Maintain chronological order
        all_kept = kept_older + recent
        all_kept.sort(key=lambda m: m.timestamp)

        self._messages = system_msgs + all_kept
        self._total_tokens = sum(m.token_count for m in self._messages)

    def _summarize_old(self):
        """Summarize old messages, keep recent ones."""
        system_msgs = [m for m in self._messages if m.role == "system"]
        other_msgs = [m for m in self._messages if m.role != "system"]

        if len(other_msgs) <= 6:
            return  # Not enough to summarize

        # Keep last 6 messages
        recent = other_msgs[-6:]
        to_summarize = other_msgs[:-6]

        # Create placeholder summary
        summary_content = "[Summary of %d earlier messages]" % len(to_summarize)
        if self._summarizer:
            # Would call async summarizer here
            pass

        summary = Message(
            role="system", content=summary_content,
            priority=0.3, is_summary=True,
            summarized_ids=[m.id for m in to_summarize],
        )
        summary.token_count = self._estimate_tokens(summary_content)

        self._messages = system_msgs + [summary] + recent
        self._total_tokens = sum(m.token_count for m in self._messages)

    def _smart_compress(self):
        """Combination: summarize old + priority keep recent."""
        if self._total_tokens <= self.max_tokens:
            return

        # First try priority-based
        self._priority_keep()

        # If still over, summarize
        if self._total_tokens > self.max_tokens:
            self._summarize_old()

        # If STILL over, sliding window
        if self._total_tokens > self.max_tokens:
            self._sliding_window()

    def deduplicate_tool_results(self):
        """Remove duplicate tool results (same tool, same args)."""
        seen = {}
        to_remove = []
        for i, msg in enumerate(self._messages):
            if msg.role == "tool":
                key = hashlib.md5(msg.content[:200].encode()).hexdigest()
                if key in seen:
                    to_remove.append(i)
                else:
                    seen[key] = i

        for i in reversed(to_remove):
            removed = self._messages.pop(i)
            self._total_tokens -= removed.token_count

    def merge_consecutive(self, role: str = "user", max_gap_seconds: float = 5):
        """Merge consecutive short messages from same role."""
        merged = []
        i = 0
        while i < len(self._messages):
            msg = self._messages[i]
            if msg.role == role and len(msg.content) < 100:
                # Look ahead for mergeable messages
                parts = [msg.content]
                j = i + 1
                while (j < len(self._messages) and
                       self._messages[j].role == role and
                       len(self._messages[j].content) < 100 and
                       self._messages[j].timestamp - self._messages[j-1].timestamp < max_gap_seconds):
                    parts.append(self._messages[j].content)
                    j += 1
                if len(parts) > 1:
                    merged_content = "\n".join(parts)
                    merged_msg = Message(
                        role=role, content=merged_content,
                        priority=msg.priority,
                        token_count=self._estimate_tokens(merged_content),
                    )
                    merged.append(merged_msg)
                    i = j
                    continue
            merged.append(msg)
            i += 1

        self._messages = merged
        self._total_tokens = sum(m.token_count for m in self._messages)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count (rough: 1 token per 4 chars)."""
        return len(text) // 4 + 1

    def stats(self) -> Dict:
        return {
            "messages": len(self._messages),
            "total_tokens": self._total_tokens,
            "max_tokens": self.max_tokens,
            "utilization": round(self._total_tokens / self.max_tokens * 100, 1),
            "compressions": self._compressions,
            "strategy": self.strategy.value,
        }
