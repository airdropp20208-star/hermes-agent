"""
Streaming Engine — SSE, WebSocket, real-time response delivery.
Supports: Server-Sent Events, WebSocket, chunked transfer.
"""
import asyncio
import json
import time
import logging
import uuid
from typing import Optional, Dict, Any, List, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class StreamType(Enum):
    SSE = "sse"              # Server-Sent Events
    WEBSOCKET = "websocket"  # WebSocket
    CHUNKED = "chunked"      # HTTP chunked transfer
    CALLBACK = "callback"    # Direct callback


@dataclass
class StreamChunk:
    """A single chunk of streamed data."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    data: Any = None
    event: str = "message"     # SSE event type
    retry: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    is_final: bool = False

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        lines = []
        if self.event != "message":
            lines.append(f"event: {self.event}")
        if self.id:
            lines.append(f"id: {self.id}")
        if self.retry:
            lines.append(f"retry: {self.retry}")
        data = json.dumps(self.data) if not isinstance(self.data, str) else self.data
        lines.append(f"data: {data}")
        return "\n".join(lines) + "\n\n"

    def to_json(self) -> str:
        """Format as JSON."""
        return json.dumps({
            "id": self.id,
            "event": self.event,
            "data": self.data,
            "timestamp": self.timestamp,
            "is_final": self.is_final,
        })


class StreamSession:
    """An active streaming session."""

    def __init__(self, session_id: str, stream_type: StreamType):
        self.session_id = session_id
        self.stream_type = stream_type
        self.created_at = time.time()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._closed = False
        self._chunks_sent = 0

    async def send(self, chunk: StreamChunk):
        """Send a chunk to the stream."""
        if self._closed:
            return
        await self._queue.put(chunk)
        self._chunks_sent += 1

    async def send_text(self, text: str, event: str = "message"):
        """Convenience: send text as a chunk."""
        await self.send(StreamChunk(data=text, event=event))

    async def send_json(self, data: Dict, event: str = "message"):
        """Convenience: send JSON as a chunk."""
        await self.send(StreamChunk(data=data, event=event))

    async def send_error(self, error: str):
        """Send an error event."""
        await self.send(StreamChunk(data={"error": error}, event="error"))

    async def close(self):
        """Close the stream."""
        self._closed = True
        await self._queue.put(StreamChunk(data="[DONE]", event="done", is_final=True))

    async def __aiter__(self):
        """Async iterator for consuming chunks."""
        while True:
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=1)
                yield chunk
                if chunk.is_final:
                    break
            except asyncio.TimeoutError:
                if self._closed:
                    break
                # Send keepalive
                yield StreamChunk(data="", event="keepalive")


class StreamingEngine:
    """
    Streaming engine with:
    - Multiple concurrent streams
    - Session management
    - Backpressure handling
    - Keepalive pings
    - Stream replay (last N chunks)
    - Fan-out (one source → multiple consumers)
    """

    def __init__(self, max_sessions: int = 100, keepalive_interval: float = 15):
        self.max_sessions = max_sessions
        self.keepalive_interval = keepalive_interval
        self._sessions: Dict[str, StreamSession] = {}
        self._replay_buffer: Dict[str, deque] = {}
        self._fan_out: Dict[str, List[str]] = {}  # source → [session_ids]
        self._keepalive_task: Optional[asyncio.Task] = None

    def create_session(self, stream_type: StreamType = StreamType.SSE,
                       session_id: str = None, replay_size: int = 50) -> StreamSession:
        """Create a new streaming session."""
        sid = session_id or str(uuid.uuid4())[:8]
        session = StreamSession(sid, stream_type)
        self._sessions[sid] = session
        self._replay_buffer[sid] = deque(maxlen=replay_size)
        return session

    def get_session(self, session_id: str) -> Optional[StreamSession]:
        """Get an existing session."""
        return self._sessions.get(session_id)

    async def close_session(self, session_id: str):
        """Close and cleanup a session."""
        session = self._sessions.pop(session_id, None)
        if session:
            await session.close()
        self._replay_buffer.pop(session_id, None)
        # Remove from fan-out
        for source, consumers in self._fan_out.items():
            if session_id in consumers:
                consumers.remove(session_id)

    async def broadcast(self, data: Any, event: str = "message",
                        source: str = None):
        """Broadcast to all sessions or fan-out consumers."""
        targets = self._fan_out.get(source, list(self._sessions.keys()))
        chunk = StreamChunk(data=data, event=event)

        for sid in targets:
            session = self._sessions.get(sid)
            if session and not session._closed:
                await session.send(chunk)
                self._replay_buffer.setdefault(sid, deque(maxlen=50)).append(chunk)

    def fan_out(self, source_session: str, consumer_sessions: List[str]):
        """Set up fan-out: data from source goes to consumers."""
        self._fan_out[source_session] = consumer_sessions

    def get_replay(self, session_id: str) -> List[StreamChunk]:
        """Get recent chunks from replay buffer."""
        return list(self._replay_buffer.get(session_id, []))

    async def start_keepalive(self):
        """Start keepalive ping task."""
        async def _keepalive():
            while True:
                await asyncio.sleep(self.keepalive_interval)
                for sid, session in list(self._sessions.items()):
                    if not session._closed:
                        try:
                            await session.send(StreamChunk(data="", event="keepalive"))
                        except Exception:
                            await self.close_session(sid)

        self._keepalive_task = asyncio.create_task(_keepalive())

    def stop_keepalive(self):
        """Stop keepalive task."""
        if self._keepalive_task:
            self._keepalive_task.cancel()

    def stats(self) -> Dict:
        """Get streaming stats."""
        return {
            "active_sessions": len(self._sessions),
            "max_sessions": self.max_sessions,
            "fan_out_sources": len(self._fan_out),
            "sessions": {
                sid: {
                    "type": s.stream_type.value,
                    "chunks_sent": s._chunks_sent,
                    "age": time.time() - s.created_at,
                    "closed": s._closed,
                }
                for sid, s in self._sessions.items()
            },
        }

    async def shutdown(self):
        """Shutdown all sessions."""
        self.stop_keepalive()
        for sid in list(self._sessions.keys()):
            await self.close_session(sid)
