"""
Event Bus — pub/sub system with middleware pipeline.
Supports: sync/async handlers, middleware chains, event replay, history.
"""
import asyncio
import time
import logging
import weakref
from typing import Optional, Dict, Any, List, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """A bus event."""
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    propagation_stopped: bool = False

    def stop(self):
        """Stop event propagation to remaining handlers."""
        self.propagation_stopped = True


@dataclass
class EventHandler:
    """Registered event handler."""
    callback: Callable
    priority: EventPriority = EventPriority.NORMAL
    once: bool = False
    filter_fn: Optional[Callable] = None


class EventBus:
    """
    Event bus with:
    - Wildcard subscriptions (e.g., "tool.*", "*")
    - Priority-ordered handlers
    - Middleware pipeline (can transform events)
    - Event history with configurable buffer
    - Async and sync handler support
    - One-shot handlers
    - Event replay
    - Weak reference support
    """

    def __init__(self, history_size: int = 1000):
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._wildcard_handlers: List[EventHandler] = []
        self._middleware: List[Callable] = []
        self._history: deque = deque(maxlen=history_size)
        self._event_count = 0
        self._suppressed: Set[str] = set()

    def on(self, event: str, callback: Callable, priority: EventPriority = EventPriority.NORMAL,
           once: bool = False, filter_fn: Callable = None):
        """Subscribe to an event."""
        handler = EventHandler(
            callback=callback, priority=priority, once=once, filter_fn=filter_fn
        )
        if event == "*":
            self._wildcard_handlers.append(handler)
        else:
            self._handlers[event].append(handler)
            self._handlers[event].sort(key=lambda h: h.priority.value, reverse=True)

    def off(self, event: str, callback: Callable = None):
        """Unsubscribe from an event."""
        if callback:
            self._handlers[event] = [
                h for h in self._handlers[event] if h.callback != callback
            ]
        else:
            self._handlers.pop(event, None)

    def once(self, event: str, callback: Callable, priority: EventPriority = EventPriority.NORMAL):
        """Subscribe to an event, auto-remove after first trigger."""
        self.on(event, callback, priority, once=True)

    def use(self, middleware: Callable):
        """Add middleware to the pipeline. Middleware receives (event, next_fn)."""
        self._middleware.append(middleware)

    def suppress(self, event_pattern: str):
        """Suppress events matching pattern."""
        self._suppressed.add(event_pattern)

    def unsuppress(self, event_pattern: str):
        """Stop suppressing events."""
        self._suppressed.discard(event_pattern)

    async def emit(self, event_name: str, data: Dict[str, Any] = None,
                   source: str = "", priority: EventPriority = EventPriority.NORMAL) -> Event:
        """Emit an event through the bus."""
        # Check suppression
        for pattern in self._suppressed:
            if self._matches(event_name, pattern):
                return Event(name=event_name, data=data or {})

        event = Event(name=event_name, data=data or {}, source=source, priority=priority)

        # Run middleware pipeline
        for mw in self._middleware:
            try:
                if asyncio.iscoroutinefunction(mw):
                    event = await mw(event, lambda e: e)
                else:
                    event = mw(event, lambda e: e)
                if event.propagation_stopped:
                    return event
            except Exception as e:
                logger.warning(f"Middleware error for {event_name}: {e}")

        # Record in history
        self._history.append(event)
        self._event_count += 1

        # Collect matching handlers
        handlers = list(self._handlers.get(event_name, []))

        # Pattern matching (e.g., "tool.start" matches subscription "tool.*")
        for pattern, pattern_handlers in self._handlers.items():
            if pattern != event_name and self._matches(event_name, pattern):
                handlers.extend(pattern_handlers)

        # Wildcard handlers
        handlers.extend(self._wildcard_handlers)

        # Sort by priority
        handlers.sort(key=lambda h: h.priority.value, reverse=True)

        # Execute handlers
        to_remove = []
        for handler in handlers:
            if event.propagation_stopped:
                break

            # Apply filter
            if handler.filter_fn and not handler.filter_fn(event):
                continue

            try:
                if asyncio.iscoroutinefunction(handler.callback):
                    await handler.callback(event)
                else:
                    handler.callback(event)
            except Exception as e:
                logger.warning(f"Handler error for {event_name}: {e}")

            if handler.once:
                to_remove.append(handler)

        # Remove one-shot handlers
        for handler in to_remove:
            if handler in self._wildcard_handlers:
                self._wildcard_handlers.remove(handler)
            else:
                for h_list in self._handlers.values():
                    if handler in h_list:
                        h_list.remove(handler)

        return event

    def _matches(self, event_name: str, pattern: str) -> bool:
        """Check if event name matches a wildcard pattern."""
        if pattern == "*":
            return True
        if "*" in pattern:
            prefix = pattern.replace("*", "")
            return event_name.startswith(prefix)
        return event_name == pattern

    def replay(self, event_name: str = None, since: float = None) -> List[Event]:
        """Replay historical events."""
        events = list(self._history)
        if event_name:
            events = [e for e in events if self._matches(e.name, event_name)]
        if since:
            events = [e for e in events if e.timestamp >= since]
        return events

    def stats(self) -> Dict:
        """Get event bus statistics."""
        return {
            "total_events": self._event_count,
            "history_size": len(self._history),
            "subscriptions": {k: len(v) for k, v in self._handlers.items()},
            "wildcard_handlers": len(self._wildcard_handlers),
            "middleware_count": len(self._middleware),
            "suppressed": list(self._suppressed),
        }
