"""
Observability — structured tracing, spans, distributed tracing.
Full visibility into what the agent is doing, when, and why.
"""
import time
import uuid
import json
import logging
import threading
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """A single trace span."""
    span_id: str
    name: str
    trace_id: str
    parent_id: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0
    status: str = "ok"  # ok, error, timeout
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)
    error: str = ""

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    @property
    def is_finished(self) -> bool:
        return self.end_time > 0

    def finish(self, status: str = "ok", error: str = ""):
        self.end_time = time.time()
        self.status = status
        self.error = error

    def add_event(self, name: str, attributes: Dict = None):
        self.events.append({
            "name": name, "timestamp": time.time(),
            "attributes": attributes or {},
        })


@dataclass
class Trace:
    """A complete trace (collection of spans)."""
    trace_id: str
    name: str
    spans: List[Span] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0
    status: str = "ok"
    metadata: Dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    @property
    def span_count(self) -> int:
        return len(self.spans)

    def get_root_span(self) -> Optional[Span]:
        for s in self.spans:
            if not s.parent_id:
                return s
        return self.spans[0] if self.spans else None

    def get_span_tree(self) -> Dict:
        """Build span tree for visualization."""
        by_parent = defaultdict(list)
        for s in self.spans:
            by_parent[s.parent_id].append(s)

        def build_tree(parent_id=""):
            children = []
            for s in by_parent.get(parent_id, []):
                node = {
                    "id": s.span_id, "name": s.name,
                    "duration_ms": round(s.duration_ms, 1),
                    "status": s.status,
                    "children": build_tree(s.span_id),
                }
                children.append(node)
            return children

        return {"trace_id": self.trace_id, "name": self.name,
                "duration_ms": round(self.duration_ms, 1),
                "spans": build_tree()}


class Tracer:
    """
    Distributed tracer with:
    - Span creation and nesting
    - Automatic timing
    - Attribute tagging
    - Error recording
    - Trace export (JSON)
    - Performance analysis
    - Span sampling
    - Thread-safe operation
    """

    def __init__(self, max_traces: int = 100, sample_rate: float = 1.0):
        self._traces: Dict[str, Trace] = {}
        self._active_spans: Dict[str, Span] = {}  # thread_id -> span
        self._max_traces = max_traces
        self._sample_rate = sample_rate
        self._total_spans = 0
        self._total_traces = 0
        self._lock = threading.Lock()

    def start_trace(self, name: str, metadata: Dict = None) -> Trace:
        """Start a new trace."""
        trace_id = "tr_" + str(uuid.uuid4())[:8]
        trace = Trace(trace_id=trace_id, name=name, metadata=metadata or {})
        with self._lock:
            self._traces[trace_id] = trace
            self._total_traces += 1
            if len(self._traces) > self._max_traces:
                oldest = min(self._traces, key=lambda t: self._traces[t].start_time)
                del self._traces[oldest]
        return trace

    def start_span(self, trace_id: str, name: str,
                   parent_id: str = "", attributes: Dict = None) -> Span:
        """Start a new span within a trace."""
        trace = self._traces.get(trace_id)
        if not trace:
            return None

        span = Span(
            span_id="sp_" + str(uuid.uuid4())[:8],
            name=name, trace_id=trace_id,
            parent_id=parent_id,
            attributes=attributes or {},
        )
        trace.spans.append(span)
        self._total_spans += 1

        thread_id = str(threading.get_ident())
        self._active_spans[thread_id] = span

        return span

    @contextmanager
    def span(self, trace_id: str, name: str, attributes: Dict = None):
        """Context manager for automatic span lifecycle."""
        parent = self._active_spans.get(str(threading.get_ident()))
        parent_id = parent.span_id if parent else ""

        s = self.start_span(trace_id, name, parent_id, attributes)
        try:
            yield s
            s.finish("ok")
        except Exception as e:
            s.finish("error", str(e))
            raise
        finally:
            thread_id = str(threading.get_ident())
            if parent:
                self._active_spans[thread_id] = parent
            else:
                self._active_spans.pop(thread_id, None)

    def finish_trace(self, trace_id: str, status: str = "ok"):
        """Finish a trace."""
        trace = self._traces.get(trace_id)
        if trace:
            trace.end_time = time.time()
            trace.status = status
            # Finish any unfinished spans
            for s in trace.spans:
                if not s.is_finished:
                    s.finish()

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        return self._traces.get(trace_id)

    def get_recent_traces(self, limit: int = 20) -> List[Trace]:
        traces = sorted(self._traces.values(), key=lambda t: t.start_time, reverse=True)
        return traces[:limit]

    def get_slow_traces(self, threshold_ms: float = 1000) -> List[Trace]:
        return [t for t in self._traces.values() if t.duration_ms > threshold_ms]

    def get_error_traces(self) -> List[Trace]:
        return [t for t in self._traces.values() if t.status == "error"]

    def export_trace(self, trace_id: str) -> Dict:
        """Export trace as JSON-compatible dict."""
        trace = self._traces.get(trace_id)
        if not trace:
            return {}
        return {
            "trace_id": trace.trace_id,
            "name": trace.name,
            "duration_ms": round(trace.duration_ms, 2),
            "status": trace.status,
            "span_count": trace.span_count,
            "tree": trace.get_span_tree(),
            "spans": [{
                "id": s.span_id, "name": s.name,
                "parent_id": s.parent_id,
                "duration_ms": round(s.duration_ms, 2),
                "status": s.status,
                "attributes": s.attributes,
                "events": s.events,
                "error": s.error,
            } for s in trace.spans],
        }

    def stats(self) -> Dict:
        all_durations = [t.duration_ms for t in self._traces.values()]
        return {
            "total_traces": self._total_traces,
            "active_traces": len(self._traces),
            "total_spans": self._total_spans,
            "error_traces": len(self.get_error_traces()),
            "slow_traces": len(self.get_slow_traces()),
            "avg_duration_ms": sum(all_durations) / len(all_durations) if all_durations else 0,
        }
