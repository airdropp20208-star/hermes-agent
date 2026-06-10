"""
True Async Conversation Loop — not a wrapper, real asyncio.
Handles concurrent tool execution, streaming, and interrupt support.
"""
import asyncio
import time
import uuid
import logging
from typing import Optional, Dict, Any, List, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    TOOL_CALLING = "tool_calling"
    STREAMING = "streaming"
    WAITING = "waiting"
    ERROR = "error"
    INTERRUPTED = "interrupted"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0
    status: str = "pending"  # pending | running | done | error


@dataclass
class ConversationTurn:
    turn_id: str
    role: str  # user | assistant | system | tool
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    model: str = ""
    provider: str = ""
    api_key: str = ""
    base_url: str = ""
    max_iterations: int = 90
    max_tokens: int = 4096
    temperature: float = 0.7
    system_prompt: str = ""
    enabled_toolsets: List[str] = field(default_factory=list)
    streaming: bool = True
    parallel_tools: bool = True
    max_parallel_tools: int = 5
    timeout_per_turn: float = 300
    interrupt_on_error: bool = False


class AsyncToolRunner:
    """Execute tools concurrently with semaphore control."""

    def __init__(self, tool_registry: Dict[str, Callable], max_parallel: int = 5):
        self.registry = tool_registry
        self.semaphore = asyncio.Semaphore(max_parallel)
        self._running: Dict[str, asyncio.Task] = {}

    async def execute(self, call: ToolCall) -> ToolCall:
        """Execute a single tool call with timeout and error handling."""
        async with self.semaphore:
            call.status = "running"
            start = time.monotonic()
            try:
                func = self.registry.get(call.name)
                if not func:
                    call.error = f"Unknown tool: {call.name}"
                    call.status = "error"
                    return call

                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(func(**call.arguments), timeout=120)
                else:
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: func(**call.arguments)),
                        timeout=120
                    )
                call.result = str(result)
                call.status = "done"
            except asyncio.TimeoutError:
                call.error = f"Tool {call.name} timed out after 120s"
                call.status = "error"
            except Exception as e:
                call.error = f"{type(e).__name__}: {e}"
                call.status = "error"
            finally:
                call.duration_ms = (time.monotonic() - start) * 1000
            return call

    async def execute_batch(self, calls: List[ToolCall]) -> List[ToolCall]:
        """Execute multiple tool calls concurrently."""
        tasks = [asyncio.create_task(self.execute(c)) for c in calls]
        self._running.update({c.id: t for c, t in zip(calls, tasks)})
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for call, result in zip(calls, results):
            if isinstance(result, Exception):
                call.error = str(result)
                call.status = "error"
        return calls


class AsyncConversationLoop:
    """
    True async conversation loop with:
    - Concurrent tool execution
    - Streaming responses
    - Interrupt support
    - State management
    - Event hooks
    - Budget tracking
    """

    def __init__(self, config: AgentConfig, tool_registry: Dict[str, Callable] = None):
        self.config = config
        self.state = AgentState.IDLE
        self.session_id = str(uuid.uuid4())
        self.turns: List[ConversationTurn] = []
        self.tool_runner = AsyncToolRunner(tool_registry or {}, config.max_parallel_tools)

        self._interrupt = asyncio.Event()
        self._interrupt.set()  # Not interrupted
        self._hooks: Dict[str, List[Callable]] = {
            "on_turn_start": [],
            "on_turn_end": [],
            "on_tool_start": [],
            "on_tool_end": [],
            "on_state_change": [],
            "on_error": [],
            "on_stream_token": [],
        }
        self._total_tokens = 0
        self._total_cost = 0.0
        self._api_calls = 0

    def on(self, event: str, callback: Callable):
        """Register event hook."""
        if event in self._hooks:
            self._hooks[event].append(callback)

    async def _emit(self, event: str, **kwargs):
        """Emit event to all registered hooks."""
        for hook in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(**kwargs)
                else:
                    hook(**kwargs)
            except Exception as e:
                logger.warning(f"Hook {event} error: {e}")

    async def _set_state(self, new_state: AgentState):
        """Transition state with event emission."""
        old = self.state
        self.state = new_state
        await self._emit("on_state_change", old=old, new=new_state)

    async def _call_llm(self, messages: List[Dict], stream: bool = False):
        """Call LLM provider. Override in subclass for actual API calls."""
        raise NotImplementedError("Subclass must implement _call_llm")

    async def _process_tool_calls(self, tool_calls_data: List[Dict]) -> List[ToolCall]:
        """Process tool calls — parallel if enabled."""
        calls = []
        for tc in tool_calls_data:
            call = ToolCall(
                id=tc.get("id", str(uuid.uuid4())),
                name=tc["name"],
                arguments=tc.get("arguments", {})
            )
            calls.append(call)

        if self.config.parallel_tools and len(calls) > 1:
            return await self.tool_runner.execute_batch(calls)
        else:
            return [await self.tool_runner.execute(c) for c in calls]

    async def chat(self, user_message: str) -> AsyncIterator[str]:
        """
        Main chat method. Yields tokens as they stream.
        Final response is the last yielded value.
        """
        # Add user turn
        user_turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            role="user",
            content=user_message
        )
        self.turns.append(user_turn)
        await self._emit("on_turn_start", turn=user_turn)

        iteration = 0
        full_response = ""

        while iteration < self.config.max_iterations:
            if not self._interrupt.is_set():
                await self._set_state(AgentState.INTERRUPTED)
                yield "[INTERRUPTED]"
                break

            await self._set_state(AgentState.THINKING)

            # Build messages for API
            messages = self._build_messages()

            # Call LLM
            try:
                response = await self._call_llm(messages, stream=self.config.streaming)
            except Exception as e:
                await self._set_state(AgentState.ERROR)
                await self._emit("on_error", error=e)
                yield f"[ERROR] {e}"
                break

            self._api_calls += 1

            # Check if response has tool calls
            if response.get("tool_calls"):
                await self._set_state(AgentState.TOOL_CALLING)
                tool_calls = await self._process_tool_calls(response["tool_calls"])

                # Add assistant turn with tool calls
                assistant_turn = ConversationTurn(
                    turn_id=str(uuid.uuid4()),
                    role="assistant",
                    content=response.get("content", ""),
                    tool_calls=tool_calls
                )
                self.turns.append(assistant_turn)

                # Add tool results as messages
                for tc in tool_calls:
                    tool_turn = ConversationTurn(
                        turn_id=str(uuid.uuid4()),
                        role="tool",
                        content=tc.result or tc.error or "",
                        metadata={"tool_call_id": tc.id, "tool_name": tc.name}
                    )
                    self.turns.append(tool_turn)
                    await self._emit("on_tool_end", tool_call=tc)

                iteration += 1
                continue

            # No tool calls — final response
            content = response.get("content", "")
            full_response = content

            assistant_turn = ConversationTurn(
                turn_id=str(uuid.uuid4()),
                role="assistant",
                content=content,
                token_count=response.get("token_count", 0)
            )
            self.turns.append(assistant_turn)
            self._total_tokens += response.get("token_count", 0)

            # Stream tokens
            if self.config.streaming and isinstance(content, str):
                for token in content.split():
                    await self._emit("on_stream_token", token=token + " ")
                    yield token + " "

            break

        await self._set_state(AgentState.IDLE)
        await self._emit("on_turn_end", turn=self.turns[-1] if self.turns else None)

        if not full_response:
            yield "[NO RESPONSE]"

    def interrupt(self):
        """Interrupt the current conversation loop."""
        self._interrupt.clear()

    def resume(self):
        """Resume after interrupt."""
        self._interrupt.set()

    def _build_messages(self) -> List[Dict]:
        """Build message list for API call."""
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})

        for turn in self.turns[-20:]:  # Last 20 turns for context
            msg = {"role": turn.role, "content": turn.content}
            if turn.tool_calls:
                msg["tool_calls"] = [{
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments}
                } for tc in turn.tool_calls]
            if turn.metadata.get("tool_call_id"):
                msg["tool_call_id"] = turn.metadata["tool_call_id"]
            messages.append(msg)
        return messages

    def get_stats(self) -> Dict:
        """Get conversation statistics."""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "total_turns": len(self.turns),
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
            "api_calls": self._api_calls,
            "tool_calls": sum(len(t.tool_calls) for t in self.turns),
            "uptime": time.time() - (self.turns[0].timestamp if self.turns else time.time()),
        }

    def export_history(self) -> List[Dict]:
        """Export conversation history as serializable dicts."""
        return [{
            "turn_id": t.turn_id,
            "role": t.role,
            "content": t.content,
            "tool_calls": [{
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
                "result": tc.result,
                "error": tc.error,
                "duration_ms": tc.duration_ms,
                "status": tc.status,
            } for tc in t.tool_calls],
            "timestamp": t.timestamp,
            "token_count": t.token_count,
        } for t in self.turns]
