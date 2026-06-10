"""
Hermes Agent Async Core — full async framework.

Modules:
    async_loop      - True async conversation loop with concurrent tools
    orchestrator    - Multi-agent orchestrator (spawn, delegate, coordinate)
    memory          - Persistent memory with vector search
    plugins         - Hot-loadable plugin engine
    events          - Event bus (pub/sub with middleware)
    streaming       - SSE/WebSocket streaming engine
    budget          - Token usage and cost tracking
    sessions        - Multi-session manager with branching
    cli_enhanced    - Rich terminal CLI with autocomplete
"""

from .async_loop import (
    AsyncConversationLoop,
    AsyncToolRunner,
    AgentConfig,
    AgentState,
    ToolCall,
    ConversationTurn,
)

from .orchestrator import (
    AsyncOrchestrator,
    AgentSpec,
    AgentRole,
    AgentResult,
    AgentProcess,
)

from .memory import (
    MemoryStore,
    MemoryEntry,
    VectorIndex,
)

from .plugins import (
    PluginEngine,
    PluginBase,
    PluginInfo,
)

from .events import (
    EventBus,
    Event,
    EventPriority,
    EventHandler,
)

from .streaming import (
    StreamingEngine,
    StreamSession,
    StreamChunk,
    StreamType,
)

from .budget import (
    BudgetTracker,
    BudgetLimit,
    BudgetScope,
    UsageRecord,
    MODEL_PRICING,
)

from .sessions import (
    SessionManager,
    Session,
    Checkpoint,
)

from .cli_enhanced import (
    EnhancedCLI,
    CLIConfig,
    CommandRegistry,
)

__version__ = "0.2.0"
__all__ = [
    # Core
    "AsyncConversationLoop", "AsyncToolRunner", "AgentConfig", "AgentState",
    "ToolCall", "ConversationTurn",
    # Orchestrator
    "AsyncOrchestrator", "AgentSpec", "AgentRole", "AgentResult", "AgentProcess",
    # Memory
    "MemoryStore", "MemoryEntry", "VectorIndex",
    # Plugins
    "PluginEngine", "PluginBase", "PluginInfo",
    # Events
    "EventBus", "Event", "EventPriority", "EventHandler",
    # Streaming
    "StreamingEngine", "StreamSession", "StreamChunk", "StreamType",
    # Budget
    "BudgetTracker", "BudgetLimit", "BudgetScope", "UsageRecord", "MODEL_PRICING",
    # Sessions
    "SessionManager", "Session", "Checkpoint",
    # CLI
    "EnhancedCLI", "CLIConfig", "CommandRegistry",
]
