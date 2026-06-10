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
    llm_client      - Multi-provider LLM client (OpenAI/Anthropic/DeepSeek/Ollama)
    embeddings      - TF-IDF + API embeddings for vector search
    config          - Config manager (YAML, env vars, profiles)
    server          - HTTP server (REST, SSE, WebSocket)
    health          - Health monitoring, metrics, watchdog
    container       - Dependency injection container
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

from .llm_client import (
    LLMClient,
    LLMConfig,
    LLMResponse,
    Provider,
    RateLimiter,
)

from .embeddings import (
    TFIDFEmbedder,
    APIEmbedder,
    HybridEmbedder,
)

from .config import (
    ConfigManager,
)

from .server import (
    AsyncHTTPServer,
    ServerConfig,
    RequestHandler,
)

from .health import (
    HealthMonitor,
    HealthCheck,
    HealthStatus,
    Metrics,
)

from .container import (
    Container,
    create_default_container,
)

__version__ = "0.3.0"
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
    # LLM
    "LLMClient", "LLMConfig", "LLMResponse", "Provider", "RateLimiter",
    # Embeddings
    "TFIDFEmbedder", "APIEmbedder", "HybridEmbedder",
    # Config
    "ConfigManager",
    # Server
    "AsyncHTTPServer", "ServerConfig", "RequestHandler",
    # Health
    "HealthMonitor", "HealthCheck", "HealthStatus", "Metrics",
    # Container
    "Container", "create_default_container",
]
