"""
Hermes Agent Async Core v0.4.0 — complete agent framework.

22 modules covering every aspect of autonomous agent operation:
    Core:       async_loop, orchestrator, llm_client
    Memory:     memory, embeddings, sessions, context
    Tools:      tools, sandbox, rag
    Infra:      config, events, streaming, budget, server
    Safety:     guardrails, recovery, health
    Extend:     plugins, workflows, container, cli_enhanced
"""

# === Core ===
from .async_loop import (
    AsyncConversationLoop, AsyncToolRunner, AgentConfig, AgentState,
    ToolCall, ConversationTurn,
)
from .orchestrator import (
    AsyncOrchestrator, AgentSpec, AgentRole, AgentResult, AgentProcess,
)
from .llm_client import (
    LLMClient, LLMConfig, LLMResponse, Provider, RateLimiter,
)

# === Memory ===
from .memory import MemoryStore, MemoryEntry, VectorIndex
from .embeddings import TFIDFEmbedder, APIEmbedder, HybridEmbedder
from .sessions import SessionManager, Session, Checkpoint
from .context import ContextCompressor, CompressionStrategy, Message

# === Tools ===
from .tools import (
    ToolRegistry, ToolSchema, ToolResult, ToolCategory,
    ToolMiddleware, LoggingMiddleware, MetricsMiddleware,
)
from .sandbox import CodeSandbox, SandboxConfig, ExecutionResult
from .rag import DocumentLoader, TextChunker, RAGRetriever, Document, Chunk

# === Infrastructure ===
from .config import ConfigManager
from .events import EventBus, Event, EventPriority, EventHandler
from .streaming import StreamingEngine, StreamSession, StreamChunk, StreamType
from .budget import BudgetTracker, BudgetLimit, BudgetScope, UsageRecord, MODEL_PRICING
from .server import AsyncHTTPServer, ServerConfig, RequestHandler

# === Safety ===
from .guardrails import (
    GuardrailsPipeline, GuardResult, RiskLevel,
    PromptInjectionGuard, ContentFilterGuard, OutputValidatorGuard, RateLimitGuard,
)
from .recovery import AutoRecovery, ConversationState, ConversationCheckpoint, StateMachine
from .health import HealthMonitor, HealthCheck, HealthStatus, Metrics

# === Extend ===
from .plugins import PluginEngine, PluginBase, PluginInfo
from .workflows import WorkflowEngine, WorkflowStep, WorkflowDef, WorkflowRun, StepStatus
from .container import Container, create_default_container
from .cli_enhanced import EnhancedCLI, CLIConfig, CommandRegistry

# === Super Upgrade ===
from .knowledge_graph import KnowledgeGraph, Entity, Relationship, GraphQuery
from .self_improve import SelfImprovementEngine, ActionRecord, Strategy, Lesson
from .planner import AutonomousPlanner, Plan, Task, GoalNode, PlanStatus, TaskStatus
from .semantic_cache import SemanticCache, CacheEntry
from .code_intel import CodeAnalyzer, FileAnalysis, FunctionInfo, ClassInfo, CodeIssue

# === Super Cognition (AGI-grade) ===
from .metacognition import MetacognitionEngine, ReasoningTrace, CognitiveState, ConfidenceCalibrator, BiasDetector
from .theory_of_mind import TheoryOfMind, UserBelief, UserGoal, UserEmotionalState, CommunicationStyle
from .chain_of_thought import ChainOfThoughtEngine, Thought, ThoughtType, ReasoningChain
from .world_model import WorldModel, WorldState, Action, Prediction, Simulation
from .curiosity import CuriosityEngine, Hypothesis, CuriosityType, ExplorationGoal

# === Production-Grade Infrastructure ===
from .tracing import Tracer, Trace, Span
from .prompt_optimizer import PromptOptimizer, PromptVariant
from .web_intel import WebExtractor, WebPage, ExtractedData, APIEndpoint
from .agent_protocol import AgentProtocol, AgentProfile, AgentMessage, MessageType, Proposal
from .test_gen import TestGenerator, TestCase, CoverageInfo

__version__ = "0.7.0"
__all__ = [
    # Core
    "AsyncConversationLoop", "AsyncToolRunner", "AgentConfig", "AgentState",
    "ToolCall", "ConversationTurn",
    "AsyncOrchestrator", "AgentSpec", "AgentRole", "AgentResult", "AgentProcess",
    "LLMClient", "LLMConfig", "LLMResponse", "Provider", "RateLimiter",
    # Memory
    "MemoryStore", "MemoryEntry", "VectorIndex",
    "TFIDFEmbedder", "APIEmbedder", "HybridEmbedder",
    "SessionManager", "Session", "Checkpoint",
    "ContextCompressor", "CompressionStrategy", "Message",
    # Tools
    "ToolRegistry", "ToolSchema", "ToolResult", "ToolCategory",
    "ToolMiddleware", "LoggingMiddleware", "MetricsMiddleware",
    "CodeSandbox", "SandboxConfig", "ExecutionResult",
    "DocumentLoader", "TextChunker", "RAGRetriever", "Document", "Chunk",
    # Infrastructure
    "ConfigManager",
    "EventBus", "Event", "EventPriority", "EventHandler",
    "StreamingEngine", "StreamSession", "StreamChunk", "StreamType",
    "BudgetTracker", "BudgetLimit", "BudgetScope", "UsageRecord", "MODEL_PRICING",
    "AsyncHTTPServer", "ServerConfig", "RequestHandler",
    # Safety
    "GuardrailsPipeline", "GuardResult", "RiskLevel",
    "PromptInjectionGuard", "ContentFilterGuard", "OutputValidatorGuard", "RateLimitGuard",
    "AutoRecovery", "ConversationState", "ConversationCheckpoint", "StateMachine",
    "HealthMonitor", "HealthCheck", "HealthStatus", "Metrics",
    # Extend
    "PluginEngine", "PluginBase", "PluginInfo",
    "WorkflowEngine", "WorkflowStep", "WorkflowDef", "WorkflowRun", "StepStatus",
    "Container", "create_default_container",
    "EnhancedCLI", "CLIConfig", "CommandRegistry",
    # Super Upgrade
    "KnowledgeGraph", "Entity", "Relationship", "GraphQuery",
    "SelfImprovementEngine", "ActionRecord", "Strategy", "Lesson",
    "AutonomousPlanner", "Plan", "Task", "GoalNode", "PlanStatus", "TaskStatus",
    "SemanticCache", "CacheEntry",
    "CodeAnalyzer", "FileAnalysis", "FunctionInfo", "ClassInfo", "CodeIssue",
    # Super Cognition
    "MetacognitionEngine", "ReasoningTrace", "CognitiveState", "ConfidenceCalibrator", "BiasDetector",
    "TheoryOfMind", "UserBelief", "UserGoal", "UserEmotionalState", "CommunicationStyle",
    "ChainOfThoughtEngine", "Thought", "ThoughtType", "ReasoningChain",
    "WorldModel", "WorldState", "Action", "Prediction", "Simulation",
    "CuriosityEngine", "Hypothesis", "CuriosityType", "ExplorationGoal",
    # Production Infrastructure
    "Tracer", "Trace", "Span",
    "PromptOptimizer", "PromptVariant",
    "WebExtractor", "WebPage", "ExtractedData", "APIEndpoint",
    "AgentProtocol", "AgentProfile", "AgentMessage", "MessageType", "Proposal",
    "TestGenerator", "TestCase", "CoverageInfo",
]
