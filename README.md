# Hermes Agent — AGI-Grade Autonomous Agent Framework

> 37 production modules · 10,805 LOC · Complete cognitive architecture

## What Makes This Different

This is the **only agent framework** with integrated:
- 🧠 **Metacognition** — agent thinks about its own thinking, detects biases
- 👁️ **Theory of Mind** — models user intent, emotions, knowledge gaps
- 🔗 **Chain-of-Thought** — structured reasoning with backtracking
- 🌍 **World Model** — simulates outcomes before acting
- ❓ **Curiosity Engine** — intrinsic motivation to explore and learn
- 📊 **Knowledge Graph** — structured entity-relationship memory
- 🔄 **Self-Improvement** — learns from its own mistakes

## Architecture

```
COGNITION (AGI-grade)
├── metacognition.py     Self-aware reasoning, bias detection
├── theory_of_mind.py    User intent modeling, emotion detection
├── chain_of_thought.py  Structured reasoning, backtracking
├── world_model.py       Simulation, prediction, counterfactuals
└── curiosity.py         Intrinsic motivation, hypothesis testing

INTELLIGENCE
├── knowledge_graph.py   Entity-relationship memory
├── self_improve.py      Learn from mistakes, adaptive strategies
├── planner.py           Goal decomposition, autonomous planning
├── code_intel.py        AST analysis, dependency graphs
└── rag.py               Retrieval-Augmented Generation pipeline

INFRASTRUCTURE
├── async_loop.py        Async conversation loop
├── llm_client.py        5 LLM providers (OpenAI/Anthropic/DeepSeek/Ollama/OpenRouter)
├── orchestrator.py      Multi-agent coordination
├── tools.py             Dynamic tool registry + middleware
├── sandbox.py           Safe Python/Bash execution
├── workflows.py         DAG workflow engine
├── semantic_cache.py    Embedding-based response caching
├── context.py           Smart context compression
├── prompt_optimizer.py  A/B test and optimize prompts
├── tracing.py           Distributed tracing
├── web_intel.py         Web scraping + API discovery
├── agent_protocol.py    Agent-to-agent messaging + consensus
├── test_gen.py          Auto test generation + fuzzing
├── guardrails.py        Safety pipeline + prompt injection detection
├── recovery.py          Auto-recovery + conversation checkpoints
├── health.py            Health monitoring + watchdog
├── budget.py            Token/cost tracking + rate limiting
├── memory.py            Persistent vector memory + SQLite
├── embeddings.py        TF-IDF + API embeddings
├── sessions.py          Multi-session + branching + checkpointing
├── events.py            Event bus (pub/sub + middleware)
├── streaming.py         SSE/WebSocket streaming
├── server.py            FastAPI HTTP server
├── plugins.py           Hot-loadable plugin engine
├── config.py            YAML config + env vars + profiles
├── container.py         Dependency injection
└── cli_enhanced.py      Rich terminal CLI
```

## Quick Start

```python
from agent.async_core import (
    AsyncConversationLoop, AgentConfig,
    KnowledgeGraph, MetacognitionEngine,
    TheoryOfMind, CuriosityEngine,
)

# Create agent with cognitive architecture
config = AgentConfig(model="gpt-4o-mini", streaming=True)
agent = AsyncConversationLoop(config)

# Add knowledge graph
kg = KnowledgeGraph()
kg.add_entity("Python", "language")
kg.add_entity("FastAPI", "framework")
kg.add_relationship(...)

# Agent self-assessment
meta = MetacognitionEngine()
assessment = meta.self_assess()
```

## Modules by Category

| Category | Modules | LOC | Description |
|----------|---------|-----|-------------|
| Cognition | 5 | 1,637 | Metacognition, Theory of Mind, CoT, World Model, Curiosity |
| Intelligence | 5 | 1,458 | Knowledge Graph, Self-Improvement, Planner, Code Intel, RAG |
| Core | 5 | 1,486 | Async Loop, LLM Client, Orchestrator, Tools, Sandbox |
| Infrastructure | 10 | 2,541 | Workflows, Cache, Context, Tracing, Web, Protocol, Test Gen |
| Safety | 3 | 775 | Guardrails, Recovery, Health |
| Platform | 9 | 2,908 | Memory, Sessions, Events, Streaming, Server, Plugins, Config, Container, CLI |

## License

MIT
