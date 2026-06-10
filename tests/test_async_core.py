"""
Integration tests for the async core framework.
Tests all 9 modules independently and together.
"""
import asyncio
import sys
import os
import time
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.async_core import (
    AsyncConversationLoop, AsyncToolRunner, AgentConfig, AgentState,
    ToolCall, ConversationTurn,
    AsyncOrchestrator, AgentSpec, AgentRole, AgentResult,
    MemoryStore, MemoryEntry,
    PluginEngine, PluginBase,
    EventBus, Event, EventPriority,
    StreamingEngine, StreamChunk, StreamType,
    BudgetTracker, BudgetLimit, BudgetScope,
    SessionManager, Session,
    EnhancedCLI, CLIConfig, CommandRegistry,
)

passed = 0
failed = 0
errors = []

def test(name, fn):
    global passed, failed
    try:
        result = fn()
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        passed += 1
        print(f"  PASS {name}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  FAIL {name}: {e}")

# ========== Module 1: Async Loop ==========
print("\n--- Module 1: Async Conversation Loop ---")

def test_tool_call():
    call = ToolCall(id="t1", name="echo", arguments={"text": "hello"})
    assert call.status == "pending"
    assert call.name == "echo"

def test_agent_config():
    config = AgentConfig(model="gpt-4o", max_iterations=50, streaming=True)
    assert config.model == "gpt-4o"
    assert config.parallel_tools is True

def test_agent_state():
    assert AgentState.IDLE.value == "idle"
    assert AgentState.THINKING.value == "thinking"

async def test_tool_runner():
    async def echo_tool(text):
        return "echo: " + text
    runner = AsyncToolRunner({"echo": echo_tool})
    call = ToolCall(id="t1", name="echo", arguments={"text": "hello"})
    result = await runner.execute(call)
    assert result.status == "done"
    assert result.result == "echo: hello"

async def test_tool_runner_batch():
    async def add(a, b):
        return a + b
    runner = AsyncToolRunner({"add": add}, max_parallel=3)
    calls = [ToolCall(id="t%d" % i, name="add", arguments={"a": i, "b": i}) for i in range(5)]
    results = await runner.execute_batch(calls)
    assert len(results) == 5
    assert all(r.status == "done" for r in results)

async def test_tool_runner_error():
    async def fail_tool():
        raise ValueError("intentional error")
    runner = AsyncToolRunner({"fail": fail_tool})
    call = ToolCall(id="t1", name="fail", arguments={})
    result = await runner.execute(call)
    assert result.status == "error"
    assert "intentional error" in result.error

async def test_tool_runner_unknown():
    runner = AsyncToolRunner({})
    call = ToolCall(id="t1", name="nonexistent", arguments={})
    result = await runner.execute(call)
    assert result.status == "error"
    assert "Unknown tool" in result.error

def test_conversation_turn():
    turn = ConversationTurn(turn_id="t1", role="user", content="hello")
    assert turn.role == "user"
    assert len(turn.tool_calls) == 0
    assert turn.timestamp > 0

test("ToolCall creation", test_tool_call)
test("AgentConfig defaults", test_agent_config)
test("AgentState enum", test_agent_state)
test("ToolRunner execute", test_tool_runner)
test("ToolRunner batch execute", test_tool_runner_batch)
test("ToolRunner error handling", test_tool_runner_error)
test("ToolRunner unknown tool", test_tool_runner_unknown)
test("ConversationTurn creation", test_conversation_turn)

# ========== Module 2: Orchestrator ==========
print("\n--- Module 2: Multi-Agent Orchestrator ---")

def test_agent_spec():
    spec = AgentSpec(role=AgentRole.WORKER, goal="test task", context="some context")
    assert spec.role == AgentRole.WORKER
    assert spec.goal == "test task"

async def test_orchestrator_spawn():
    orch = AsyncOrchestrator(max_concurrent=5)
    spec = AgentSpec(role=AgentRole.WORKER, goal="test")
    result = await orch.spawn(spec)
    assert result.success
    assert result.agent_id

async def test_orchestrator_parallel():
    orch = AsyncOrchestrator(max_concurrent=5)
    specs = [AgentSpec(role=AgentRole.WORKER, goal="task %d" % i) for i in range(5)]
    results = await orch.spawn_parallel(specs)
    assert len(results) == 5
    assert all(r.success for r in results)

async def test_orchestrator_status():
    orch = AsyncOrchestrator()
    spec = AgentSpec(role=AgentRole.WORKER, goal="test")
    await orch.spawn(spec)
    status = orch.get_status()
    assert status["total_agents"] >= 1

test("AgentSpec creation", test_agent_spec)
test("Orchestrator spawn", test_orchestrator_spawn)
test("Orchestrator parallel", test_orchestrator_parallel)
test("Orchestrator status", test_orchestrator_status)

# ========== Module 3: Memory ==========
print("\n--- Module 3: Memory System ---")

def test_memory_add():
    store = MemoryStore()
    mid = store.add("test memory", category="fact", importance=0.8)
    assert mid
    entry = store.get(mid)
    assert entry.content == "test memory"
    assert entry.category == "fact"

def test_memory_search():
    store = MemoryStore()
    store.add("Python is a programming language", category="fact")
    store.add("User prefers dark mode", category="preference")
    store.add("The sky is blue", category="fact")
    results = store.search(query="Python", category="fact")
    assert len(results) >= 1
    assert "Python" in results[0].content

def test_memory_update():
    store = MemoryStore()
    mid = store.add("original content")
    store.update(mid, content="updated content", importance=0.9)
    entry = store.get(mid)
    assert entry.content == "updated content"
    assert entry.importance == 0.9

def test_memory_delete():
    store = MemoryStore()
    mid = store.add("to be deleted")
    assert store.delete(mid) is True
    assert store.get(mid) is None

def test_memory_ttl():
    store = MemoryStore()
    mid = store.add("expires soon", ttl=0.1)
    assert store.get(mid) is not None
    time.sleep(0.2)
    assert store.get(mid) is None

def test_memory_context():
    store = MemoryStore()
    store.add("Important fact 1", importance=0.9)
    store.add("Important fact 2", importance=0.8)
    store.add("Less important", importance=0.2)
    context = store.get_context(max_tokens=100)
    assert "Important fact" in context

def test_memory_stats():
    store = MemoryStore()
    store.add("fact 1", category="fact")
    store.add("pref 1", category="preference")
    stats = store.stats()
    assert stats["total"] == 2
    assert stats["by_category"]["fact"] == 1

def test_memory_persistence():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store1 = MemoryStore(db_path=db_path)
        store1.add("persistent memory", category="fact")
        store2 = MemoryStore(db_path=db_path)
        results = store2.search(query="persistent")
        assert len(results) >= 1
    finally:
        os.unlink(db_path)

def test_vector_index():
    from agent.async_core.memory import VectorIndex
    idx = VectorIndex(dimension=3)
    idx.add("a", [1.0, 0.0, 0.0])
    idx.add("b", [0.0, 1.0, 0.0])
    idx.add("c", [0.7, 0.7, 0.0])
    results = idx.search([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0][0] == "a"

test("Memory add/get", test_memory_add)
test("Memory search", test_memory_search)
test("Memory update", test_memory_update)
test("Memory delete", test_memory_delete)
test("Memory TTL expiry", test_memory_ttl)
test("Memory context generation", test_memory_context)
test("Memory stats", test_memory_stats)
test("Memory SQLite persistence", test_memory_persistence)
test("Vector index similarity", test_vector_index)

# ========== Module 4: Plugin Engine ==========
print("\n--- Module 4: Plugin Engine ---")

async def test_plugin_load():
    engine = PluginEngine()
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", dir=".", prefix="_test_plugin_") as f:
        f.write("from agent.async_core.plugins import PluginBase\n")
        f.write("class MyPlugin(PluginBase):\n")
        f.write('    name = "temp_plugin"\n')
        f.write('    version = "0.1.0"\n')
        f.write("    async def initialize(self): return True\n")
        f.write("    def get_tools(self): return {'hello': lambda: 'world'}\n")
        f.flush()
        path = f.name
    try:
        result = await engine.load(path)
        assert result is True
        plugins = [p["name"] for p in engine.list_plugins()]
        assert len(plugins) == 1
        tools = engine.get_tools()
        assert "hello" in tools
    finally:
        os.unlink(path)

async def test_plugin_unload():
    engine = PluginEngine()
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", dir=".", prefix="_test_unload_") as f:
        f.write("from agent.async_core.plugins import PluginBase\n")
        f.write("class UP(PluginBase):\n")
        f.write('    name = "unload_test"\n')
        f.write('    version = "0.1.0"\n')
        f.write("    async def initialize(self): return True\n")
        f.flush()
        path = f.name
    try:
        await engine.load(path)
        plugins_before = engine.list_plugins()
        assert len(plugins_before) == 1
        actual_name = plugins_before[0]["name"]
        result = await engine.unload(actual_name)
        assert result is True
        assert len(engine.list_plugins()) == 0
    finally:
        if os.path.exists(path):
            os.unlink(path)

test("Plugin load", test_plugin_load)
test("Plugin unload", test_plugin_unload)

# ========== Module 5: Event Bus ==========
print("\n--- Module 5: Event Bus ---")

async def test_event_emit():
    bus = EventBus()
    received = []
    bus.on("test", lambda e: received.append(e.data))
    await bus.emit("test", {"key": "value"})
    assert len(received) == 1
    assert received[0]["key"] == "value"

async def test_event_priority():
    bus = EventBus()
    order = []
    bus.on("test", lambda e: order.append("low"), priority=EventPriority.LOW)
    bus.on("test", lambda e: order.append("high"), priority=EventPriority.HIGH)
    bus.on("test", lambda e: order.append("normal"), priority=EventPriority.NORMAL)
    await bus.emit("test")
    assert order == ["high", "normal", "low"]

async def test_event_once():
    bus = EventBus()
    count = [0]
    bus.once("test", lambda e: count.__setitem__(0, count[0] + 1))
    await bus.emit("test")
    await bus.emit("test")
    assert count[0] == 1

async def test_event_wildcard():
    bus = EventBus()
    received = []
    bus.on("tool.*", lambda e: received.append(e.name))
    await bus.emit("tool.start", {"tool": "echo"})
    await bus.emit("tool.end", {"tool": "echo"})
    await bus.emit("chat.message", {})
    assert len(received) == 2

async def test_event_history():
    bus = EventBus()
    await bus.emit("test1")
    await bus.emit("test2")
    await bus.emit("test1")
    history = bus.replay("test1")
    assert len(history) == 2

async def test_event_stop():
    bus = EventBus()
    order = []
    def stopper(e):
        order.append("first")
        e.stop()
    bus.on("test", stopper, priority=EventPriority.HIGH)
    bus.on("test", lambda e: order.append("second"), priority=EventPriority.LOW)
    await bus.emit("test")
    assert order == ["first"]

async def test_event_stats():
    bus = EventBus()
    bus.on("test", lambda e: None)
    bus.on("other", lambda e: None)
    stats = bus.stats()
    assert stats["subscriptions"]["test"] == 1

test("Event emit/subscribe", test_event_emit)
test("Event priority ordering", test_event_priority)
test("Event once handler", test_event_once)
test("Event wildcard matching", test_event_wildcard)
test("Event history/replay", test_event_history)
test("Event stop propagation", test_event_stop)
test("Event stats", test_event_stats)

# ========== Module 6: Streaming ==========
print("\n--- Module 6: Streaming Engine ---")

async def test_stream_session():
    engine = StreamingEngine()
    session = engine.create_session(StreamType.SSE)
    assert session.session_id
    assert not session._closed

async def test_stream_send_close():
    engine = StreamingEngine()
    session = engine.create_session()
    await session.send_text("hello")
    await session.send_text("world")
    await session.close()
    chunks = []
    async for chunk in session:
        chunks.append(chunk)
    assert len(chunks) >= 2

async def test_stream_sse_format():
    chunk = StreamChunk(data="test data", event="message", id="abc")
    sse = chunk.to_sse()
    assert "data: test data" in sse
    assert "id: abc" in sse

async def test_stream_json_format():
    chunk = StreamChunk(data={"key": "value"}, event="test")
    j = json.loads(chunk.to_json())
    assert j["event"] == "test"
    assert j["data"]["key"] == "value"

async def test_stream_broadcast():
    engine = StreamingEngine()
    s1 = engine.create_session()
    s2 = engine.create_session()
    await engine.broadcast("hello everyone")
    await s1.close()
    await s2.close()

async def test_stream_stats():
    engine = StreamingEngine()
    engine.create_session()
    engine.create_session()
    stats = engine.stats()
    assert stats["active_sessions"] == 2

test("Stream session creation", test_stream_session)
test("Stream send/close", test_stream_send_close)
test("Stream SSE format", test_stream_sse_format)
test("Stream JSON format", test_stream_json_format)
test("Stream broadcast", test_stream_broadcast)
test("Stream stats", test_stream_stats)

# ========== Module 7: Budget ==========
print("\n--- Module 7: Budget Tracker ---")

def test_budget_calculate():
    tracker = BudgetTracker()
    cost = tracker.calculate_cost("gpt-4o", 1000, 500)
    expected = (1000/1_000_000 * 2.50) + (500/1_000_000 * 10.00)
    assert abs(cost - expected) < 0.0001

def test_budget_record():
    tracker = BudgetTracker()
    cost = tracker.record("gpt-4o", 1000, 500, session_id="s1")
    assert cost > 0
    usage = tracker.get_usage("global")
    assert usage["tokens"] == 1500

def test_budget_limit():
    tracker = BudgetTracker()
    tracker.set_limit("daily", BudgetLimit(scope=BudgetScope.GLOBAL, max_tokens=100, hard_limit=True))
    cost = tracker.record("gpt-4o-mini", 50, 30)
    assert cost >= 0
    cost = tracker.record("gpt-4o-mini", 50, 30)
    assert cost == -1

def test_budget_breakdown():
    tracker = BudgetTracker()
    tracker.record("gpt-4o", 1000, 500, operation="chat")
    tracker.record("gpt-4o-mini", 2000, 1000, operation="tool")
    breakdown = tracker.get_breakdown()
    assert "gpt-4o" in breakdown["by_model"]
    assert "chat" in breakdown["by_operation"]

def test_budget_custom_pricing():
    tracker = BudgetTracker()
    tracker.set_pricing("my-model", 1.0, 2.0)
    cost = tracker.calculate_cost("my-model", 1_000_000, 1_000_000)
    assert cost == 3.0

def test_budget_stats():
    tracker = BudgetTracker()
    tracker.record("gpt-4o", 100, 200)
    stats = tracker.stats()
    assert stats["total_records"] == 1
    assert stats["total_tokens"] == 300

test("Budget cost calculation", test_budget_calculate)
test("Budget record usage", test_budget_record)
test("Budget limits", test_budget_limit)
test("Budget breakdown", test_budget_breakdown)
test("Budget custom pricing", test_budget_custom_pricing)
test("Budget stats", test_budget_stats)

# ========== Module 8: Sessions ==========
print("\n--- Module 8: Session Manager ---")

def test_session_create():
    mgr = SessionManager()
    session = mgr.create(title="Test Session")
    assert session.session_id
    assert session.title == "Test Session"
    assert mgr.active == session

def test_session_messages():
    mgr = SessionManager()
    session = mgr.create()
    mgr.add_message(session.session_id, "user", "hello")
    mgr.add_message(session.session_id, "assistant", "hi there")
    assert session.message_count == 2

def test_session_checkpoint():
    mgr = SessionManager()
    session = mgr.create()
    mgr.add_message(session.session_id, "user", "message 1")
    cp = mgr.checkpoint(session.session_id, "before change")
    assert cp is not None
    mgr.add_message(session.session_id, "user", "message 2")
    assert session.message_count == 2
    mgr.restore(session.session_id, cp.checkpoint_id)
    assert session.message_count == 1

def test_session_branch():
    mgr = SessionManager()
    parent = mgr.create(title="Parent")
    mgr.add_message(parent.session_id, "user", "msg 1")
    mgr.add_message(parent.session_id, "assistant", "reply 1")
    child = mgr.branch(parent.session_id, title="Branch")
    assert child.parent_id == parent.session_id
    assert child.message_count == 2

def test_session_search():
    mgr = SessionManager()
    s1 = mgr.create(title="Session A")
    mgr.add_message(s1.session_id, "user", "Python is great")
    s2 = mgr.create(title="Session B")
    mgr.add_message(s2.session_id, "user", "JavaScript is also good")
    results = mgr.search_messages("Python")
    assert len(results) == 1
    assert results[0]["session_id"] == s1.session_id

def test_session_export_import():
    mgr = SessionManager()
    session = mgr.create(title="Export Test")
    mgr.add_message(session.session_id, "user", "hello")
    exported = mgr.export_session(session.session_id)
    assert exported is not None
    imported = mgr.import_session(exported)
    assert imported is not None
    assert imported.title == "Export Test"

def test_session_merge():
    mgr = SessionManager()
    s1 = mgr.create(title="Source")
    mgr.add_message(s1.session_id, "user", "from source")
    s2 = mgr.create(title="Target")
    mgr.add_message(s2.session_id, "user", "from target")
    mgr.merge(s1.session_id, s2.session_id)
    assert s2.message_count >= 2

def test_session_archive():
    mgr = SessionManager()
    session = mgr.create()
    mgr.archive(session.session_id)
    assert session.state == "archived"

def test_session_persistence():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        mgr1 = SessionManager(db_path=db_path)
        s = mgr1.create(title="Persistent")
        mgr1.add_message(s.session_id, "user", "hello")
        mgr2 = SessionManager(db_path=db_path)
        sessions = mgr2.list_sessions()
        assert any(ss["title"] == "Persistent" for ss in sessions)
    finally:
        os.unlink(db_path)

test("Session create", test_session_create)
test("Session messages", test_session_messages)
test("Session checkpoint/restore", test_session_checkpoint)
test("Session branching", test_session_branch)
test("Session search", test_session_search)
test("Session export/import", test_session_export_import)
test("Session merge", test_session_merge)
test("Session archive", test_session_archive)
test("Session SQLite persistence", test_session_persistence)

# ========== Module 9: CLI ==========
print("\n--- Module 9: Enhanced CLI ---")

def test_cli_config():
    config = CLIConfig(prompt="> ", color=True, syntax_highlight=True)
    assert config.prompt == "> "
    assert config.color is True

def test_command_registry():
    reg = CommandRegistry()
    reg.register("/test", lambda args: None, "Test command", ["/t"])
    assert reg.get("/test") is not None
    assert reg.get("/t") is not None
    completions = reg.complete("/te")
    assert "/test" in completions

def test_command_list():
    reg = CommandRegistry()
    reg.register("/cmd1", lambda a: None, "First command")
    reg.register("/cmd2", lambda a: None, "Second command")
    cmds = reg.list_all()
    assert len(cmds) == 2

test("CLI config", test_cli_config)
test("Command registry", test_command_registry)
test("Command listing", test_command_list)

# ========== Module 10: LLM Client ==========
print("\n--- Module 10: LLM Client ---")

def test_llm_config():
    from agent.async_core.llm_client import LLMConfig, Provider
    config = LLMConfig(provider=Provider.OPENAI, model="gpt-4o-mini", api_key="test-key")
    assert config.provider == Provider.OPENAI
    assert config.max_retries == 3

def test_llm_response():
    from agent.async_core.llm_client import LLMResponse
    resp = LLMResponse(content="hello", model="gpt-4o", input_tokens=10, output_tokens=5)
    assert resp.content == "hello"
    assert resp.input_tokens + resp.output_tokens == 15

def test_rate_limiter():
    from agent.async_core.llm_client import RateLimiter
    limiter = RateLimiter(requests_per_minute=100, tokens_per_minute=100000)
    # Should not block for first request
    asyncio.run(limiter.acquire(estimated_tokens=100))

def test_llm_stats():
    from agent.async_core.llm_client import LLMClient, LLMConfig
    client = LLMClient(LLMConfig())
    stats = client.stats()
    assert stats["total_requests"] == 0
    assert stats["cache_size"] == 0

test("LLM config", test_llm_config)
test("LLM response", test_llm_response)
test("Rate limiter", test_rate_limiter)
test("LLM stats", test_llm_stats)

# ========== Module 11: Embeddings ==========
print("\n--- Module 11: Embeddings ---")

def test_tfidf_embed():
    from agent.async_core.embeddings import TFIDFEmbedder
    embedder = TFIDFEmbedder(dimension=64)
    vec = embedder.embed("Hello world this is a test")
    assert len(vec) == 64
    assert sum(x * x for x in vec) > 0  # Not all zeros

def test_tfidf_batch():
    from agent.async_core.embeddings import TFIDFEmbedder
    embedder = TFIDFEmbedder(dimension=64)
    vecs = embedder.embed_batch(["hello", "world", "test"])
    assert len(vecs) == 3
    assert all(len(v) == 64 for v in vecs)

def test_tfidf_fit():
    from agent.async_core.embeddings import TFIDFEmbedder
    embedder = TFIDFEmbedder(dimension=64)
    embedder.fit(["Python is great", "JavaScript is also good", "Python and JavaScript"])
    vec1 = embedder.embed("Python programming")
    vec2 = embedder.embed("JavaScript web")
    # Different texts should produce different vectors
    assert vec1 != vec2

def test_tfidf_similarity():
    from agent.async_core.embeddings import TFIDFEmbedder
    embedder = TFIDFEmbedder(dimension=128)
    embedder.fit(["machine learning", "deep learning", "cooking recipes"])
    v1 = embedder.embed("machine learning algorithms")
    v2 = embedder.embed("deep learning neural networks")
    v3 = embedder.embed("cooking Italian pasta")
    # v1 and v2 should be more similar than v1 and v3
    sim_12 = sum(a * b for a, b in zip(v1, v2))
    sim_13 = sum(a * b for a, b in zip(v1, v3))
    assert sim_12 > sim_13

test("TF-IDF embed", test_tfidf_embed)
test("TF-IDF batch", test_tfidf_batch)
test("TF-IDF fit", test_tfidf_fit)
test("TF-IDF similarity", test_tfidf_similarity)

# ========== Module 12: Config Manager ==========
print("\n--- Module 12: Config Manager ---")

def test_config_defaults():
    from agent.async_core.config import ConfigManager
    cfg = ConfigManager(config_dir=tempfile.mkdtemp())
    assert cfg.get("llm.provider") == "openai"
    assert cfg.get("llm.max_tokens") == 4096
    assert cfg.get("budget.daily_limit_usd") == 10.0

def test_config_set_get():
    from agent.async_core.config import ConfigManager
    cfg = ConfigManager(config_dir=tempfile.mkdtemp())
    cfg.set("llm.model", "gpt-4o")
    assert cfg.get("llm.model") == "gpt-4o"
    cfg.set("server.port", 9090)
    assert cfg.get("server.port") == 9090

def test_config_section():
    from agent.async_core.config import ConfigManager
    cfg = ConfigManager(config_dir=tempfile.mkdtemp())
    llm_section = cfg.get_section("llm")
    assert "provider" in llm_section
    assert "model" in llm_section

def test_config_save_load():
    from agent.async_core.config import ConfigManager
    d = tempfile.mkdtemp()
    cfg1 = ConfigManager(config_dir=d)
    cfg1.set("llm.model", "custom-model")
    cfg1.save()
    cfg2 = ConfigManager(config_dir=d)
    assert cfg2.get("llm.model") == "custom-model"

def test_config_api_key():
    from agent.async_core.config import ConfigManager
    cfg = ConfigManager(config_dir=tempfile.mkdtemp())
    cfg.set("llm.api_key", "sk-test-123")
    assert cfg.get_api_key("openai") == "sk-test-123"

test("Config defaults", test_config_defaults)
test("Config set/get", test_config_set_get)
test("Config section", test_config_section)
test("Config save/load", test_config_save_load)
test("Config API key", test_config_api_key)

# ========== Module 13: Health Monitor ==========
print("\n--- Module 13: Health Monitor ---")

def test_health_check():
    from agent.async_core.health import HealthMonitor, HealthCheck, HealthStatus
    monitor = HealthMonitor()
    monitor.register_check("test", lambda: HealthCheck(name="test", status=HealthStatus.HEALTHY))
    results = asyncio.run(monitor.run_checks())
    assert "test" in results
    assert results["test"].status == HealthStatus.HEALTHY

def test_health_overall():
    from agent.async_core.health import HealthMonitor, HealthCheck, HealthStatus
    monitor = HealthMonitor()
    monitor.register_check("ok", lambda: HealthCheck(name="ok", status=HealthStatus.HEALTHY))
    monitor.register_check("bad", lambda: HealthCheck(name="bad", status=HealthStatus.CRITICAL))
    asyncio.run(monitor.run_checks())
    assert monitor.overall_status() == HealthStatus.CRITICAL

def test_health_metrics():
    from agent.async_core.health import HealthMonitor
    monitor = HealthMonitor()
    metrics = monitor.get_metrics()
    assert metrics.uptime >= 0
    assert metrics.request_count == 0

def test_health_request_recording():
    from agent.async_core.health import HealthMonitor
    monitor = HealthMonitor()
    monitor.record_request(latency_ms=100)
    monitor.record_request(latency_ms=200, error=True)
    metrics = monitor.get_metrics()
    assert metrics.request_count == 2
    assert metrics.error_count == 1

def test_health_report():
    from agent.async_core.health import HealthMonitor
    monitor = HealthMonitor()
    report = monitor.report()
    assert "status" in report
    assert "uptime" in report

test("Health check", test_health_check)
test("Health overall status", test_health_overall)
test("Health metrics", test_health_metrics)
test("Health request recording", test_health_request_recording)
test("Health report", test_health_report)

# ========== Module 14: DI Container ==========
print("\n--- Module 14: DI Container ---")

def test_container_register_get():
    from agent.async_core.container import Container
    c = Container()
    c.register_singleton("greeting", "hello world")
    assert c.get("greeting") == "hello world"

def test_container_factory():
    from agent.async_core.container import Container
    c = Container()
    c.register_factory("counter", lambda: {"count": 0})
    result = c.get("counter")
    assert result["count"] == 0
    # Should be singleton after first get
    result["count"] = 42
    assert c.get("counter")["count"] == 42

def test_container_has():
    from agent.async_core.container import Container
    c = Container()
    c.register_singleton("x", 1)
    c.register_factory("y", lambda: 2)
    assert c.has("x")
    assert c.has("y")
    assert not c.has("z")

def test_container_list():
    from agent.async_core.container import Container
    c = Container()
    c.register_singleton("a", "A")
    c.register_factory("b", lambda: "B")
    services = c.list_services()
    assert "a" in services
    assert "b" in services

def test_container_key_error():
    from agent.async_core.container import Container
    c = Container()
    try:
        c.get("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

test("Container register/get", test_container_register_get)
test("Container factory", test_container_factory)
test("Container has", test_container_has)
test("Container list", test_container_list)
test("Container KeyError", test_container_key_error)

# ========== Module 15: Server ==========
print("\n--- Module 15: HTTP Server ---")

def test_server_config():
    from agent.async_core.server import ServerConfig
    cfg = ServerConfig(host="127.0.0.1", port=9090)
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9090
    assert cfg.cors_origins == ["*"]

def test_request_handler_routing():
    from agent.async_core.server import RequestHandler
    h = RequestHandler()
    @h.route("/test", ["GET"])
    def handler():
        return {"ok": True}
    route = h.get_route("GET", "/test")
    assert route is not None
    assert route["path"] == "/test"

def test_request_handler_404():
    from agent.async_core.server import RequestHandler
    h = RequestHandler()
    route = h.get_route("GET", "/nonexistent")
    assert route is None

def test_server_stats():
    from agent.async_core.server import AsyncHTTPServer, ServerConfig
    server = AsyncHTTPServer(ServerConfig(port=19999))
    stats = server.stats()
    assert stats["connections"] == 0
    assert stats["routes"] >= 2  # health + stats

test("Server config", test_server_config)
test("Request handler routing", test_request_handler_routing)
test("Request handler 404", test_request_handler_404)
test("Server stats", test_server_stats)

# ========== Summary ==========
print("\n" + "=" * 50)
print("  Results: %d passed, %d failed" % (passed, failed))
if errors:
    print("\n  Errors:")
    for name, err in errors:
        print("    FAIL %s: %s" % (name, err))
print("=" * 50)
sys.exit(1 if failed > 0 else 0)
