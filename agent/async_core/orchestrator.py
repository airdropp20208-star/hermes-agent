"""
Multi-Agent Orchestrator — spawn, manage, coordinate multiple agents.
Supports delegation trees, parallel execution, and result aggregation.
"""
import asyncio
import uuid
import time
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    WORKER = "worker"          # Does specific tasks
    ORCHESTRATOR = "orchestrator"  # Delegates to workers
    REVIEWER = "reviewer"      # Reviews worker output
    RESEARCHER = "researcher"  # Gathers information
    CODER = "coder"            # Writes code
    ANALYST = "analyst"        # Analyzes data


@dataclass
class AgentSpec:
    """Specification for spawning an agent."""
    role: AgentRole = AgentRole.WORKER
    goal: str = ""
    context: str = ""
    toolsets: List[str] = field(default_factory=list)
    model: str = ""
    max_iterations: int = 50
    timeout: float = 300
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from a completed agent."""
    agent_id: str
    spec: AgentSpec
    output: str
    success: bool
    duration: float
    token_count: int = 0
    error: Optional[str] = None
    child_results: List['AgentResult'] = field(default_factory=list)


class AgentProcess:
    """Running agent instance."""

    def __init__(self, agent_id: str, spec: AgentSpec):
        self.agent_id = agent_id
        self.spec = spec
        self.state = "pending"
        self.task: Optional[asyncio.Task] = None
        self.result: Optional[AgentResult] = None
        self.started_at: Optional[float] = None
        self.children: List['AgentProcess'] = []

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at if self.started_at else 0


class AsyncOrchestrator:
    """
    Multi-agent orchestrator with:
    - Parallel agent spawning
    - Delegation trees (orchestrator → workers)
    - Result aggregation
    - Resource limits (max concurrent agents)
    - Timeout management
    - Progress tracking
    """

    def __init__(self, max_concurrent: int = 10, max_depth: int = 3):
        self.max_concurrent = max_concurrent
        self.max_depth = max_depth
        self._agents: Dict[str, AgentProcess] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._agent_factory: Optional[Callable] = None
        self._results: Dict[str, AgentResult] = {}

    def set_agent_factory(self, factory: Callable):
        """Set the factory function that creates agent instances."""
        self._agent_factory = factory

    async def spawn(self, spec: AgentSpec) -> AgentResult:
        """Spawn a single agent and wait for result."""
        agent_id = str(uuid.uuid4())[:8]
        process = AgentProcess(agent_id, spec)
        self._agents[agent_id] = process

        async with self._semaphore:
            process.state = "running"
            process.started_at = time.time()
            try:
                result = await asyncio.wait_for(
                    self._run_agent(process),
                    timeout=spec.timeout
                )
                process.state = "done"
                process.result = result
                self._results[agent_id] = result
                return result
            except asyncio.TimeoutError:
                process.state = "timeout"
                result = AgentResult(
                    agent_id=agent_id,
                    spec=spec,
                    output="",
                    success=False,
                    duration=time.time() - process.started_at,
                    error=f"Timeout after {spec.timeout}s"
                )
                process.result = result
                return result
            except Exception as e:
                process.state = "error"
                result = AgentResult(
                    agent_id=agent_id,
                    spec=spec,
                    output="",
                    success=False,
                    duration=time.time() - process.started_at,
                    error=str(e)
                )
                process.result = result
                return result

    async def spawn_parallel(self, specs: List[AgentSpec]) -> List[AgentResult]:
        """Spawn multiple agents in parallel."""
        tasks = [asyncio.create_task(self.spawn(s)) for s in specs]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def spawn_tree(self, root_spec: AgentSpec, child_specs: List[AgentSpec]) -> AgentResult:
        """Spawn orchestrator with child workers — delegation tree."""
        # First spawn the orchestrator
        root_result = await self.spawn(root_spec)

        # If orchestrator succeeded and has children to delegate
        if root_result.success and child_specs:
            # Update child specs with parent context
            for cs in child_specs:
                cs.parent_id = root_result.agent_id
                cs.context = f"Parent output:\n{root_result.output}\n\nYour task:\n{cs.context}"

            # Spawn children in parallel
            child_results = await self.spawn_parallel(child_specs)

            # Aggregate results
            root_result.child_results = child_results
            successful = [r for r in child_results if r.success]
            failed = [r for r in child_results if not r.success]

            if failed:
                root_result.output += f"\n\n--- Children Results ({len(successful)}/{len(child_results)} succeeded) ---"
                for r in child_results:
                    status = "✅" if r.success else "❌"
                    root_result.output += f"\n{status} [{r.agent_id}] {r.spec.goal[:50]}...: {r.output[:200]}"

        return root_result

    async def _run_agent(self, process: AgentProcess) -> AgentResult:
        """Run a single agent — calls the factory."""
        start = time.time()

        if self._agent_factory:
            output = await self._agent_factory(process.spec)
        else:
            # Default: simulate agent work
            output = f"Agent {process.agent_id} ({process.spec.role.value}): {process.spec.goal}"

        return AgentResult(
            agent_id=process.agent_id,
            spec=process.spec,
            output=str(output),
            success=True,
            duration=time.time() - start,
        )

    def get_status(self) -> Dict:
        """Get orchestrator status."""
        agents_by_state = {}
        for a in self._agents.values():
            agents_by_state.setdefault(a.state, []).append(a.agent_id)

        return {
            "total_agents": len(self._agents),
            "by_state": {k: len(v) for k, v in agents_by_state.items()},
            "max_concurrent": self.max_concurrent,
            "results_count": len(self._results),
        }

    def get_agent(self, agent_id: str) -> Optional[AgentProcess]:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def cancel_all(self):
        """Cancel all running agents."""
        for a in self._agents.values():
            if a.task and not a.task.done():
                a.task.cancel()
                a.state = "cancelled"
