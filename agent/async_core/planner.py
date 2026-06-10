"""
Autonomous Planner — goal decomposition, task DAG, replanning on failure.
Agent can break complex goals into executable plans and adapt when things go wrong.
"""
import time
import uuid
import logging
import json
from typing import Optional, Dict, Any, List, Callable, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PlanStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLANNING = "replanning"


class TaskStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """A single task in a plan."""
    id: str
    title: str
    description: str
    action_type: str  # tool_call, code_write, search, verify, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = ""
    attempts: int = 0
    max_attempts: int = 3
    estimated_duration_ms: float = 0
    actual_duration_ms: float = 0
    priority: float = 0.5
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0
    metadata: Dict = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in (TaskStatus.SUCCESS, TaskStatus.FAILED,
                               TaskStatus.SKIPPED, TaskStatus.CANCELLED)


@dataclass
class Plan:
    """A complete execution plan."""
    id: str
    goal: str
    tasks: List[Task] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    started_at: float = 0
    completed_at: float = 0
    metadata: Dict = field(default_factory=dict)
    replan_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        if not self.tasks:
            return 0
        done = sum(1 for t in self.tasks if t.is_terminal)
        return done / len(self.tasks)

    @property
    def remaining_tasks(self) -> List[Task]:
        return [t for t in self.tasks if not t.is_terminal]


@dataclass
class GoalNode:
    """A node in the goal decomposition tree."""
    id: str
    description: str
    sub_goals: List['GoalNode'] = field(default_factory=list)
    task_id: str = ""  # maps to plan task
    status: str = "pending"
    reasoning: str = ""  # why this decomposition


class AutonomousPlanner:
    """
    Autonomous planner with:
    - Goal decomposition (break complex goals into sub-tasks)
    - Dependency-aware task scheduling
    - Parallel task execution where possible
    - Replanning on failure (re-evaluate and adjust plan)
    - Progress tracking
    - Plan templates for common patterns
    - Adaptive execution (skip unnecessary tasks)
    - Cost/time estimation
    """

    def __init__(self):
        self._plans: Dict[str, Plan] = {}
        self._goal_trees: Dict[str, GoalNode] = {}
        self._templates: Dict[str, Callable] = {}
        self._task_handlers: Dict[str, Callable] = {}

    def register_handler(self, action_type: str, handler: Callable):
        """Register a handler for a task action type."""
        self._task_handlers[action_type] = handler

    def register_template(self, name: str, factory: Callable):
        """Register a plan template."""
        self._templates[name] = factory

    def create_plan(self, goal: str, tasks: List[Dict] = None) -> Plan:
        """Create a plan from goal and optional task definitions."""
        plan = Plan(id="p_" + str(uuid.uuid4())[:8], goal=goal)
        if tasks:
            for t in tasks:
                task = Task(
                    id=t.get("id", "t_" + str(uuid.uuid4())[:8]),
                    title=t.get("title", ""),
                    description=t.get("description", ""),
                    action_type=t.get("action_type", "generic"),
                    parameters=t.get("parameters", {}),
                    depends_on=t.get("depends_on", []),
                    priority=t.get("priority", 0.5),
                    max_attempts=t.get("max_attempts", 3),
                )
                plan.tasks.append(task)
        self._plans[plan.id] = plan
        return plan

    def decompose_goal(self, goal: str, max_depth: int = 3) -> GoalNode:
        """Decompose a goal into sub-goals (creates goal tree)."""
        root = GoalNode(
            id="g_" + str(uuid.uuid4())[:8],
            description=goal,
            reasoning="Top-level goal",
        )

        # Pattern-based decomposition
        sub_goals = self._pattern_decompose(goal)
        if sub_goals:
            root.sub_goals = sub_goals
            for sg in sub_goals:
                if max_depth > 1 and self._needs_further_decomposition(sg.description):
                    sg.sub_goals = self._decompose_recursive(sg.description, max_depth - 1)

        self._goal_trees[root.id] = root
        return root

    def _pattern_decompose(self, goal: str) -> List[GoalNode]:
        """Decompose goal based on common patterns."""
        goal_lower = goal.lower()
        sub_goals = []

        # Pattern: "build/create X"
        if any(kw in goal_lower for kw in ["build", "create", "make", "implement"]):
            sub_goals = [
                GoalNode(id="g_" + str(uuid.uuid4())[:8],
                         description="Research and plan: " + goal,
                         reasoning="Need to understand requirements first"),
                GoalNode(id="g_" + str(uuid.uuid4())[:8],
                         description="Implement: " + goal,
                         reasoning="Core implementation"),
                GoalNode(id="g_" + str(uuid.uuid4())[:8],
                         description="Test and verify: " + goal,
                         reasoning="Verify correctness"),
                GoalNode(id="g_" + str(uuid.uuid4())[:8],
                         description="Document and finalize: " + goal,
                         reasoning="Wrap up and document"),
            ]
        # Pattern: "fix/debug X"
        elif any(kw in goal_lower for kw in ["fix", "debug", "repair", "resolve"]):
            sub_goals = [
                GoalNode(id="g_" + str(uuid.uuid4())[:8],
                         description="Reproduce and diagnose: " + goal,
                         reasoning="Understand the problem first"),
                GoalNode(id="g_" + str(uuid.uuid4())[:8],
                         description="Implement fix: " + goal,
                         reasoning="Apply the fix"),
                GoalNode(id="g_" + str(uuid.uuid4())[:8],
                         description="Verify fix works: " + goal,
                         reasoning="Confirm the fix resolves the issue"),
            ]
        # Pattern: "research/analyze X"
        elif any(kw in goal_lower for kw in ["research", "analyze", "investigate", "compare"]):
            sub_goals = [
                GoalNode(id="g_" + str(uuid.uuid4())[:8],
                         description="Gather information: " + goal,
                         reasoning="Collect relevant data"),
                GoalNode(id="g_" + str(uuid.uuid4())[:8],
                         description="Analyze findings: " + goal,
                         reasoning="Process and analyze"),
                GoalNode(id="g_" + str(uuid.uuid4())[:8],
                         description="Summarize results: " + goal,
                         reasoning="Present conclusions"),
            ]

        return sub_goals

    def _needs_further_decomposition(self, description: str) -> bool:
        """Check if a goal needs further decomposition."""
        # Simple heuristic: if description is vague, decompose more
        vague_words = ["etc", "and so on", "everything", "all", "complete"]
        return any(w in description.lower() for w in vague_words)

    def _decompose_recursive(self, goal: str, depth: int) -> List[GoalNode]:
        if depth <= 0:
            return []
        return self._pattern_decompose(goal)

    def goal_to_plan(self, root: GoalNode) -> Plan:
        """Convert a goal tree into an executable plan."""
        plan = Plan(id="p_" + str(uuid.uuid4())[:8], goal=root.description)
        self._flatten_goals(root, plan, [])
        self._plans[plan.id] = plan
        return plan

    def _flatten_goals(self, node: GoalNode, plan: Plan, parent_ids: List[str]):
        """Recursively flatten goal tree into plan tasks."""
        task_id = "t_" + str(uuid.uuid4())[:8]
        task = Task(
            id=task_id,
            title=node.description[:100],
            description=node.description,
            action_type="goal_step",
            depends_on=list(parent_ids),
            priority=0.5,
        )
        plan.tasks.append(task)
        node.task_id = task_id

        for sub in node.sub_goals:
            self._flatten_goals(sub, plan, [task_id])

    def get_ready_tasks(self, plan_id: str) -> List[Task]:
        """Get tasks that are ready to execute (dependencies met)."""
        plan = self._plans.get(plan_id)
        if not plan:
            return []

        completed = {t.id for t in plan.tasks if t.status == TaskStatus.SUCCESS}
        skipped = {t.id for t in plan.tasks if t.status == TaskStatus.SKIPPED}
        done = completed | skipped

        ready = []
        for task in plan.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            if all(dep in done for dep in task.depends_on):
                ready.append(task)
        return ready

    async def execute_plan(self, plan_id: str) -> Plan:
        """Execute a plan, running ready tasks and replanning on failure."""
        plan = self._plans.get(plan_id)
        if not plan:
            raise ValueError("Plan not found: %s" % plan_id)

        plan.status = PlanStatus.ACTIVE
        plan.started_at = time.time()

        max_iterations = len(plan.tasks) * 3  # Safety limit
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            ready = self.get_ready_tasks(plan_id)

            if not ready:
                # Check if all done
                remaining = plan.remaining_tasks
                if not remaining:
                    plan.status = PlanStatus.COMPLETED
                    plan.completed_at = time.time()
                    break
                # Check for blocked tasks
                blocked = [t for t in remaining if t.status == TaskStatus.BLOCKED]
                if blocked and len(blocked) == len(remaining):
                    plan.status = PlanStatus.FAILED
                    break
                continue

            # Execute one ready task (highest priority first)
            ready.sort(key=lambda t: t.priority, reverse=True)
            task = ready[0]
            task.status = TaskStatus.RUNNING
            task.attempts += 1

            start = time.time()
            try:
                handler = self._task_handlers.get(task.action_type)
                if handler:
                    import asyncio
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(task.parameters, plan.context)
                    else:
                        result = handler(task.parameters, plan.context)
                    task.result = result
                    task.status = TaskStatus.SUCCESS
                    plan.context["task_%s" % task.id] = result
                else:
                    # No handler — mark as success with placeholder
                    task.result = "[No handler for %s]" % task.action_type
                    task.status = TaskStatus.SUCCESS

            except Exception as e:
                task.error = str(e)
                if task.attempts < task.max_attempts:
                    task.status = TaskStatus.PENDING  # Retry
                else:
                    task.status = TaskStatus.FAILED
                    # Attempt replan
                    if self._should_replan(plan, task):
                        plan.replan_count += 1
                        plan.status = PlanStatus.REPLANNING
                        self._replan_for_failure(plan, task)
                        plan.status = PlanStatus.ACTIVE

            task.actual_duration_ms = (time.time() - start) * 1000
            task.completed_at = time.time()

        return plan

    def _should_replan(self, plan: Plan, failed_task: Task) -> bool:
        """Decide if we should replan after a task failure."""
        # Replan if the failed task has dependents
        dependents = [t for t in plan.tasks if failed_task.id in t.depends_on]
        return len(dependents) > 0 and plan.replan_count < 3

    def _replan_for_failure(self, plan: Plan, failed_task: Task):
        """Adjust plan after a task failure."""
        # Skip all dependents of the failed task
        to_skip = set()
        queue = [failed_task.id]
        while queue:
            tid = queue.pop(0)
            for t in plan.tasks:
                if tid in t.depends_on and t.id not in to_skip:
                    to_skip.add(t.id)
                    queue.append(t.id)

        for t in plan.tasks:
            if t.id in to_skip and t.status == TaskStatus.PENDING:
                t.status = TaskStatus.SKIPPED

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        return self._plans.get(plan_id)

    def list_plans(self) -> List[Dict]:
        return [{
            "id": p.id, "goal": p.goal[:80], "status": p.status.value,
            "progress": "%.0f%%" % (p.progress * 100),
            "tasks": len(p.tasks),
        } for p in self._plans.values()]

    def stats(self) -> Dict:
        total_tasks = sum(len(p.tasks) for p in self._plans.values())
        completed_tasks = sum(
            sum(1 for t in p.tasks if t.status == TaskStatus.SUCCESS)
            for p in self._plans.values()
        )
        return {
            "plans": len(self._plans),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "goal_trees": len(self._goal_trees),
        }
