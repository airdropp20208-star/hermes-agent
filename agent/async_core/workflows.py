"""
Workflow Engine — DAG execution, parallel steps, branching, retry.
Execute complex multi-step workflows with dependency resolution.
"""
import asyncio
import time
import uuid
import logging
from typing import Optional, Dict, Any, List, Callable, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class StepResult:
    """Result of a workflow step."""
    step_id: str
    status: StepStatus
    output: Any = None
    error: str = ""
    duration_ms: float = 0
    attempt: int = 1


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    id: str
    name: str
    handler: Callable  # async function(context) -> result
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[Callable] = None  # fn(context) -> bool
    retry_count: int = 0
    retry_delay: float = 1.0
    timeout: float = 60
    on_failure: str = "fail"  # "fail" | "skip" | "abort"
    metadata: Dict = field(default_factory=dict)


@dataclass
class WorkflowDef:
    """Complete workflow definition."""
    id: str
    name: str
    steps: List[WorkflowStep]
    description: str = ""
    max_parallel: int = 5
    on_step_failure: str = "abort"  # "abort" | "continue" | "skip_dependents"
    timeout: float = 600


@dataclass
class WorkflowRun:
    """A running/completed workflow instance."""
    run_id: str
    workflow_id: str
    status: str = "pending"
    started_at: float = 0
    finished_at: float = 0
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class WorkflowEngine:
    """
    Workflow engine with:
    - DAG-based step execution
    - Automatic dependency resolution
    - Parallel step execution (respecting max_parallel)
    - Step retry with backoff
    - Conditional steps
    - Context passing between steps
    - Workflow templates
    - Run history
    """

    def __init__(self, max_parallel: int = 10):
        self.max_parallel = max_parallel
        self._workflows: Dict[str, WorkflowDef] = {}
        self._runs: List[WorkflowRun] = []
        self._templates: Dict[str, Callable] = {}

    def define(self, name: str, steps: List[WorkflowStep],
               description: str = "", max_parallel: int = 5,
               on_step_failure: str = "abort") -> WorkflowDef:
        """Define a workflow."""
        wf = WorkflowDef(
            id=str(uuid.uuid4())[:8],
            name=name,
            steps=steps,
            description=description,
            max_parallel=max_parallel,
            on_step_failure=on_step_failure,
        )
        self._workflows[wf.id] = wf
        return wf

    def register_template(self, name: str, factory: Callable):
        """Register a workflow template factory."""
        self._templates[name] = factory

    async def run(self, workflow_id: str, context: Dict = None) -> WorkflowRun:
        """Execute a workflow."""
        wf = self._workflows.get(workflow_id)
        if not wf:
            raise ValueError(f"Workflow not found: {workflow_id}")

        run = WorkflowRun(
            run_id=str(uuid.uuid4())[:8],
            workflow_id=workflow_id,
            status="running",
            started_at=time.time(),
            context=context or {},
        )
        self._runs.append(run)

        # Build dependency graph
        step_map = {s.id: s for s in wf.steps}
        completed: Set[str] = set()
        failed: Set[str] = set()
        semaphore = asyncio.Semaphore(wf.max_parallel)

        try:
            while len(completed) + len(failed) < len(wf.steps):
                # Find steps ready to run
                ready = []
                for step in wf.steps:
                    if step.id in completed or step.id in failed:
                        continue
                    if step.id in run.step_results and run.step_results[step.id].status in (
                        StepStatus.RUNNING, StepStatus.RETRYING
                    ):
                        continue
                    # Check dependencies
                    deps_met = all(d in completed for d in step.depends_on)
                    deps_failed = any(d in failed for d in step.depends_on)

                    if deps_failed:
                        if wf.on_step_failure == "skip_dependents":
                            run.step_results[step.id] = StepResult(
                                step_id=step.id, status=StepStatus.SKIPPED
                            )
                            failed.add(step.id)
                        continue

                    if deps_met:
                        ready.append(step)

                if not ready:
                    # Check if all steps are done
                    if len(completed) + len(failed) >= len(wf.steps):
                        break
                    # Deadlock or waiting
                    await asyncio.sleep(0.1)
                    continue

                # Execute ready steps in parallel
                tasks = []
                for step in ready:
                    tasks.append(asyncio.create_task(
                        self._execute_step(step, run, semaphore)
                    ))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for step, result in zip(ready, results):
                    if isinstance(result, Exception):
                        run.step_results[step.id] = StepResult(
                            step_id=step.id, status=StepStatus.FAILED,
                            error=str(result)
                        )
                        failed.add(step.id)
                        if wf.on_step_failure == "abort":
                            run.status = "failed"
                            run.error = f"Step {step.id} failed: {result}"
                            run.finished_at = time.time()
                            return run
                    else:
                        run.step_results[step.id] = result
                        if result.status == StepStatus.SUCCESS:
                            completed.add(step.id)
                            run.context[f"step_{step.id}"] = result.output
                        else:
                            failed.add(step.id)

            # Determine final status
            if failed:
                run.status = "failed"
                run.error = "Steps failed: %s" % ", ".join(failed)
            else:
                run.status = "completed"

        except Exception as e:
            run.status = "failed"
            run.error = str(e)

        run.finished_at = time.time()
        return run

    async def _execute_step(self, step: WorkflowStep, run: WorkflowRun,
                            semaphore: asyncio.Semaphore) -> StepResult:
        """Execute a single workflow step with retry."""
        async with semaphore:
            # Check condition
            if step.condition:
                try:
                    if asyncio.iscoroutinefunction(step.condition):
                        should_run = await step.condition(run.context)
                    else:
                        should_run = step.condition(run.context)
                    if not should_run:
                        return StepResult(step_id=step.id, status=StepStatus.SKIPPED)
                except Exception as e:
                    return StepResult(step_id=step.id, status=StepStatus.FAILED,
                                     error="Condition error: %s" % e)

            last_error = ""
            for attempt in range(1 + step.retry_count):
                start = time.monotonic()
                try:
                    if asyncio.iscoroutinefunction(step.handler):
                        output = await asyncio.wait_for(
                            step.handler(run.context), timeout=step.timeout
                        )
                    else:
                        output = step.handler(run.context)

                    return StepResult(
                        step_id=step.id, status=StepStatus.SUCCESS,
                        output=output, attempt=attempt + 1,
                        duration_ms=(time.monotonic() - start) * 1000,
                    )
                except asyncio.TimeoutError:
                    last_error = "Timeout after %ds" % step.timeout
                except Exception as e:
                    last_error = "%s: %s" % (type(e).__name__, e)

                if attempt < step.retry_count:
                    await asyncio.sleep(step.retry_delay * (2 ** attempt))

            return StepResult(
                step_id=step.id, status=StepStatus.FAILED,
                error=last_error, attempt=step.retry_count + 1,
            )

    def get_run(self, run_id: str) -> Optional[WorkflowRun]:
        for run in self._runs:
            if run.run_id == run_id:
                return run
        return None

    def list_workflows(self) -> List[Dict]:
        return [{"id": wf.id, "name": wf.name, "steps": len(wf.steps),
                 "description": wf.description}
                for wf in self._workflows.values()]

    def stats(self) -> Dict:
        total_runs = len(self._runs)
        completed = sum(1 for r in self._runs if r.status == "completed")
        failed = sum(1 for r in self._runs if r.status == "failed")
        return {
            "workflows": len(self._workflows),
            "total_runs": total_runs,
            "completed": completed,
            "failed": failed,
        }
