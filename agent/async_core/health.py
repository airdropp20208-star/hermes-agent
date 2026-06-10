"""
Health Monitor — metrics, watchdog, auto-recovery.
Monitors all async core components and alerts on issues.
"""
import asyncio
import time
import logging
import os
import psutil
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """A single health check result."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metrics:
    """System metrics snapshot."""
    cpu_percent: float = 0
    memory_mb: float = 0
    memory_percent: float = 0
    disk_percent: float = 0
    open_files: int = 0
    threads: int = 0
    uptime: float = 0
    request_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0
    active_agents: int = 0
    active_sessions: int = 0


class HealthMonitor:
    """
    Health monitoring with:
    - Periodic health checks
    - System metrics collection
    - Auto-recovery actions
    - Alert callbacks
    - Component status tracking
    - Uptime tracking
    """

    def __init__(self, check_interval: float = 30):
        self.check_interval = check_interval
        self._checks: Dict[str, Callable] = {}
        self._results: Dict[str, HealthCheck] = {}
        self._alert_callbacks: List[Callable] = []
        self._recovery_actions: Dict[str, Callable] = {}
        self._metrics_history: List[Metrics] = []
        self._start_time = time.time()
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Counters
        self._request_count = 0
        self._error_count = 0
        self._latencies: List[float] = []

    def register_check(self, name: str, check_fn: Callable,
                       recovery_fn: Callable = None):
        """Register a health check."""
        self._checks[name] = check_fn
        if recovery_fn:
            self._recovery_actions[name] = recovery_fn

    def on_alert(self, callback: Callable):
        """Register alert callback."""
        self._alert_callbacks.append(callback)

    def record_request(self, latency_ms: float = 0, error: bool = False):
        """Record a request for metrics."""
        self._request_count += 1
        if error:
            self._error_count += 1
        if latency_ms > 0:
            self._latencies.append(latency_ms)
            if len(self._latencies) > 1000:
                self._latencies = self._latencies[-1000:]

    async def run_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        for name, check_fn in self._checks.items():
            try:
                start = time.monotonic()
                if asyncio.iscoroutinefunction(check_fn):
                    result = await check_fn()
                else:
                    result = check_fn()
                latency = (time.monotonic() - start) * 1000

                if isinstance(result, HealthCheck):
                    result.latency_ms = latency
                    self._results[name] = result
                elif isinstance(result, bool):
                    self._results[name] = HealthCheck(
                        name=name,
                        status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                        latency_ms=latency,
                    )
                else:
                    self._results[name] = HealthCheck(
                        name=name, status=HealthStatus.HEALTHY,
                        message=str(result), latency_ms=latency,
                    )
            except Exception as e:
                self._results[name] = HealthCheck(
                    name=name, status=HealthStatus.CRITICAL,
                    message=str(e),
                )
                # Try recovery
                if name in self._recovery_actions:
                    try:
                        logger.info(f"Running recovery for {name}")
                        recovery = self._recovery_actions[name]
                        if asyncio.iscoroutinefunction(recovery):
                            await recovery()
                        else:
                            recovery()
                    except Exception as re:
                        logger.error(f"Recovery for {name} failed: {re}")

        # Check for degraded status
        critical = [r for r in self._results.values() if r.status == HealthStatus.CRITICAL]
        unhealthy = [r for r in self._results.values() if r.status == HealthStatus.UNHEALTHY]

        if critical:
            self._alert("critical", f"{len(critical)} critical health checks")

        return dict(self._results)

    def get_metrics(self) -> Metrics:
        """Collect current system metrics."""
        try:
            proc = psutil.Process(os.getpid())
            mem = proc.memory_info()
            cpu = proc.cpu_percent(interval=0.1)

            metrics = Metrics(
                cpu_percent=cpu,
                memory_mb=mem.rss / 1024 / 1024,
                memory_percent=proc.memory_percent(),
                disk_percent=psutil.disk_usage('/').percent,
                open_files=len(proc.open_files()),
                threads=proc.num_threads(),
                uptime=time.time() - self._start_time,
                request_count=self._request_count,
                error_count=self._error_count,
                avg_latency_ms=sum(self._latencies) / len(self._latencies) if self._latencies else 0,
            )
        except Exception:
            metrics = Metrics(
                uptime=time.time() - self._start_time,
                request_count=self._request_count,
                error_count=self._error_count,
            )

        self._metrics_history.append(metrics)
        if len(self._metrics_history) > 100:
            self._metrics_history = self._metrics_history[-100:]

        return metrics

    def overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self._results:
            return HealthStatus.HEALTHY

        statuses = [r.status for r in self._results.values()]
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def _alert(self, level: str, message: str):
        """Trigger alert."""
        for cb in self._alert_callbacks:
            try:
                cb(level, message)
            except Exception:
                pass

    async def start_monitoring(self):
        """Start periodic health checks."""
        self._running = True
        async def _monitor():
            while self._running:
                await self.run_checks()
                self.get_metrics()
                await asyncio.sleep(self.check_interval)
        self._task = asyncio.create_task(_monitor())

    def stop_monitoring(self):
        """Stop periodic health checks."""
        self._running = False
        if self._task:
            self._task.cancel()

    def report(self) -> Dict:
        """Generate health report."""
        return {
            "status": self.overall_status().value,
            "uptime": time.time() - self._start_time,
            "checks": {
                name: {
                    "status": r.status.value,
                    "message": r.message,
                    "latency_ms": r.latency_ms,
                }
                for name, r in self._results.items()
            },
            "metrics": {
                "requests": self._request_count,
                "errors": self._error_count,
                "avg_latency_ms": sum(self._latencies) / len(self._latencies) if self._latencies else 0,
            },
        }
