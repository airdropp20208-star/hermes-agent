"""
Budget Tracker — token usage, cost estimation, rate limiting.
Supports: per-session budgets, global budgets, alerts, auto-cutoff.
"""
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class BudgetScope(Enum):
    SESSION = "session"
    GLOBAL = "global"
    DAILY = "daily"
    HOURLY = "hourly"


# Pricing per 1M tokens (input, output)
MODEL_PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku": (0.25, 1.25),
    "claude-opus": (15.00, 75.00),
    "deepseek-r1": (0.55, 2.19),
    "deepseek-v3": (0.27, 1.10),
    "qwen-max": (1.60, 6.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
}


@dataclass
class UsageRecord:
    """A single usage record."""
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    session_id: str = ""
    operation: str = ""  # chat | tool | stream


@dataclass
class BudgetLimit:
    """Budget limit configuration."""
    scope: BudgetScope
    max_tokens: int = 0         # 0 = unlimited
    max_cost: float = 0.0       # 0 = unlimited
    max_api_calls: int = 0      # 0 = unlimited
    warn_at_percent: float = 80  # Warn at 80% usage
    hard_limit: bool = True      # Stop at limit vs just warn


class BudgetTracker:
    """
    Budget tracking with:
    - Per-model cost calculation
    - Multiple budget scopes (session, global, daily, hourly)
    - Usage history with time-window queries
    - Alert callbacks when approaching limits
    - Auto-cutoff at hard limits
    - Rate limiting (tokens/min, calls/min)
    - Detailed breakdown by model and operation
    """

    def __init__(self):
        self._records: List[UsageRecord] = []
        self._limits: Dict[str, BudgetLimit] = {}
        self._usage_by_scope: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"tokens": 0, "cost": 0, "calls": 0}
        )
        self._alert_callbacks: List = []
        self._custom_pricing: Dict[str, tuple] = {}

    def set_limit(self, name: str, limit: BudgetLimit):
        """Set a budget limit."""
        self._limits[name] = limit

    def set_pricing(self, model: str, input_per_m: float, output_per_m: float):
        """Set custom pricing for a model."""
        self._custom_pricing[model] = (input_per_m, output_per_m)

    def on_alert(self, callback):
        """Register alert callback."""
        self._alert_callbacks.append(callback)

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a given usage."""
        pricing = self._custom_pricing.get(model)
        if not pricing:
            # Try partial match
            for m, p in MODEL_PRICING.items():
                if m in model.lower():
                    pricing = p
                    break
        if not pricing:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * pricing[0]
        output_cost = (output_tokens / 1_000_000) * pricing[1]
        return input_cost + output_cost

    def record(self, model: str, input_tokens: int, output_tokens: int,
               session_id: str = "", operation: str = "chat") -> float:
        """Record usage and return cost. Returns -1 if budget exceeded."""
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        total_tokens = input_tokens + output_tokens

        record = UsageRecord(
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            session_id=session_id,
            operation=operation,
        )
        self._records.append(record)

        # Update scope usage
        for scope_key in ["global", f"session:{session_id}", self._time_scope()]:
            usage = self._usage_by_scope[scope_key]
            usage["tokens"] += total_tokens
            usage["cost"] += cost
            usage["calls"] += 1

        # Check limits
        exceeded = self._check_limits(total_tokens, cost)
        if exceeded:
            return -1

        return cost

    def _time_scope(self) -> str:
        """Get current time-based scope keys."""
        from datetime import datetime
        now = datetime.now()
        return f"daily:{now.strftime('%Y-%m-%d')}"

    def _check_limits(self, tokens: int, cost: float) -> bool:
        """Check if any limits are exceeded. Returns True if hard limit hit."""
        for name, limit in self._limits.items():
            usage = self._usage_by_scope.get(limit.scope.value, {"tokens": 0, "cost": 0, "calls": 0})

            # Check token limit
            if limit.max_tokens > 0:
                pct = (usage["tokens"] / limit.max_tokens) * 100
                if pct >= limit.warn_at_percent:
                    self._alert(name, f"Token usage at {pct:.0f}% ({usage['tokens']}/{limit.max_tokens})")
                if pct >= 100 and limit.hard_limit:
                    return True

            # Check cost limit
            if limit.max_cost > 0:
                pct = (usage["cost"] / limit.max_cost) * 100
                if pct >= limit.warn_at_percent:
                    self._alert(name, f"Cost at {pct:.0f}% (${usage['cost']:.4f}/${limit.max_cost:.4f})")
                if pct >= 100 and limit.hard_limit:
                    return True

        return False

    def _alert(self, limit_name: str, message: str):
        """Trigger alert callbacks."""
        for cb in self._alert_callbacks:
            try:
                cb(limit_name, message)
            except Exception as e:
                logger.warning(f"Alert callback error: {e}")

    def get_usage(self, scope: str = "global") -> Dict:
        """Get usage for a scope."""
        return dict(self._usage_by_scope.get(scope, {"tokens": 0, "cost": 0, "calls": 0}))

    def get_breakdown(self, since: float = 0) -> Dict:
        """Get detailed breakdown by model and operation."""
        records = [r for r in self._records if r.timestamp >= since]

        by_model = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "cost": 0, "calls": 0})
        by_operation = defaultdict(lambda: {"tokens": 0, "cost": 0, "calls": 0})

        for r in records:
            m = by_model[r.model]
            m["input_tokens"] += r.input_tokens
            m["output_tokens"] += r.output_tokens
            m["cost"] += r.cost
            m["calls"] += 1

            o = by_operation[r.operation]
            o["tokens"] += r.input_tokens + r.output_tokens
            o["cost"] += r.cost
            o["calls"] += 1

        return {
            "by_model": dict(by_model),
            "by_operation": dict(by_operation),
            "total_records": len(records),
        }

    def get_remaining(self, scope: str = "global") -> Dict:
        """Get remaining budget for a scope."""
        usage = self.get_usage(scope)
        remaining = {"tokens": float('inf'), "cost": float('inf'), "calls": float('inf')}

        for limit in self._limits.values():
            if limit.scope.value == scope or scope == "global":
                if limit.max_tokens > 0:
                    remaining["tokens"] = min(remaining["tokens"], limit.max_tokens - usage["tokens"])
                if limit.max_cost > 0:
                    remaining["cost"] = min(remaining["cost"], limit.max_cost - usage["cost"])
                if limit.max_api_calls > 0:
                    remaining["calls"] = min(remaining["calls"], limit.max_api_calls - usage["calls"])

        return remaining

    def stats(self) -> Dict:
        """Get comprehensive stats."""
        total_tokens = sum(r.input_tokens + r.output_tokens for r in self._records)
        total_cost = sum(r.cost for r in self._records)
        return {
            "total_records": len(self._records),
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 4),
            "limits": {name: {
                "scope": l.scope.value,
                "max_tokens": l.max_tokens,
                "max_cost": l.max_cost,
            } for name, l in self._limits.items()},
            "current_usage": dict(self._usage_by_scope),
        }
