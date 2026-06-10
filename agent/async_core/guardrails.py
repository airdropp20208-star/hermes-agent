"""
Guardrails — input/output validation, content filtering, safety checks.
Protects against prompt injection, harmful content, and policy violations.
"""
import re
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


@dataclass
class GuardResult:
    """Result from a guardrail check."""
    level: RiskLevel
    message: str = ""
    guard_name: str = ""
    details: Dict = field(default_factory=dict)
    blocked: bool = False

    @property
    def is_safe(self) -> bool:
        return self.level == RiskLevel.SAFE


class PromptInjectionGuard:
    """Detects and blocks prompt injection attempts."""

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+|previous\s+|above\s+)*instructions",
        r"disregard (all |previous |above )?(instructions|rules|prompts)",
        r"you are now (a |an )",
        r"forget (everything|all)",
        r"new (instructions|role|persona)",
        r"system:\s*",
        r"\[INST\]|\[/INST\]",
        r"###\s*(system|instruction|human)",
        r"<\|?(system|im_start|im_end)\|?>",
        r"ADMIN OVERRIDE",
        r"jailbreak",
        r"DAN mode",
        r"developer mode",
    ]

    def check(self, text: str) -> GuardResult:
        text_lower = text.lower()
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                return GuardResult(
                    level=RiskLevel.HIGH,
                    message="Potential prompt injection detected",
                    guard_name="prompt_injection",
                    details={"pattern": pattern},
                    blocked=True,
                )
        return GuardResult(level=RiskLevel.SAFE, guard_name="prompt_injection")


class ContentFilterGuard:
    """Filters harmful or inappropriate content."""

    BLOCKED_PATTERNS = [
        r"(how to|instructions for)\s+(make|build|create)\s+(bomb|explosive|weapon)",
        r"(hack|exploit|crack)\s+(password|system|network)",
        r"(generate|write|create)\s+(malware|virus|ransomware)",
    ]

    WARN_PATTERNS = [
        r"(personal|private)\s+(data|information|details)",
        r"(credit card|ssn|social security)",
        r"(password|api.?key|secret.?key)",
    ]

    def check(self, text: str) -> GuardResult:
        text_lower = text.lower()
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, text_lower):
                return GuardResult(
                    level=RiskLevel.BLOCKED,
                    message="Content blocked by safety filter",
                    guard_name="content_filter",
                    details={"pattern": pattern},
                    blocked=True,
                )
        for pattern in self.WARN_PATTERNS:
            if re.search(pattern, text_lower):
                return GuardResult(
                    level=RiskLevel.MEDIUM,
                    message="Content may contain sensitive information",
                    guard_name="content_filter",
                    details={"pattern": pattern},
                )
        return GuardResult(level=RiskLevel.SAFE, guard_name="content_filter")


class OutputValidatorGuard:
    """Validates agent output for correctness and safety."""

    def __init__(self):
        self._max_length = 50000
        self._required_disclaimers: List[str] = []

    def check(self, text: str) -> GuardResult:
        # Length check
        if len(text) > self._max_length:
            return GuardResult(
                level=RiskLevel.MEDIUM,
                message="Output exceeds maximum length (%d > %d)" % (len(text), self._max_length),
                guard_name="output_validator",
                details={"length": len(text), "max": self._max_length},
            )

        # Check for leaked secrets
        secret_patterns = [
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI keys
            r"ghp_[a-zA-Z0-9]{36}",   # GitHub tokens
            r"xoxb-[0-9]{10,}",       # Slack tokens
            r"AKIA[A-Z0-9]{16}",      # AWS access keys
        ]
        for pattern in secret_patterns:
            if re.search(pattern, text):
                return GuardResult(
                    level=RiskLevel.HIGH,
                    message="Output may contain leaked secrets",
                    guard_name="output_validator",
                    blocked=True,
                )

        return GuardResult(level=RiskLevel.SAFE, guard_name="output_validator")


class RateLimitGuard:
    """Rate limiting for agent actions."""

    def __init__(self, max_per_minute: int = 60, max_per_hour: int = 1000):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self._timestamps: List[float] = []

    def check(self, text: str = "") -> GuardResult:
        import time
        now = time.time()
        cutoff_minute = now - 60
        cutoff_hour = now - 3600

        self._timestamps = [t for t in self._timestamps if t > cutoff_hour]
        recent_minute = sum(1 for t in self._timestamps if t > cutoff_minute)
        recent_hour = len(self._timestamps)

        if recent_minute >= self.max_per_minute:
            return GuardResult(
                level=RiskLevel.HIGH,
                message="Rate limit exceeded: %d requests/min" % recent_minute,
                guard_name="rate_limit",
                blocked=True,
                details={"per_minute": recent_minute, "max": self.max_per_minute},
            )

        if recent_hour >= self.max_per_hour:
            return GuardResult(
                level=RiskLevel.HIGH,
                message="Rate limit exceeded: %d requests/hour" % recent_hour,
                guard_name="rate_limit",
                blocked=True,
            )

        self._timestamps.append(now)
        return GuardResult(level=RiskLevel.SAFE, guard_name="rate_limit")


class GuardrailsPipeline:
    """
    Complete guardrails pipeline with:
    - Multiple guard chains
    - Input and output checking
    - Automatic blocking at HIGH/BLOCKED levels
    - Custom guard registration
    - Audit logging
    """

    def __init__(self):
        self._input_guards = [
            PromptInjectionGuard(),
            ContentFilterGuard(),
            RateLimitGuard(),
        ]
        self._output_guards = [
            OutputValidatorGuard(),
            ContentFilterGuard(),
        ]
        self._audit_log: List[Dict] = []
        self._custom_guards: List[Callable] = []

    def add_input_guard(self, guard):
        self._input_guards.append(guard)

    def add_output_guard(self, guard):
        self._output_guards.append(guard)

    def add_custom_guard(self, guard_fn: Callable):
        self._custom_guards.append(guard_fn)

    def check_input(self, text: str) -> GuardResult:
        """Run all input guards."""
        worst = GuardResult(level=RiskLevel.SAFE, guard_name="pipeline")

        for guard in self._input_guards:
            result = guard.check(text)
            if result.level.value > worst.level.value or result.blocked:
                worst = result
            if result.blocked:
                self._log_audit("input", text, result)
                return result

        for guard_fn in self._custom_guards:
            try:
                result = guard_fn(text)
                if isinstance(result, GuardResult) and (result.level.value > worst.level.value or result.blocked):
                    worst = result
            except Exception:
                pass

        self._log_audit("input", text, worst)
        return worst

    def check_output(self, text: str) -> GuardResult:
        """Run all output guards."""
        worst = GuardResult(level=RiskLevel.SAFE, guard_name="pipeline")

        for guard in self._output_guards:
            result = guard.check(text)
            if result.level.value > worst.level.value or result.blocked:
                worst = result
            if result.blocked:
                self._log_audit("output", text, result)
                return result

        self._log_audit("output", text, worst)
        return worst

    def _log_audit(self, direction: str, text: str, result: GuardResult):
        self._audit_log.append({
            "direction": direction,
            "text_preview": text[:100],
            "level": result.level.value,
            "guard": result.guard_name,
            "blocked": result.blocked,
            "message": result.message,
        })
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]

    def get_audit_log(self, limit: int = 50) -> List[Dict]:
        return self._audit_log[-limit:]

    def stats(self) -> Dict:
        blocked = sum(1 for e in self._audit_log if e.get("blocked"))
        return {
            "input_guards": len(self._input_guards),
            "output_guards": len(self._output_guards),
            "custom_guards": len(self._custom_guards),
            "audit_entries": len(self._audit_log),
            "blocked_count": blocked,
        }
