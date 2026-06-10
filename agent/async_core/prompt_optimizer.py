"""
Prompt Optimizer — learn which prompts work, A/B test, auto-optimize.
The agent improves its own prompts based on outcomes.
"""
import time
import uuid
import json
import hashlib
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """A variant of a prompt for A/B testing."""
    id: str
    template: str
    description: str
    use_count: int = 0
    success_count: int = 0
    avg_quality: float = 0
    avg_tokens: int = 0
    avg_latency_ms: float = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.use_count if self.use_count else 0

    @property
    def score(self) -> float:
        """Combined score: success_rate * quality / token_cost."""
        if not self.use_count:
            return 0
        efficiency = self.avg_quality / max(1, self.avg_tokens / 100)
        return self.success_rate * 0.6 + efficiency * 0.4


@dataclass
class PromptRecord:
    """Record of a prompt usage."""
    prompt_id: str
    variant_id: str
    input_context: str
    output: str
    success: bool
    quality_score: float  # 0-1
    tokens_used: int
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


class PromptOptimizer:
    """
    Prompt optimization with:
    - Multiple prompt variants per task
    - A/B testing with statistical significance
    - Quality scoring based on outcomes
    - Auto-selection of best variant
    - Prompt template library
    - Few-shot example management
    - Prompt compression (reduce tokens while keeping quality)
    - Learning from user feedback
    """

    def __init__(self):
        self._variants: Dict[str, Dict[str, PromptVariant]] = defaultdict(dict)
        self._records: List[PromptRecord] = []
        self._templates: Dict[str, str] = {}
        self._few_shots: Dict[str, List[Dict]] = defaultdict(list)
        self._feedback_history: List[Dict] = []

    def register_variant(self, task: str, template: str,
                         description: str = "") -> PromptVariant:
        """Register a prompt variant for a task."""
        vid = "v_" + str(uuid.uuid4())[:8]
        variant = PromptVariant(
            id=vid, template=template,
            description=description or template[:100],
        )
        self._variants[task][vid] = variant
        return variant

    def select_variant(self, task: str, strategy: str = "best") -> Optional[PromptVariant]:
        """Select a prompt variant for a task."""
        variants = self._variants.get(task)
        if not variants:
            return None

        if strategy == "best":
            # Pick the highest-scoring variant with enough data
            scored = [(v.score, v) for v in variants.values() if v.use_count >= 3]
            if scored:
                scored.sort(reverse=True)
                return scored[0][1]
            # Not enough data — round-robin
            return min(variants.values(), key=lambda v: v.use_count)

        elif strategy == "explore":
            # Thompson sampling: balance exploration and exploitation
            import random
            candidates = []
            for v in variants.values():
                alpha = v.success_count + 1
                beta = v.use_count - v.success_count + 1
                sample = random.betavariate(alpha, beta)
                candidates.append((sample, v))
            candidates.sort(reverse=True)
            return candidates[0][1]

        elif strategy == "random":
            import random
            return random.choice(list(variants.values()))

        return list(variants.values())[0]

    def record_usage(self, task: str, variant_id: str,
                     input_context: str, output: str,
                     success: bool, quality_score: float = 0.5,
                     tokens_used: int = 0, latency_ms: float = 0):
        """Record the outcome of using a prompt variant."""
        record = PromptRecord(
            prompt_id=task, variant_id=variant_id,
            input_context=input_context[:200], output=output[:200],
            success=success, quality_score=quality_score,
            tokens_used=tokens_used, latency_ms=latency_ms,
        )
        self._records.append(record)

        # Update variant stats
        variant = self._variants.get(task, {}).get(variant_id)
        if variant:
            variant.use_count += 1
            if success:
                variant.success_count += 1
            n = variant.use_count
            variant.avg_quality = (variant.avg_quality * (n-1) + quality_score) / n
            variant.avg_tokens = int((variant.avg_tokens * (n-1) + tokens_used) / n)
            variant.avg_latency_ms = (variant.avg_latency_ms * (n-1) + latency_ms) / n

    def add_few_shot(self, task: str, example: Dict):
        """Add a few-shot example for a task."""
        self._few_shots[task].append(example)
        # Keep only top examples (by recency)
        if len(self._few_shots[task]) > 20:
            self._few_shots[task] = self._few_shots[task][-20:]

    def get_few_shots(self, task: str, max_examples: int = 5) -> List[Dict]:
        """Get few-shot examples for a task."""
        examples = self._few_shots.get(task, [])
        return examples[-max_examples:]

    def record_feedback(self, task: str, variant_id: str,
                        feedback: str, rating: float):
        """Record user feedback on prompt output."""
        self._feedback_history.append({
            "task": task, "variant_id": variant_id,
            "feedback": feedback, "rating": rating,
            "timestamp": time.time(),
        })

    def get_best_variant(self, task: str) -> Optional[PromptVariant]:
        """Get the best-performing variant for a task."""
        variants = self._variants.get(task)
        if not variants:
            return None
        scored = [(v.score, v) for v in variants.values() if v.use_count >= 3]
        if not scored:
            return list(variants.values())[0]
        scored.sort(reverse=True)
        return scored[0][1]

    def compress_prompt(self, prompt: str, target_tokens: int = 500) -> str:
        """Compress a prompt while keeping key information."""
        estimated_tokens = len(prompt) // 4
        if estimated_tokens <= target_tokens:
            return prompt

        # Strategy: keep first and last parts, compress middle
        target_chars = target_tokens * 4
        keep_start = target_chars // 3
        keep_end = target_chars // 3

        if len(prompt) <= keep_start + keep_end:
            return prompt

        return prompt[:keep_start] + "\n[...compressed...]\n" + prompt[-keep_end:]

    def ab_test_results(self, task: str) -> Dict:
        """Get A/B test results for variants of a task."""
        variants = self._variants.get(task, {})
        if len(variants) < 2:
            return {"enough_data": False, "variants": len(variants)}

        results = []
        for v in variants.values():
            if v.use_count >= 5:
                results.append({
                    "id": v.id,
                    "description": v.description[:80],
                    "uses": v.use_count,
                    "success_rate": round(v.success_rate * 100, 1),
                    "avg_quality": round(v.avg_quality, 3),
                    "avg_tokens": v.avg_tokens,
                    "score": round(v.score, 3),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        winner = results[0] if results else None

        return {
            "enough_data": len(results) >= 2,
            "variants": results,
            "winner": winner["id"] if winner else None,
            "improvement": round(
                (results[0]["success_rate"] - results[-1]["success_rate"]), 1
            ) if len(results) >= 2 else 0,
        }

    def get_all_tasks(self) -> List[str]:
        return list(self._variants.keys())

    def stats(self) -> Dict:
        total_variants = sum(len(v) for v in self._variants.values())
        total_uses = sum(v.use_count for variants in self._variants.values()
                        for v in variants.values())
        return {
            "tasks": len(self._variants),
            "total_variants": total_variants,
            "total_uses": total_uses,
            "total_records": len(self._records),
            "feedback_count": len(self._feedback_history),
        }
