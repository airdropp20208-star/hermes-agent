"""
Metacognition Engine — the agent thinks about its own thinking.
Self-awareness of reasoning quality, confidence calibration, bias detection,
strategy self-assessment, and cognitive load management.

This is the closest thing to genuine AI self-awareness in any framework.
"""
import time
import uuid
import json
import math
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    DEDUCTIVE = "deductive"       # From general to specific
    INDUCTIVE = "inductive"       # From specific to general
    ABDUCTIVE = "abductive"       # Best explanation
    ANALOGICAL = "analogical"     # Similarity-based
    CAUSAL = "causal"             # Cause-effect
    COUNTERFACTUAL = "counterfactual"  # What-if
    METACOGNITIVE = "metacognitive"    # Thinking about thinking


class CognitiveBias(Enum):
    CONFIRMATION = "confirmation_bias"       # Seek confirming evidence
    ANCHORING = "anchoring_bias"             # Over-rely on first info
    AVAILABILITY = "availability_bias"       # Over-weight recent info
    OVERCONFIDENCE = "overconfidence_bias"   # Overestimate accuracy
    FRAMING = "framing_bias"                 # Influenced by presentation
    SUNK_COST = "sunk_cost_bias"             # Continue because invested
    DUNNING_KRUGER = "dunning_kruger"        # Overestimate ability
    SURVIVORSHIP = "survivorship_bias"       # Only see successes


@dataclass
class ReasoningTrace:
    """A trace of the agent's reasoning process."""
    id: str
    step: str
    reasoning_type: ReasoningType
    input_data: str
    output_data: str
    confidence: float  # 0-1
    assumptions: List[str] = field(default_factory=list)
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    alternatives_considered: List[str] = field(default_factory=list)
    biases_detected: List[CognitiveBias] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0


@dataclass
class CognitiveState:
    """Current cognitive state of the agent."""
    confidence: float = 0.5
    cognitive_load: float = 0.0  # 0-1, how much mental effort
    uncertainty_areas: List[str] = field(default_factory=list)
    active_biases: List[CognitiveBias] = field(default_factory=list)
    reasoning_quality: float = 0.5  # assessed quality of recent reasoning
    knowledge_gaps: List[str] = field(default_factory=list)
    current_strategy: str = ""
    exploration_level: float = 0.5  # 0=exploit, 1=explore


class ConfidenceCalibrator:
    """
    Calibrates confidence based on historical accuracy.
    If agent says 80% confident, it should be right ~80% of the time.
    """

    def __init__(self):
        self._predictions: List[Tuple[float, bool]] = []  # (stated_confidence, was_correct)
        self._calibration_curve: Dict[str, float] = {}  # bucket -> actual_accuracy

    def record(self, stated_confidence: float, was_correct: bool):
        """Record a prediction outcome."""
        self._predictions.append((stated_confidence, was_correct))
        self._update_calibration()

    def _update_calibration(self):
        """Update calibration curve."""
        buckets = defaultdict(list)
        for conf, correct in self._predictions:
            bucket = "%.1f" % (round(conf * 10) / 10)
            buckets[bucket].append(correct)

        for bucket, outcomes in buckets.items():
            self._calibration_curve[bucket] = sum(outcomes) / len(outcomes)

    def calibrate(self, raw_confidence: float, context: str = "") -> float:
        """Adjust raw confidence based on calibration history."""
        if len(self._predictions) < 10:
            return raw_confidence  # Not enough data

        bucket = "%.1f" % (round(raw_confidence * 10) / 10)
        actual = self._calibration_curve.get(bucket)
        if actual is None:
            return raw_confidence

        # Blend stated confidence with calibration
        calibrated = raw_confidence * 0.4 + actual * 0.6

        # Apply overconfidence correction
        if raw_confidence > 0.8 and actual < 0.6:
            calibrated = raw_confidence * 0.7  # Significant correction

        return max(0.01, min(0.99, calibrated))

    def get_overconfidence_score(self) -> float:
        """How overconfident is the agent? 0=well-calibrated, 1=very overconfident."""
        if len(self._predictions) < 5:
            return 0
        diffs = []
        for conf, correct in self._predictions[-50:]:
            diffs.append(conf - (1.0 if correct else 0.0))
        avg_diff = sum(diffs) / len(diffs)
        return max(0, avg_diff)  # Positive = overconfident

    def stats(self) -> Dict:
        if not self._predictions:
            return {"predictions": 0}
        total = len(self._predictions)
        correct = sum(1 for _, c in self._predictions if c)
        return {
            "predictions": total,
            "accuracy": correct / total,
            "overconfidence": self.get_overconfidence_score(),
            "calibration_curve": dict(self._calibration_curve),
        }


class BiasDetector:
    """Detects cognitive biases in the agent's reasoning."""

    def __init__(self):
        self._detected_biases: List[Tuple[CognitiveBias, str, float]] = []

    def analyze(self, reasoning: ReasoningTrace) -> List[CognitiveBias]:
        """Analyze a reasoning trace for biases."""
        detected = []

        # Confirmation bias: only evidence_for, no evidence_against
        if reasoning.evidence_for and not reasoning.evidence_against:
            if len(reasoning.evidence_for) > 2:
                detected.append(CognitiveBias.CONFIRMATION)

        # Overconfidence: high confidence with few alternatives
        if reasoning.confidence > 0.9 and len(reasoning.alternatives_considered) <= 1:
            detected.append(CognitiveBias.OVERCONFIDENCE)

        # Anchoring: first alternative dominates
        if (reasoning.alternatives_considered and
            reasoning.output_data and
            reasoning.alternatives_considered[0][:20] in reasoning.output_data[:50]):
            detected.append(CognitiveBias.ANCHORING)

        # Sunk cost: many assumptions suggest commitment to initial approach
        if len(reasoning.assumptions) > 5:
            detected.append(CognitiveBias.SUNK_COST)

        for bias in detected:
            self._detected_biases.append(
                (bias, reasoning.step[:100], reasoning.confidence))

        return detected

    def get_bias_report(self) -> Dict[str, int]:
        """Get summary of detected biases."""
        counts = defaultdict(int)
        for bias, _, _ in self._detected_biases:
            counts[bias.value] += 1
        return dict(counts)


class MetacognitionEngine:
    """
    Full metacognition system with:
    - Reasoning traces (what was thought, why, with what confidence)
    - Confidence calibration (learn from prediction accuracy)
    - Bias detection (catch confirmation, overconfidence, anchoring)
    - Strategy self-assessment (is my approach working?)
    - Cognitive load management (don't overload)
    - Knowledge gap identification (what don't I know?)
    - Uncertainty quantification (how sure am I really?)
    - Self-questioning prompts (am I reasoning well?)
    """

    def __init__(self):
        self.calibrator = ConfidenceCalibrator()
        self.bias_detector = BiasDetector()
        self._traces: List[ReasoningTrace] = []
        self._state = CognitiveState()
        self._strategy_outcomes: Dict[str, List[float]] = defaultdict(list)
        self._self_questions: List[str] = []

    def start_reasoning(self, step: str, reasoning_type: ReasoningType,
                        input_data: str) -> ReasoningTrace:
        """Begin a reasoning step."""
        trace = ReasoningTrace(
            id="r_" + str(uuid.uuid4())[:8],
            step=step,
            reasoning_type=reasoning_type,
            input_data=input_data,
            output_data="",
            confidence=0.5,
        )
        self._traces.append(trace)
        return trace

    def finish_reasoning(self, trace: ReasoningTrace, output: str,
                         confidence: float, assumptions: List[str] = None,
                         evidence_for: List[str] = None,
                         evidence_against: List[str] = None,
                         alternatives: List[str] = None) -> ReasoningTrace:
        """Complete a reasoning step with full metacognitive assessment."""
        trace.output_data = output
        trace.assumptions = assumptions or []
        trace.evidence_for = evidence_for or []
        trace.evidence_against = evidence_against or []
        trace.alternatives_considered = alternatives or []

        # Calibrate confidence
        trace.confidence = self.calibrator.calibrate(confidence, trace.step)

        # Detect biases
        trace.biases_detected = self.bias_detector.analyze(trace)

        # Update cognitive state
        self._update_state(trace)

        return trace

    def self_assess(self) -> Dict[str, Any]:
        """Agent assesses its own reasoning quality."""
        recent = self._traces[-20:] if self._traces else []
        if not recent:
            return {"quality": 0.5, "assessment": "No reasoning history"}

        avg_confidence = sum(t.confidence for t in recent) / len(recent)
        bias_count = sum(len(t.biases_detected) for t in recent)
        evidence_balance = []
        for t in recent:
            if t.evidence_for or t.evidence_against:
                balance = len(t.evidence_for) / max(1, len(t.evidence_for) + len(t.evidence_against))
                evidence_balance.append(balance)

        avg_balance = sum(evidence_balance) / len(evidence_balance) if evidence_balance else 0.5

        # Quality score
        quality = (
            avg_confidence * 0.3 +
            (1 - min(bias_count / 20, 1)) * 0.3 +
            avg_balance * 0.2 +
            (1 - self.calibrator.get_overconfidence_score()) * 0.2
        )

        assessment = []
        if self.calibrator.get_overconfidence_score() > 0.3:
            assessment.append("I notice I've been overconfident. Need more verification.")
        if bias_count > 5:
            assessment.append("Several cognitive biases detected in recent reasoning.")
        if avg_balance < 0.3:
            assessment.append("Not considering enough counter-evidence.")
        if not assessment:
            assessment.append("Reasoning quality appears adequate.")

        return {
            "quality": quality,
            "confidence": avg_confidence,
            "biases_detected": bias_count,
            "evidence_balance": avg_balance,
            "overconfidence": self.calibrator.get_overconfidence_score(),
            "assessment": " ".join(assessment),
            "recommendations": self._generate_recommendations(quality),
        }

    def _generate_recommendations(self, quality: float) -> List[str]:
        """Generate metacognitive recommendations."""
        recs = []
        if quality < 0.4:
            recs.append("Slow down. Consider more alternatives before deciding.")
        if self.calibrator.get_overconfidence_score() > 0.3:
            recs.append("Actively seek disconfirming evidence.")
        if len(self._state.knowledge_gaps) > 3:
            recs.append("Fill knowledge gaps before proceeding: %s" %
                       ", ".join(self._state.knowledge_gaps[:3]))
        if self._state.cognitive_load > 0.8:
            recs.append("Cognitive load is high. Break task into smaller pieces.")
        if self._state.exploration_level < 0.2:
            recs.append("Stuck in exploitation mode. Try a different approach.")
        return recs

    def _update_state(self, trace: ReasoningTrace):
        """Update cognitive state based on reasoning trace."""
        # Update confidence (moving average)
        self._state.confidence = self._state.confidence * 0.7 + trace.confidence * 0.3

        # Track biases
        self._state.active_biases = list(set(
            self._state.active_biases + trace.biases_detected
        ))[-10:]  # Keep last 10

        # Update knowledge gaps
        for gap in trace.assumptions:
            if gap not in self._state.knowledge_gaps:
                self._state.knowledge_gaps.append(gap)
        self._state.knowledge_gaps = self._state.knowledge_gaps[-20:]

        # Cognitive load
        self._state.cognitive_load = min(1.0,
            len(trace.alternatives_considered) * 0.1 +
            len(trace.assumptions) * 0.05 +
            (1 - trace.confidence) * 0.3
        )

    def generate_self_question(self) -> str:
        """Generate a self-questioning prompt for the agent."""
        questions = []

        if self._state.confidence > 0.8:
            questions.append("Am I sure? What evidence contradicts my conclusion?")
        if self._state.active_biases:
            questions.append("Could %s bias be affecting my reasoning?" %
                           self._state.active_biases[0].value)
        if len(self._state.knowledge_gaps) > 0:
            questions.append("What do I not know that could change my answer?")
        if self._state.cognitive_load > 0.7:
            questions.append("Am I trying to do too much at once?")
        if not questions:
            questions.append("What assumption am I making that might be wrong?")

        question = questions[0]
        self._self_questions.append(question)
        return question

    def record_outcome(self, reasoning_id: str, was_correct: bool):
        """Record whether a reasoning outcome was correct."""
        for trace in self._traces:
            if trace.id == reasoning_id:
                self.calibrator.record(trace.confidence, was_correct)
                break

    def get_state(self) -> CognitiveState:
        return self._state

    def to_context(self, max_tokens: int = 500) -> str:
        """Generate metacognitive context for LLM prompt."""
        assessment = self.self_assess()
        parts = [
            "## Metacognitive Assessment",
            "Confidence: %.0f%%" % (assessment["confidence"] * 100),
            "Reasoning quality: %.0f%%" % (assessment["quality"] * 100),
            "Assessment: %s" % assessment["assessment"],
        ]
        if assessment["recommendations"]:
            parts.append("Recommendations:")
            for r in assessment["recommendations"]:
                parts.append("- %s" % r)

        question = self.generate_self_question()
        parts.append("Self-question: %s" % question)

        return "\n".join(parts)

    def stats(self) -> Dict:
        return {
            "traces": len(self._traces),
            "calibration": self.calibrator.stats(),
            "bias_report": self.bias_detector.get_bias_report(),
            "cognitive_state": {
                "confidence": self._state.confidence,
                "cognitive_load": self._state.cognitive_load,
                "active_biases": len(self._state.active_biases),
                "knowledge_gaps": len(self._state.knowledge_gaps),
            },
        }
