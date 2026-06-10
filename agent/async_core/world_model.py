"""
World Model — internal simulation of how things work.
Predict outcomes of actions, simulate alternatives, counterfactual reasoning.

This is the foundation of genuine planning and foresight.
"""
import time
import uuid
import json
import logging
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class WorldState:
    """A snapshot of the known world."""
    facts: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    confidence: Dict[str, float] = field(default_factory=dict)  # fact -> confidence

    def set(self, key: str, value: Any, confidence: float = 1.0):
        self.facts[key] = value
        self.confidence[key] = confidence

    def get(self, key: str, default=None):
        return self.facts.get(key, default)

    def merge(self, other: 'WorldState'):
        for k, v in other.facts.items():
            if k not in self.facts or other.confidence.get(k, 0) > self.confidence.get(k, 0):
                self.facts[k] = v
                self.confidence[k] = other.confidence.get(k, 0.5)


@dataclass
class Action:
    """An action that can be taken in the world."""
    name: str
    preconditions: Dict[str, Any] = field(default_factory=dict)
    effects: Dict[str, Any] = field(default_factory=dict)
    side_effects: List[str] = field(default_factory=list)
    cost: float = 1.0
    duration_estimate: float = 0
    reversible: bool = True
    risk: float = 0  # 0-1


@dataclass
class Prediction:
    """A predicted outcome of an action."""
    action_name: str
    predicted_state: WorldState
    confidence: float
    risks: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    alternative_outcomes: List[Tuple[WorldState, float]] = field(default_factory=list)


@dataclass
class Simulation:
    """Result of simulating a sequence of actions."""
    actions: List[str]
    initial_state: WorldState
    final_state: WorldState
    intermediate_states: List[WorldState] = field(default_factory=list)
    total_cost: float = 0
    total_risk: float = 0
    success_probability: float = 0
    warnings: List[str] = field(default_factory=list)


class WorldModel:
    """
    World model with:
    - State tracking (what is currently true)
    - Action modeling (what can be done, with what effects)
    - Outcome prediction (what will happen if I do X)
    - Multi-step simulation (simulate a whole plan)
    - Counterfactual reasoning (what if X hadn't happened?)
    - Causal understanding (X causes Y)
    - Risk assessment (what could go wrong?)
    - State rollback (undo predictions)
    """

    def __init__(self):
        self._state = WorldState()
        self._actions: Dict[str, Action] = {}
        self._causal_rules: List[Tuple[str, str, float]] = []  # (cause, effect, probability)
        self._state_history: List[WorldState] = []
        self._simulation_count = 0

    def update_state(self, key: str, value: Any, confidence: float = 1.0):
        """Update the world state."""
        self._state.set(key, value, confidence)

    def get_state(self, key: str = None):
        """Get world state."""
        if key:
            return self._state.get(key)
        return self._state

    def register_action(self, action: Action):
        """Register an action with its preconditions and effects."""
        self._actions[action.name] = action

    def add_causal_rule(self, cause: str, effect: str, probability: float = 0.8):
        """Add a causal rule: cause -> effect with probability."""
        self._causal_rules.append((cause, effect, probability))

    def predict(self, action_name: str, params: Dict = None) -> Prediction:
        """Predict the outcome of taking an action."""
        action = self._actions.get(action_name)
        if not action:
            return Prediction(
                action_name=action_name,
                predicted_state=WorldState(),
                confidence=0.1,
                risks=["Unknown action: %s" % action_name],
            )

        # Check preconditions
        missing = []
        for key, expected in action.preconditions.items():
            actual = self._state.get(key)
            if actual != expected:
                missing.append("%s: expected %s, got %s" % (key, expected, actual))

        # Predict new state
        predicted = WorldState()
        predicted.facts = dict(self._state.facts)
        predicted.confidence = dict(self._state.confidence)

        for key, value in action.effects.items():
            predicted.set(key, value, confidence=0.8)

        # Apply causal rules
        risks = []
        for cause, effect, prob in self._causal_rules:
            if cause in predicted.facts and predicted.facts[cause]:
                if prob < 0.7:
                    risks.append("May cause: %s (prob: %.0f%%)" % (effect, prob * 100))

        # Calculate confidence
        base_confidence = 0.8
        if missing:
            base_confidence *= 0.5  # Preconditions not met
        if action.risk > 0.3:
            base_confidence *= (1 - action.risk)

        return Prediction(
            action_name=action_name,
            predicted_state=predicted,
            confidence=base_confidence,
            risks=risks + ["%s: %s" % (r, action.name) for r in action.side_effects],
            side_effects=action.side_effects,
        )

    def simulate(self, action_sequence: List[str]) -> Simulation:
        """Simulate a sequence of actions and predict the final state."""
        self._simulation_count += 1
        current = WorldState()
        current.facts = dict(self._state.facts)
        current.confidence = dict(self._state.confidence)

        intermediates = []
        total_cost = 0
        total_risk = 0
        warnings = []
        success_prob = 1.0

        for action_name in action_sequence:
            action = self._actions.get(action_name)
            if not action:
                warnings.append("Unknown action: %s" % action_name)
                success_prob *= 0.3
                continue

            # Check preconditions in current simulated state
            preconditions_met = True
            for key, expected in action.preconditions.items():
                if current.get(key) != expected:
                    preconditions_met = False
                    warnings.append("Precondition failed for %s: %s" % (action_name, key))
                    break

            if not preconditions_met:
                success_prob *= 0.4
                # Still apply effects but with lower confidence
                for key, value in action.effects.items():
                    current.set(key, value, confidence=0.3)
            else:
                for key, value in action.effects.items():
                    current.set(key, value, confidence=0.8)

            total_cost += action.cost
            total_risk += action.risk * (1 - total_risk)  # Compound risk
            intermediates.append(WorldState(
                facts=dict(current.facts),
                confidence=dict(current.confidence),
            ))

        return Simulation(
            actions=action_sequence,
            initial_state=WorldState(
                facts=dict(self._state.facts),
                confidence=dict(self._state.confidence),
            ),
            final_state=current,
            intermediate_states=intermediates,
            total_cost=total_cost,
            total_risk=total_risk,
            success_probability=success_prob,
            warnings=warnings,
        )

    def counterfactual(self, fact_key: str, alternative_value: Any) -> WorldState:
        """What if a fact were different? Explore counterfactual."""
        original = self._state.get(fact_key)
        self._state.set(fact_key, alternative_value, confidence=0.5)

        # Propagate through causal rules
        cf_state = WorldState()
        cf_state.facts = dict(self._state.facts)
        cf_state.confidence = dict(self._state.confidence)

        for cause, effect, prob in self._causal_rules:
            if cause == fact_key:
                cf_state.set(effect, True, confidence=prob * 0.5)

        # Restore original
        if original is not None:
            self._state.set(fact_key, original, confidence=self._state.confidence.get(fact_key, 1.0))
        else:
            self._state.facts.pop(fact_key, None)

        return cf_state

    def save_state(self):
        """Save current state to history."""
        snapshot = WorldState(
            facts=dict(self._state.facts),
            confidence=dict(self._state.confidence),
        )
        self._state_history.append(snapshot)

    def rollback_state(self, steps: int = 1):
        """Rollback to a previous state."""
        if self._state_history:
            idx = max(0, len(self._state_history) - steps)
            self._state = self._state_history[idx]
            self._state_history = self._state_history[:idx]

    def to_context(self, max_tokens: int = 500) -> str:
        """Generate world model context for LLM."""
        parts = ["## World State"]
        for key, value in sorted(self._state.facts.items()):
            conf = self._state.confidence.get(key, 1.0)
            parts.append("- %s = %s (%.0f%%)" % (key, value, conf * 100))

        if self._actions:
            parts.append("\n## Available Actions")
            for name, action in self._actions.items():
                parts.append("- %s (cost: %.1f, risk: %.1f)" % (name, action.cost, action.risk))

        return "\n".join(parts[:50])  # Limit lines

    def stats(self) -> Dict:
        return {
            "state_facts": len(self._state.facts),
            "registered_actions": len(self._actions),
            "causal_rules": len(self._causal_rules),
            "simulations_run": self._simulation_count,
            "state_history_depth": len(self._state_history),
        }
