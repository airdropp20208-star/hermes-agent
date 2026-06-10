"""
Curiosity Engine — intrinsic motivation, hypothesis generation, exploration.
Agent actively seeks to expand its knowledge, not just respond to requests.
This is what separates a reactive tool from a proactive intelligence.
"""
import time
import uuid
import json
import math
import logging
import random
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class CuriosityType(Enum):
    NOVELTY = "novelty"               # New information
    INCONSISTENCY = "inconsistency"    # Contradictions in knowledge
    GAP = "gap"                       # Missing information
    ANOMALY = "anomaly"              # Unexpected observations
    COMPLEXITY = "complexity"         # Unexplained complexity
    CONNECTION = "connection"         # Unexplored relationships


@dataclass
class Hypothesis:
    """A hypothesis generated from curiosity."""
    id: str
    statement: str
    curiosity_type: CuriosityType
    confidence: float = 0.3  # Start low
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    test_results: List[Dict] = field(default_factory=list)
    status: str = "untested"  # untested, testing, confirmed, refuted, uncertain
    created_at: float = field(default_factory=time.time)
    tested_at: float = 0
    importance: float = 0.5


@dataclass
class ExplorationGoal:
    """A goal driven by curiosity."""
    id: str
    description: str
    curiosity_type: CuriosityType
    priority: float
    knowledge_area: str
    status: str = "active"
    findings: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class KnowledgeFrontier:
    """The boundary between known and unknown."""
    known_topics: Dict[str, float] = field(default_factory=dict)  # topic -> mastery
    unknown_topics: List[str] = field(default_factory=list)
    partially_known: Dict[str, float] = field(default_factory=dict)  # topic -> partial mastery
    edge_cases: List[str] = field(default_factory=list)


class CuriosityEngine:
    """
    Curiosity engine with:
    - Novelty detection (what's new and interesting?)
    - Knowledge gap identification (what don't I know?)
    - Hypothesis generation (what might be true?)
    - Hypothesis testing (verify or refute)
    - Intrinsic motivation (explore even without explicit request)
    - Exploration-exploitation balance
    - Surprise detection (unexpected observations)
    - Knowledge frontier tracking
    - Question generation (what should I ask?)
    """

    def __init__(self):
        self._hypotheses: Dict[str, Hypothesis] = {}
        self._exploration_goals: List[ExplorationGoal] = []
        self._frontier = KnowledgeFrontier()
        self._observations: List[Dict] = []
        self._surprise_scores: List[float] = []
        self._exploration_rate = 0.3  # 0=exploit only, 1=explore only
        self._curiosity_decay = 0.95  # Curiosity about a topic decays over time
        self._topic_interest: Dict[str, float] = defaultdict(float)
        self._question_queue: List[str] = []

    def observe(self, observation: str, topic: str = "",
                expected: Any = None, actual: Any = None) -> float:
        """Process an observation and detect interesting patterns."""
        obs = {
            "content": observation,
            "topic": topic,
            "timestamp": time.time(),
            "expected": expected,
            "actual": actual,
        }
        self._observations.append(obs)

        # Calculate surprise
        surprise = self._calculate_surprise(expected, actual)
        self._surprise_scores.append(surprise)

        # Update topic interest
        if topic:
            self._topic_interest[topic] += surprise * 0.5

        # Generate hypotheses if surprising
        if surprise > 0.5:
            self._generate_hypotheses(obs, surprise)

        # Check for knowledge gaps
        if topic and topic not in self._frontier.known_topics:
            self._frontier.unknown_topics.append(topic)

        return surprise

    def _calculate_surprise(self, expected: Any, actual: Any) -> float:
        """Calculate how surprising an observation is."""
        if expected is None or actual is None:
            return 0.2

        if expected == actual:
            return 0.0  # Not surprising

        # Simple string comparison for now
        expected_str = str(expected)
        actual_str = str(actual)

        # Exact match = no surprise
        if expected_str == actual_str:
            return 0.0

        # Partial match = mild surprise
        common = sum(1 for a, b in zip(expected_str, actual_str) if a == b)
        max_len = max(len(expected_str), len(actual_str))
        similarity = common / max_len if max_len > 0 else 0

        return 1.0 - similarity  # Less similar = more surprising

    def _generate_hypotheses(self, observation: Dict, surprise: float):
        """Generate hypotheses to explain surprising observations."""
        # Anomaly hypothesis
        h = Hypothesis(
            id="h_" + str(uuid.uuid4())[:8],
            statement="Unexpected observation: %s" % observation["content"][:200],
            curiosity_type=CuriosityType.ANOMALY,
            confidence=0.3,
            importance=surprise,
        )
        self._hypotheses[h.id] = h

        # Gap hypothesis
        if observation.get("topic"):
            h2 = Hypothesis(
                id="h_" + str(uuid.uuid4())[:8],
                statement="Knowledge gap in: %s" % observation["topic"],
                curiosity_type=CuriosityType.GAP,
                confidence=0.5,
                importance=0.6,
            )
            self._hypotheses[h2.id] = h2

    def generate_question(self) -> Optional[str]:
        """Generate a curiosity-driven question."""
        candidates = []

        # Questions about knowledge gaps
        for topic in self._frontier.unknown_topics[:5]:
            candidates.append(("What is %s and how does it work?" % topic, 0.7))

        # Questions about untested hypotheses
        for h in self._hypotheses.values():
            if h.status == "untested":
                candidates.append(("Is it true that: %s?" % h.statement[:100], h.importance))

        # Questions about surprising observations
        recent_surprises = [(s, o) for s, o in zip(
            self._surprise_scores[-10:], self._observations[-10:])]
        for score, obs in recent_surprises:
            if score > 0.5:
                candidates.append(
                    ("Why did this happen: %s?" % obs["content"][:100], score))

        # Questions about edge cases
        for edge in self._frontier.edge_cases[:3]:
            candidates.append(("What happens when: %s?" % edge, 0.5))

        if not candidates:
            return None

        # Sort by importance
        candidates.sort(key=lambda x: x[1], reverse=True)
        question = candidates[0][0]
        self._question_queue.append(question)
        return question

    def create_exploration_goal(self, topic: str,
                                 curiosity_type: CuriosityType = CuriosityType.GAP) -> ExplorationGoal:
        """Create a goal to explore a topic."""
        goal = ExplorationGoal(
            id="eg_" + str(uuid.uuid4())[:8],
            description="Explore: %s" % topic,
            curiosity_type=curiosity_type,
            priority=self._topic_interest.get(topic, 0.5),
            knowledge_area=topic,
        )
        self._exploration_goals.append(goal)
        return goal

    def record_finding(self, goal_id: str, finding: str):
        """Record a finding from exploration."""
        for goal in self._exploration_goals:
            if goal.id == goal_id:
                goal.findings.append(finding)
                break

        # Update knowledge frontier
        for goal in self._exploration_goals:
            if goal.id == goal_id and goal.knowledge_area:
                topic = goal.knowledge_area
                current = self._frontier.known_topics.get(topic, 0)
                self._frontier.known_topics[topic] = min(1.0, current + 0.1)
                if topic in self._frontier.unknown_topics:
                    self._frontier.unknown_topics.remove(topic)

    def update_hypothesis(self, hypothesis_id: str, evidence: str,
                          supports: bool):
        """Update hypothesis with new evidence."""
        h = self._hypotheses.get(hypothesis_id)
        if not h:
            return

        if supports:
            h.evidence_for.append(evidence)
            h.confidence = min(0.95, h.confidence + 0.1)
        else:
            h.evidence_against.append(evidence)
            h.confidence = max(0.05, h.confidence - 0.15)

        # Determine status
        if h.confidence > 0.8:
            h.status = "confirmed"
        elif h.confidence < 0.2:
            h.status = "refuted"
        elif len(h.evidence_for) + len(h.evidence_against) > 5:
            h.status = "uncertain"

    def should_explore(self) -> bool:
        """Decide whether to explore or exploit."""
        # Exploration increases with: more unknowns, more surprises, fewer recent explorations
        unknown_ratio = len(self._frontier.unknown_topics) / max(1,
            len(self._frontier.known_topics) + len(self._frontier.unknown_topics))

        avg_surprise = (sum(self._surprise_scores[-10:]) / 10
                       if self._surprise_scores else 0.2)

        exploration_need = unknown_ratio * 0.4 + avg_surprise * 0.3 + self._exploration_rate * 0.3

        return random.random() < exploration_need

    def get_exploration_topics(self, max_topics: int = 5) -> List[Tuple[str, float]]:
        """Get topics worth exploring, ranked by curiosity."""
        topics = []

        # Unknown topics
        for topic in self._frontier.unknown_topics:
            interest = self._topic_interest.get(topic, 0.3)
            topics.append((topic, interest))

        # Partially known topics with low mastery
        for topic, mastery in self._frontier.partially_known.items():
            if mastery < 0.5:
                topics.append((topic, 0.5 + (1 - mastery) * 0.3))

        # Topics from surprising observations
        for score, obs in zip(self._surprise_scores[-5:], self._observations[-5:]):
            if score > 0.5 and obs.get("topic"):
                topics.append((obs["topic"], score))

        # Deduplicate and sort
        seen = set()
        unique = []
        for topic, score in topics:
            if topic not in seen:
                seen.add(topic)
                unique.append((topic, score))

        unique.sort(key=lambda x: x[1], reverse=True)
        return unique[:max_topics]

    def get_active_hypotheses(self) -> List[Hypothesis]:
        """Get hypotheses that need testing."""
        return [h for h in self._hypotheses.values()
                if h.status in ("untested", "testing")]

    def to_context(self, max_tokens: int = 500) -> str:
        """Generate curiosity context for LLM."""
        parts = ["## Curiosity State"]

        # Knowledge frontier
        known = len(self._frontier.known_topics)
        unknown = len(self._frontier.unknown_topics)
        parts.append("Known: %d topics, Unknown: %d topics" % (known, unknown))

        # Active hypotheses
        active = self.get_active_hypotheses()
        if active:
            parts.append("Active hypotheses:")
            for h in active[:3]:
                parts.append("- [%s] %s (%.0f%%)" % (
                    h.curiosity_type.value, h.statement[:80], h.confidence * 100))

        # Exploration questions
        questions = self._question_queue[-3:]
        if questions:
            parts.append("Questions:")
            for q in questions:
                parts.append("- %s" % q)

        # Top exploration topics
        topics = self.get_exploration_topics(3)
        if topics:
            parts.append("Worth exploring:")
            for topic, score in topics:
                parts.append("- %s (interest: %.0f%%)" % (topic, score * 100))

        return "\n".join(parts)

    def stats(self) -> Dict:
        return {
            "hypotheses": len(self._hypotheses),
            "confirmed": sum(1 for h in self._hypotheses.values() if h.status == "confirmed"),
            "refuted": sum(1 for h in self._hypotheses.values() if h.status == "refuted"),
            "untested": sum(1 for h in self._hypotheses.values() if h.status == "untested"),
            "exploration_goals": len(self._exploration_goals),
            "known_topics": len(self._frontier.known_topics),
            "unknown_topics": len(self._frontier.unknown_topics),
            "observations": len(self._observations),
            "avg_surprise": sum(self._surprise_scores[-20:]) / max(1, len(self._surprise_scores[-20:])),
            "exploration_rate": self._exploration_rate,
        }
