"""
Theory of Mind — model user's mental state, intentions, knowledge, preferences.
Agent can anticipate needs, adapt communication style, detect confusion.
This is what makes an agent feel truly intelligent and empathetic.
"""
import time
import uuid
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class UserBelief:
    """What we believe about the user's knowledge state."""
    topic: str
    knowledge_level: float  # 0=ignorant, 1=expert
    confidence: float  # how sure we are about this assessment
    evidence: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


@dataclass
class UserGoal:
    """Inferred or stated user goal."""
    description: str
    priority: float  # 0-1
    explicit: bool  # did user state this directly?
    progress: float = 0  # 0-1
    sub_goals: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class UserEmotionalState:
    """Inferred emotional state."""
    frustration: float = 0  # 0-1
    urgency: float = 0
    satisfaction: float = 0.5
    confusion: float = 0
    engagement: float = 0.5
    last_signal: str = ""
    last_updated: float = field(default_factory=time.time)


@dataclass
class CommunicationStyle:
    """User's preferred communication style."""
    verbosity: float = 0.5  # 0=terse, 1=verbose
    technical_level: float = 0.5  # 0=layman, 1=expert
    formality: float = 0.3  # 0=casual, 1=formal
    pace: float = 0.5  # 0=slow/careful, 1=fast/efficient
    prefers_examples: bool = True
    prefers_visual: bool = False
    language: str = "en"
    humor_tolerance: float = 0.5


class TheoryOfMind:
    """
    Theory of Mind system with:
    - User knowledge modeling (what do they know?)
    - Goal inference (what do they want?)
    - Emotional state tracking (how do they feel?)
    - Communication adaptation (how should I respond?)
    - Confusion detection (are they lost?)
    - Expertise estimation (how technical should I be?)
    - Need anticipation (what will they need next?)
    - Frustration detection and de-escalation
    """

    def __init__(self):
        self._beliefs: Dict[str, UserBelief] = {}
        self._goals: List[UserGoal] = []
        self._emotion = UserEmotionalState()
        self._style = CommunicationStyle()
        self._interaction_history: List[Dict] = []
        self._pattern_counts: Dict[str, int] = defaultdict(int)
        self._session_start = time.time()
        self._turn_count = 0

    def observe_user_message(self, message: str) -> Dict[str, Any]:
        """Analyze a user message to update mental model."""
        self._turn_count += 1
        observation = {
            "turn": self._turn_count,
            "length": len(message),
            "signals": [],
        }

        # Detect emotional signals
        self._detect_emotion(message, observation)

        # Detect knowledge signals
        self._detect_knowledge(message, observation)

        # Detect goal signals
        self._detect_goals(message, observation)

        # Update communication style
        self._update_style(message, observation)

        self._interaction_history.append(observation)
        if len(self._interaction_history) > 100:
            self._interaction_history = self._interaction_history[-100:]

        return observation

    def _detect_emotion(self, message: str, observation: Dict):
        """Detect emotional signals from message."""
        msg_lower = message.lower()
        signals = []

        # Frustration signals
        frustration_words = [
            "không được", "lỗi", "sai", "fail", "broken", "shit",
            "tại sao", "why", "ko hiểu", "frustrated", "annoying",
            "lại nữa", "again", "vẫn", "still not",
        ]
        frustration_score = sum(1 for w in frustration_words if w in msg_lower)
        if frustration_score > 0:
            self._emotion.frustration = min(1.0,
                self._emotion.frustration * 0.7 + frustration_score * 0.15)
            signals.append("frustration_detected")

        # Urgency signals
        urgency_words = ["nhanh", "gấp", "urgent", "asap", "ngay", "now", "immediately"]
        if any(w in msg_lower for w in urgency_words):
            self._emotion.urgency = min(1.0, self._emotion.urgency + 0.2)
            signals.append("urgency_detected")

        # Satisfaction signals
        satisfaction_words = ["tốt", "good", "great", "perfect", "ok", "được", "hay"]
        if any(w in msg_lower for w in satisfaction_words):
            self._emotion.satisfaction = min(1.0, self._emotion.satisfaction + 0.15)
            self._emotion.frustration = max(0, self._emotion.frustration - 0.1)
            signals.append("satisfaction_detected")

        # Confusion signals
        confusion_words = ["không hiểu", "confused", "what", "huh", "sao", "tại sao", "?"]
        confusion_count = sum(1 for w in confusion_words if w in msg_lower)
        if confusion_count >= 2:
            self._emotion.confusion = min(1.0, confusion_count * 0.2)
            signals.append("confusion_detected")

        # Very short messages might indicate disengagement
        if len(message) < 10 and self._turn_count > 3:
            self._emotion.engagement = max(0, self._emotion.engagement - 0.1)
            signals.append("low_engagement")

        # Very long messages indicate high engagement
        if len(message) > 200:
            self._emotion.engagement = min(1.0, self._emotion.engagement + 0.1)
            signals.append("high_engagement")

        self._emotion.last_signal = signals[0] if signals else ""
        self._emotion.last_updated = time.time()
        observation["signals"].extend(signals)

    def _detect_knowledge(self, message: str, observation: Dict):
        """Detect knowledge level signals."""
        msg_lower = message.lower()

        # Technical language indicates expertise
        tech_terms = [
            "api", "async", "endpoint", "regex", "oauth", "dns",
            "kubernetes", "docker", "ci/cd", "microservice", "schema",
            "AST", "tokenizer", "embedding", "vector", "neural",
        ]
        tech_count = sum(1 for t in tech_terms if t.lower() in msg_lower)
        if tech_count > 0:
            for term in tech_terms:
                if term.lower() in msg_lower:
                    self._update_belief(term, knowledge_level=0.7, confidence=0.6)

        # Questions indicate knowledge gaps
        if "?" in message:
            # Topic extraction (simple)
            words = message.split()
            topic = " ".join(words[:5])
            self._update_belief(topic, knowledge_level=0.3, confidence=0.4)
            observation["signals"].append("knowledge_gap_signal")

    def _detect_goals(self, message: str, observation: Dict):
        """Detect goal signals."""
        msg_lower = message.lower()

        # Goal verbs
        goal_patterns = {
            "muốn": "desire",
            "want": "desire",
            "cần": "need",
            "need": "need",
            "làm": "action",
            "build": "create",
            "tạo": "create",
            "fix": "repair",
            "sửa": "repair",
            "tìm": "search",
            "find": "search",
            "hiểu": "understand",
            "understand": "understand",
        }

        for pattern, goal_type in goal_patterns.items():
            if pattern in msg_lower:
                # Check if this goal already exists
                existing = [g for g in self._goals if goal_type in g.description.lower()]
                if not existing:
                    goal = UserGoal(
                        description=message[:200],
                        priority=0.7,
                        explicit=True,
                    )
                    self._goals.append(goal)
                    observation["signals"].append("goal_detected:%s" % goal_type)
                break

    def _update_style(self, message: str, observation: Dict):
        """Update communication style based on message."""
        msg_len = len(message)

        # Verbosity
        if msg_len < 20:
            self._style.verbosity = self._style.verbosity * 0.8 + 0.2 * 0.2
        elif msg_len > 200:
            self._style.verbosity = self._style.verbosity * 0.8 + 0.2 * 0.8

        # Formality
        informal = ["mày", "tao", "ok", "lol", "haha", "ừ", "đc"]
        if any(w in message.lower() for w in informal):
            self._style.formality = max(0, self._style.formality - 0.1)

        # Language detection
        vietnamese_chars = set("ăâđêôơư")
        if any(c in message.lower() for c in vietnamese_chars):
            self._style.language = "vi"

    def _update_belief(self, topic: str, knowledge_level: float, confidence: float):
        """Update a belief about user's knowledge."""
        key = topic.lower()
        if key in self._beliefs:
            belief = self._beliefs[key]
            belief.knowledge_level = belief.knowledge_level * 0.7 + knowledge_level * 0.3
            belief.confidence = min(1.0, belief.confidence + 0.1)
            belief.last_updated = time.time()
        else:
            self._beliefs[key] = UserBelief(
                topic=topic, knowledge_level=knowledge_level,
                confidence=confidence,
            )

    def get_response_guidance(self) -> Dict[str, Any]:
        """Get guidance for how to respond to the user."""
        guidance = {
            "verbosity": "detailed" if self._style.verbosity > 0.6 else "concise",
            "technical_level": self._style.technical_level,
            "formality": "formal" if self._style.formality > 0.6 else "casual",
            "language": self._style.language,
            "pace": "fast" if self._style.pace > 0.6 else "thorough",
        }

        # Emotional adaptations
        if self._emotion.frustration > 0.6:
            guidance["tone"] = "apologetic, solution-focused"
            guidance["verbosity"] = "concise"
            guidance["extra"] = "Acknowledge frustration. Focus on fixing."
        elif self._emotion.confusion > 0.5:
            guidance["tone"] = "patient, explanatory"
            guidance["extra"] = "Break down explanation. Use examples."
        elif self._emotion.urgency > 0.6:
            guidance["tone"] = "efficient"
            guidance["verbosity"] = "concise"
            guidance["extra"] = "Skip explanations. Give solution directly."
        elif self._emotion.satisfaction > 0.7:
            guidance["tone"] = "warm, encouraging"
        else:
            guidance["tone"] = "neutral, helpful"

        # Knowledge adaptations
        low_knowledge_topics = [
            b.topic for b in self._beliefs.values()
            if b.knowledge_level < 0.3 and b.confidence > 0.5
        ]
        if low_knowledge_topics:
            guidance["explain_basics"] = low_knowledge_topics[:3]

        return guidance

    def anticipate_needs(self) -> List[str]:
        """Anticipate what the user might need next."""
        needs = []

        # Based on goals
        for goal in self._goals:
            if goal.progress < 0.5:
                needs.append("Continue working on: %s" % goal.description[:80])

        # Based on confusion
        if self._emotion.confusion > 0.5:
            needs.append("Provide clearer explanation with examples")

        # Based on knowledge gaps
        gaps = [b.topic for b in self._beliefs.values()
                if b.knowledge_level < 0.3 and b.confidence > 0.5]
        if gaps:
            needs.append("Explain: %s" % ", ".join(gaps[:3]))

        return needs[:5]

    def to_context(self, max_tokens: int = 500) -> str:
        """Generate Theory of Mind context for LLM."""
        guidance = self.get_response_guidance()
        parts = [
            "## User Model",
            "Language: %s" % guidance["language"],
            "Style: %s, %s" % (guidance["verbosity"], guidance["tone"]),
            "Frustration: %.0f%%" % (self._emotion.frustration * 100),
            "Confusion: %.0f%%" % (self._emotion.confusion * 100),
        ]
        if guidance.get("extra"):
            parts.append("Note: %s" % guidance["extra"])

        needs = self.anticipate_needs()
        if needs:
            parts.append("Anticipated needs:")
            for n in needs:
                parts.append("- %s" % n)

        return "\n".join(parts)

    def stats(self) -> Dict:
        return {
            "turns": self._turn_count,
            "beliefs": len(self._beliefs),
            "goals": len(self._goals),
            "emotion": {
                "frustration": round(self._emotion.frustration, 2),
                "confusion": round(self._emotion.confusion, 2),
                "satisfaction": round(self._emotion.satisfaction, 2),
                "engagement": round(self._emotion.engagement, 2),
            },
            "style": {
                "language": self._style.language,
                "verbosity": round(self._style.verbosity, 2),
                "formality": round(self._style.formality, 2),
            },
        }
