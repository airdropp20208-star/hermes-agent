"""
Self-Improvement Engine — agent learns from its own mistakes.
Tracks success/failure patterns, adapts strategies, improves over time.
"""
import time
import uuid
import json
import logging
import sqlite3
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ActionRecord:
    """Record of an agent action and its outcome."""
    id: str
    action_type: str  # tool_call, code_write, search, plan, etc.
    description: str
    context: Dict[str, Any]  # what was the situation
    strategy: str  # what approach was used
    outcome: str  # success, failure, partial
    result_summary: str
    duration_ms: float = 0
    error: str = ""
    retry_count: int = 0
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)


@dataclass
class Strategy:
    """A learned strategy for handling specific situations."""
    id: str
    name: str
    trigger_pattern: str  # when to use this strategy
    approach: str  # what to do
    success_rate: float = 0.0
    use_count: int = 0
    success_count: int = 0
    avg_duration_ms: float = 0
    created_from: List[str] = field(default_factory=list)  # action_ids
    confidence: float = 0.5
    last_used: float = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class Lesson:
    """A lesson learned from experience."""
    id: str
    category: str  # error_pattern, best_practice, shortcut, pitfall
    description: str
    learned_from: List[str]  # action_ids
    importance: float = 0.5
    times_applied: int = 0
    created_at: float = field(default_factory=time.time)


class SelfImprovementEngine:
    """
    Self-improvement engine with:
    - Action tracking (what was done, what happened)
    - Pattern recognition (recurring failures/successes)
    - Strategy generation (learn from experience)
    - Strategy selection (pick best approach for situation)
    - Lesson extraction (generalize from specific experiences)
    - Adaptive behavior (change approach based on history)
    - Performance metrics over time
    - SQLite persistence
    """

    def __init__(self, db_path: str = None):
        self._actions: List[ActionRecord] = []
        self._strategies: Dict[str, Strategy] = {}
        self._lessons: Dict[str, Lesson] = {}
        self._pattern_cache: Dict[str, List[str]] = defaultdict(list)
        self._db_path = db_path
        self._db = None

        if db_path:
            self._init_db(db_path)

    def _init_db(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS actions (
                id TEXT PRIMARY KEY, action_type TEXT, description TEXT,
                context TEXT, strategy TEXT, outcome TEXT, result_summary TEXT,
                duration_ms REAL, error TEXT, retry_count INTEGER,
                created_at REAL, tags TEXT, lessons TEXT
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY, name TEXT, trigger_pattern TEXT,
                approach TEXT, success_rate REAL, use_count INTEGER,
                success_count INTEGER, avg_duration_ms REAL,
                created_from TEXT, confidence REAL, last_used REAL,
                metadata TEXT
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS lessons (
                id TEXT PRIMARY KEY, category TEXT, description TEXT,
                learned_from TEXT, importance REAL, times_applied INTEGER,
                created_at REAL
            )
        """)
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_act_type ON actions(action_type)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_act_outcome ON actions(outcome)")
        self._db.commit()
        self._load()

    def _load(self):
        if not self._db:
            return
        for row in self._db.execute("SELECT * FROM strategies").fetchall():
            s = Strategy(id=row[0], name=row[1], trigger_pattern=row[2],
                         approach=row[3], success_rate=row[4], use_count=row[5],
                         success_count=row[6], avg_duration_ms=row[7],
                         created_from=json.loads(row[8] or "[]"),
                         confidence=row[9], last_used=row[10],
                         metadata=json.loads(row[11] or "{}"))
            self._strategies[s.id] = s
        for row in self._db.execute("SELECT * FROM lessons").fetchall():
            l = Lesson(id=row[0], category=row[1], description=row[2],
                       learned_from=json.loads(row[3] or "[]"),
                       importance=row[4], times_applied=row[5],
                       created_at=row[6])
            self._lessons[l.id] = l

    def record_action(self, action_type: str, description: str,
                      context: Dict = None, strategy: str = "",
                      outcome: str = "success", result_summary: str = "",
                      duration_ms: float = 0, error: str = "",
                      retry_count: int = 0, tags: List[str] = None) -> ActionRecord:
        """Record an action and its outcome."""
        action = ActionRecord(
            id="a_" + str(uuid.uuid4())[:8],
            action_type=action_type, description=description,
            context=context or {}, strategy=strategy,
            outcome=outcome, result_summary=result_summary,
            duration_ms=duration_ms, error=error,
            retry_count=retry_count, tags=tags or [],
        )
        self._actions.append(action)

        # Index by action type
        self._pattern_cache[action_type].append(action.id)

        # Auto-extract lessons on failure
        if outcome == "failure" and error:
            self._extract_lesson_from_failure(action)

        # Auto-extract lessons on success after failure
        if outcome == "success" and retry_count > 0:
            self._extract_recovery_pattern(action)

        # Save to DB
        if self._db:
            self._db.execute(
                "INSERT INTO actions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (action.id, action_type, description, json.dumps(context or {}),
                 strategy, outcome, result_summary, duration_ms, error,
                 retry_count, action.created_at, json.dumps(tags or []),
                 json.dumps(action.lessons)))
            self._db.commit()

        return action

    def create_strategy(self, name: str, trigger_pattern: str,
                        approach: str, from_actions: List[str] = None) -> Strategy:
        """Create a new strategy from experience."""
        sid = "s_" + str(uuid.uuid4())[:8]
        strategy = Strategy(
            id=sid, name=name, trigger_pattern=trigger_pattern,
            approach=approach, created_from=from_actions or [],
        )
        self._strategies[sid] = strategy

        if self._db:
            self._db.execute(
                "INSERT INTO strategies VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, name, trigger_pattern, approach, 0.0, 0, 0, 0.0,
                 json.dumps(from_actions or []), 0.5, 0, "{}"))
            self._db.commit()
        return strategy

    def suggest_strategy(self, situation: str, action_type: str = "") -> Optional[Strategy]:
        """Suggest the best strategy for a given situation."""
        candidates = []
        situation_lower = situation.lower()

        for strategy in self._strategies.values():
            # Match by trigger pattern
            if strategy.trigger_pattern.lower() in situation_lower:
                candidates.append(strategy)
            # Match by action type in metadata
            elif strategy.metadata.get("action_type") == action_type:
                candidates.append(strategy)

        if not candidates:
            return None

        # Score by: success_rate * confidence * recency
        now = time.time()
        def score(s):
            recency = 1.0 / (1.0 + (now - s.last_used) / 86400) if s.last_used else 0.5
            return s.success_rate * 0.4 + s.confidence * 0.3 + recency * 0.3

        candidates.sort(key=score, reverse=True)
        return candidates[0]

    def update_strategy_outcome(self, strategy_id: str, success: bool,
                                duration_ms: float = 0):
        """Update strategy statistics after use."""
        s = self._strategies.get(strategy_id)
        if not s:
            return
        s.use_count += 1
        if success:
            s.success_count += 1
        s.success_rate = s.success_count / s.use_count if s.use_count else 0
        s.avg_duration_ms = ((s.avg_duration_ms * (s.use_count - 1)) + duration_ms) / s.use_count
        s.last_used = time.time()
        s.confidence = min(1.0, s.confidence + 0.05 if success else s.confidence - 0.1)

        if self._db:
            self._db.execute(
                "UPDATE strategies SET success_rate=?, use_count=?, success_count=?, "
                "avg_duration_ms=?, confidence=?, last_used=? WHERE id=?",
                (s.success_rate, s.use_count, s.success_count,
                 s.avg_duration_ms, s.confidence, s.last_used, strategy_id))
            self._db.commit()

    def _extract_lesson_from_failure(self, action: ActionRecord):
        """Auto-extract lesson from a failed action."""
        if not action.error:
            return
        lid = "l_" + str(uuid.uuid4())[:8]
        lesson = Lesson(
            id=lid, category="error_pattern",
            description="%s failed: %s (context: %s)" % (
                action.action_type, action.error[:200],
                json.dumps(action.context)[:200]),
            learned_from=[action.id], importance=0.6,
        )
        self._lessons[lid] = lesson
        action.lessons.append(lid)

        if self._db:
            self._db.execute(
                "INSERT INTO lessons VALUES (?,?,?,?,?,?,?)",
                (lid, lesson.category, lesson.description,
                 json.dumps(lesson.learned_from), lesson.importance,
                 0, lesson.created_at))
            self._db.commit()

    def _extract_recovery_pattern(self, action: ActionRecord):
        """Extract pattern from successful recovery after retries."""
        lid = "l_" + str(uuid.uuid4())[:8]
        lesson = Lesson(
            id=lid, category="best_practice",
            description="After %d retries, %s succeeded with: %s" % (
                action.retry_count, action.action_type, action.strategy),
            learned_from=[action.id], importance=0.7,
        )
        self._lessons[lid] = lesson

    def get_lessons(self, category: str = None, min_importance: float = 0) -> List[Lesson]:
        """Get learned lessons."""
        results = []
        for lesson in self._lessons.values():
            if category and lesson.category != category:
                continue
            if lesson.importance < min_importance:
                continue
            results.append(lesson)
        results.sort(key=lambda l: l.importance, reverse=True)
        return results

    def get_failure_patterns(self, action_type: str = None) -> Dict[str, int]:
        """Get common failure patterns."""
        patterns = defaultdict(int)
        for a in self._actions:
            if a.outcome == "failure":
                if action_type and a.action_type != action_type:
                    continue
                error_key = "%s: %s" % (a.action_type, a.error[:100])
                patterns[error_key] += 1
        return dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True))

    def get_performance_trend(self, action_type: str = None,
                              window: int = 50) -> Dict:
        """Get performance trend over recent actions."""
        actions = [a for a in self._actions
                   if not action_type or a.action_type == action_type]
        recent = actions[-window:]

        if not recent:
            return {"total": 0, "success_rate": 0}

        successes = sum(1 for a in recent if a.outcome == "success")
        return {
            "total": len(recent),
            "successes": successes,
            "success_rate": successes / len(recent),
            "avg_duration_ms": sum(a.duration_ms for a in recent) / len(recent),
            "retry_rate": sum(1 for a in recent if a.retry_count > 0) / len(recent),
        }

    def to_context(self, max_tokens: int = 1000) -> str:
        """Generate context string of lessons and strategies for LLM."""
        parts = []
        total = 0

        # Top lessons
        lessons = self.get_lessons(min_importance=0.3)[:10]
        if lessons:
            parts.append("## Learned Lessons")
            for l in lessons:
                line = "- [%s] %s" % (l.category, l.description[:150])
                if total + len(line) > max_tokens * 4:
                    break
                parts.append(line)
                total += len(line)

        # Top strategies
        strategies = sorted(self._strategies.values(),
                          key=lambda s: s.success_rate, reverse=True)[:5]
        if strategies:
            parts.append("\n## Proven Strategies")
            for s in strategies:
                line = "- %s (%d%% success): %s" % (
                    s.name, int(s.success_rate * 100), s.approach[:150])
                if total + len(line) > max_tokens * 4:
                    break
                parts.append(line)
                total += len(line)

        return "\n".join(parts)

    def stats(self) -> Dict:
        total = len(self._actions)
        successes = sum(1 for a in self._actions if a.outcome == "success")
        return {
            "total_actions": total,
            "success_rate": successes / total if total else 0,
            "strategies": len(self._strategies),
            "lessons": len(self._lessons),
            "top_failure_patterns": list(self.get_failure_patterns().keys())[:5],
        }
