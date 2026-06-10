"""
Chain-of-Thought Engine — structured reasoning with decomposition,
verification, backtracking, and multi-path exploration.

Not just prompting tricks — this is a genuine reasoning architecture.
"""
import time
import uuid
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ThoughtType(Enum):
    OBSERVATION = "observation"    # What I see/know
    HYPOTHESIS = "hypothesis"      # What might be true
    DEDUCTION = "deduction"        # What follows logically
    INFERENCE = "inference"        # What I can conclude
    QUESTION = "question"          # What I need to find out
    VERIFICATION = "verification"  # Checking if something is correct
    REFLECTION = "reflection"      # Thinking about the process
    DECISION = "decision"          # Choosing between options


@dataclass
class Thought:
    """A single thought in a reasoning chain."""
    id: str
    type: ThoughtType
    content: str
    confidence: float  # 0-1
    parent_id: str = ""  # what thought led to this
    children_ids: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    verified: bool = False
    verification_result: str = ""


@dataclass
class ReasoningChain:
    """A complete chain of reasoning."""
    id: str
    goal: str
    thoughts: List[Thought] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0
    alternative_chains: List[str] = field(default_factory=list)
    backtrack_count: int = 0
    total_duration_ms: float = 0


class ChainOfThoughtEngine:
    """
    Structured reasoning engine with:
    - Step-by-step thought chains
    - Branching exploration (try multiple paths)
    - Backtracking on dead ends
    - Evidence accumulation
    - Verification steps
    - Confidence propagation
    - Reasoning templates for common patterns
    - Contradiction detection
    - Chain comparison (which reasoning path is best?)
    """

    def __init__(self):
        self._chains: Dict[str, ReasoningChain] = {}
        self._current_chain: Optional[ReasoningChain] = None
        self._templates: Dict[str, List[ThoughtType]] = {
            "debug": [ThoughtType.OBSERVATION, ThoughtType.HYPOTHESIS,
                     ThoughtType.VERIFICATION, ThoughtType.DECISION],
            "analyze": [ThoughtType.OBSERVATION, ThoughtType.INFERENCE,
                       ThoughtType.HYPOTHESIS, ThoughtType.VERIFICATION,
                       ThoughtType.DEDUCTION],
            "decide": [ThoughtType.OBSERVATION, ThoughtType.HYPOTHESIS,
                      ThoughtType.HYPOTHESIS, ThoughtType.VERIFICATION,
                      ThoughtType.DECISION],
            "create": [ThoughtType.OBSERVATION, ThoughtType.HYPOTHESIS,
                      ThoughtType.DEDUCTION, ThoughtType.INFERENCE,
                      ThoughtType.DECISION],
        }

    def start_chain(self, goal: str, template: str = None) -> ReasoningChain:
        """Start a new reasoning chain."""
        chain = ReasoningChain(
            id="c_" + str(uuid.uuid4())[:8],
            goal=goal,
        )
        self._chains[chain.id] = chain
        self._current_chain = chain

        if template and template in self._templates:
            chain.metadata = {"template": template,
                            "expected_steps": [t.value for t in self._templates[template]]}

        return chain

    def think(self, type: ThoughtType, content: str,
              confidence: float = 0.5, parent_id: str = None,
              evidence_for: List[str] = None,
              evidence_against: List[str] = None) -> Thought:
        """Add a thought to the current chain."""
        if not self._current_chain:
            self.start_chain("General reasoning")

        thought = Thought(
            id="t_" + str(uuid.uuid4())[:8],
            type=type,
            content=content,
            confidence=confidence,
            parent_id=parent_id or (self._current_chain.thoughts[-1].id
                                   if self._current_chain.thoughts else ""),
            supporting_evidence=evidence_for or [],
            contradicting_evidence=evidence_against or [],
        )

        # Link to parent
        if thought.parent_id:
            for t in self._current_chain.thoughts:
                if t.id == thought.parent_id:
                    t.children_ids.append(thought.id)
                    break

        # Propagate confidence
        if thought.parent_id and thought.supporting_evidence:
            thought.confidence = min(1.0, confidence * 0.8 + 0.2)
        if thought.contradicting_evidence:
            thought.confidence = max(0.01, thought.confidence * 0.6)

        self._current_chain.thoughts.append(thought)
        return thought

    def verify(self, thought_id: str, check: str, result: bool,
               evidence: str = "") -> Thought:
        """Verify a thought/hypothesis."""
        verification = self.think(
            ThoughtType.VERIFICATION,
            "Verify: %s -> %s" % (check, "PASS" if result else "FAIL"),
            confidence=0.9 if result else 0.1,
            parent_id=thought_id,
            evidence_for=[evidence] if result and evidence else [],
            evidence_against=[evidence] if not result and evidence else [],
        )
        verification.verified = True
        verification.verification_result = "pass" if result else "fail"
        return verification

    def backtrack(self, to_thought_id: str = None):
        """Backtrack to a previous thought and try a different path."""
        if not self._current_chain:
            return

        self._current_chain.backtrack_count += 1

        if to_thought_id:
            # Find the thought and mark everything after as abandoned
            for i, t in enumerate(self._current_chain.thoughts):
                if t.id == to_thought_id:
                    # Keep thoughts up to this point
                    self._current_chain.thoughts = self._current_chain.thoughts[:i+1]
                    break
        else:
            # Backtrack to last decision point
            for i in range(len(self._current_chain.thoughts) - 1, -1, -1):
                if self._current_chain.thoughts[i].type == ThoughtType.DECISION:
                    self._current_chain.thoughts = self._current_chain.thoughts[:i]
                    break

    def conclude(self, conclusion: str, confidence: float = None) -> ReasoningChain:
        """Draw conclusion from the reasoning chain."""
        if not self._current_chain:
            return None

        self._current_chain.conclusion = conclusion

        # Calculate overall confidence from chain
        if confidence is None:
            confidences = [t.confidence for t in self._current_chain.thoughts]
            if confidences:
                # Geometric mean (punishes low-confidence steps)
                product = 1.0
                for c in confidences:
                    product *= max(0.01, c)
                confidence = product ** (1.0 / len(confidences))
            else:
                confidence = 0.5

        self._current_chain.confidence = confidence
        return self._current_chain

    def explore_alternatives(self, goal: str, alternatives: List[str]) -> List[ReasoningChain]:
        """Explore multiple reasoning paths for the same goal."""
        chains = []
        for alt in alternatives:
            chain = self.start_chain(goal)
            self.think(ThoughtType.HYPOTHESIS, alt, confidence=0.5)
            chains.append(chain)

        return chains

    def compare_chains(self, chain_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple reasoning chains to find the best."""
        chains = [self._chains.get(cid) for cid in chain_ids]
        chains = [c for c in chains if c]

        if not chains:
            return {}

        scored = []
        for chain in chains:
            # Score = confidence * (1 - backtrack_penalty) * evidence_ratio
            evidence_count = sum(
                len(t.supporting_evidence) for t in chain.thoughts
            )
            contradiction_count = sum(
                len(t.contradicting_evidence) for t in chain.thoughts
            )
            evidence_ratio = evidence_count / max(1, evidence_count + contradiction_count)
            backtrack_penalty = min(chain.backtrack_count * 0.1, 0.5)

            score = chain.confidence * (1 - backtrack_penalty) * evidence_ratio
            scored.append((score, chain))

        scored.sort(reverse=True)
        best = scored[0][1]

        return {
            "best_chain": best.id,
            "best_conclusion": best.conclusion,
            "best_confidence": best.confidence,
            "all_scores": {c.id: round(s, 3) for s, c in scored},
        }

    def get_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        return self._chains.get(chain_id)

    def chain_to_text(self, chain_id: str = None) -> str:
        """Convert chain to readable text."""
        chain = self._chains.get(chain_id) if chain_id else self._current_chain
        if not chain:
            return ""

        parts = ["Goal: %s\n" % chain.goal]
        for t in chain.thoughts:
            marker = {
                ThoughtType.OBSERVATION: "OBSERVE",
                ThoughtType.HYPOTHESIS: "HYPOTHESIZE",
                ThoughtType.DEDUCTION: "DEDUCE",
                ThoughtType.INFERENCE: "INFER",
                ThoughtType.QUESTION: "QUESTION",
                ThoughtType.VERIFICATION: "VERIFY",
                ThoughtType.REFLECTION: "REFLECT",
                ThoughtType.DECISION: "DECIDE",
            }.get(t.type, "THINK")
            prefix = "[%s] (%.0f%%)" % (marker, t.confidence * 100)
            parts.append("%s %s" % (prefix, t.content))

        if chain.conclusion:
            parts.append("\nCONCLUSION (%.0f%%): %s" % (
                chain.confidence * 100, chain.conclusion))

        return "\n".join(parts)

    def stats(self) -> Dict:
        total_thoughts = sum(len(c.thoughts) for c in self._chains.values())
        return {
            "chains": len(self._chains),
            "total_thoughts": total_thoughts,
            "templates": list(self._templates.keys()),
        }
