"""
Auto-Recovery — conversation checkpoints, rollback, state machine.
Ensures agent conversations can survive failures and be resumed.
"""
import time
import uuid
import json
import copy
import logging
import sqlite3
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    TOOL_CALLING = "tool_calling"
    WAITING_INPUT = "waiting_input"
    RECOVERING = "recovering"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class ConversationCheckpoint:
    """A snapshot of conversation state."""
    checkpoint_id: str
    conversation_id: str
    state: ConversationState
    messages: List[Dict]
    context: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    description: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """A recovery action to take after failure."""
    action_type: str  # "retry", "rollback", "skip", "ask_user", "abort"
    checkpoint_id: str = ""
    message: str = ""
    params: Dict = field(default_factory=dict)


class StateMachine:
    """Conversation state machine with transitions."""

    TRANSITIONS = {
        ConversationState.ACTIVE: {
            "pause": ConversationState.PAUSED,
            "tool_call": ConversationState.TOOL_CALLING,
            "complete": ConversationState.COMPLETED,
            "fail": ConversationState.FAILED,
            "wait": ConversationState.WAITING_INPUT,
        },
        ConversationState.PAUSED: {
            "resume": ConversationState.ACTIVE,
            "abort": ConversationState.FAILED,
        },
        ConversationState.TOOL_CALLING: {
            "tool_done": ConversationState.ACTIVE,
            "tool_fail": ConversationState.RECOVERING,
            "complete": ConversationState.COMPLETED,
        },
        ConversationState.WAITING_INPUT: {
            "input_received": ConversationState.ACTIVE,
            "timeout": ConversationState.RECOVERING,
        },
        ConversationState.RECOVERING: {
            "recovered": ConversationState.ACTIVE,
            "rollback": ConversationState.ACTIVE,
            "abort": ConversationState.FAILED,
        },
        ConversationState.FAILED: {
            "restart": ConversationState.ACTIVE,
        },
        ConversationState.COMPLETED: {
            "restart": ConversationState.ACTIVE,
        },
    }

    def __init__(self, initial: ConversationState = ConversationState.ACTIVE):
        self.state = initial
        self._history: List[Tuple[ConversationState, ConversationState, str]] = []
        self._callbacks: Dict[str, List[Callable]] = {}

    def transition(self, action: str) -> bool:
        """Attempt a state transition. Returns True if successful."""
        transitions = self.TRANSITIONS.get(self.state, {})
        if action not in transitions:
            logger.warning("Invalid transition: %s -> %s" % (self.state.value, action))
            return False

        old_state = self.state
        self.state = transitions[action]
        self._history.append((old_state, self.state, action))

        for cb in self._callbacks.get("on_transition", []):
            try:
                cb(old_state, self.state, action)
            except Exception:
                pass

        return True

    def on_transition(self, callback: Callable):
        self._callbacks.setdefault("on_transition", []).append(callback)

    def can(self, action: str) -> bool:
        return action in self.TRANSITIONS.get(self.state, {})


class AutoRecovery:
    """
    Auto-recovery system with:
    - Periodic checkpointing
    - Automatic rollback on failure
    - Retry with exponential backoff
    - Conversation state machine
    - Persistent checkpoint storage
    - Recovery strategies per failure type
    - Manual rollback support
    """

    def __init__(self, conversation_id: str = None, db_path: str = None,
                 checkpoint_interval: int = 10):
        self.conversation_id = conversation_id or str(uuid.uuid4())[:12]
        self.state_machine = StateMachine()
        self.checkpoint_interval = checkpoint_interval
        self._checkpoints: List[ConversationCheckpoint] = []
        self._messages_since_checkpoint = 0
        self._retry_counts: Dict[str, int] = {}
        self._max_retries = 3
        self._recovery_strategies: Dict[str, Callable] = {}
        self._db_path = db_path
        self._db = None

        if db_path:
            self._init_db(db_path)

    def _init_db(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                conversation_id TEXT,
                state TEXT,
                messages TEXT,
                context TEXT,
                created_at REAL,
                description TEXT,
                metadata TEXT
            )
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv ON checkpoints(conversation_id)
        """)
        self._db.commit()

    def checkpoint(self, messages: List[Dict], context: Dict = None,
                   description: str = "") -> ConversationCheckpoint:
        """Create a checkpoint."""
        cp = ConversationCheckpoint(
            checkpoint_id=str(uuid.uuid4())[:8],
            conversation_id=self.conversation_id,
            state=self.state_machine.state,
            messages=copy.deepcopy(messages),
            context=copy.deepcopy(context or {}),
            description=description,
        )
        self._checkpoints.append(cp)
        self._messages_since_checkpoint = 0

        if self._db:
            self._db.execute(
                "INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?)",
                (cp.checkpoint_id, cp.conversation_id, cp.state.value,
                 json.dumps(cp.messages), json.dumps(cp.context),
                 cp.created_at, cp.description, json.dumps(cp.metadata))
            )
            self._db.commit()

        return cp

    def should_checkpoint(self, new_messages: int = 1) -> bool:
        """Check if we should create a checkpoint."""
        self._messages_since_checkpoint += new_messages
        return self._messages_since_checkpoint >= self.checkpoint_interval

    def auto_checkpoint(self, messages: List[Dict], context: Dict = None):
        """Checkpoint if interval has been reached."""
        if self.should_checkpoint():
            return self.checkpoint(messages, context, "auto-checkpoint")
        return None

    def rollback(self, checkpoint_id: str = None) -> Optional[ConversationCheckpoint]:
        """Rollback to a checkpoint. If no ID, rollback to latest."""
        if checkpoint_id:
            for cp in reversed(self._checkpoints):
                if cp.checkpoint_id == checkpoint_id:
                    self.state_machine.transition("rollback")
                    return cp
            return None

        if self._checkpoints:
            self.state_machine.transition("rollback")
            return self._checkpoints[-1]
        return None

    def handle_failure(self, error: Exception, context: Dict = None) -> RecoveryAction:
        """Determine recovery action for a failure."""
        error_type = type(error).__name__
        retries = self._retry_counts.get(error_type, 0)

        # Check custom strategy
        if error_type in self._recovery_strategies:
            return self._recovery_strategies[error_type](error, context)

        self.state_machine.transition("tool_fail")

        # Auto-retry for transient errors
        transient = (TimeoutError, ConnectionError, OSError, BrokenPipeError)
        if isinstance(error, transient) and retries < self._max_retries:
            self._retry_counts[error_type] = retries + 1
            self.state_machine.transition("recovered")
            return RecoveryAction(
                action_type="retry",
                message="Retrying after %s (attempt %d/%d)" % (
                    error_type, retries + 1, self._max_retries
                ),
                params={"delay": 2 ** retries},
            )

        # Rollback for persistent errors
        if self._checkpoints:
            self.state_machine.transition("rollback")
            return RecoveryAction(
                action_type="rollback",
                checkpoint_id=self._checkpoints[-1].checkpoint_id,
                message="Rolling back due to: %s" % error,
            )

        # Abort as last resort
        self.state_machine.transition("abort")
        return RecoveryAction(
            action_type="abort",
            message="No recovery possible: %s" % error,
        )

    def register_strategy(self, error_type: str, handler: Callable):
        """Register custom recovery strategy for an error type."""
        self._recovery_strategies[error_type] = handler

    def get_checkpoints(self) -> List[Dict]:
        """List all checkpoints."""
        return [{
            "id": cp.checkpoint_id,
            "state": cp.state.value,
            "messages": len(cp.messages),
            "description": cp.description,
            "created_at": cp.created_at,
        } for cp in self._checkpoints]

    def get_latest_checkpoint(self) -> Optional[ConversationCheckpoint]:
        return self._checkpoints[-1] if self._checkpoints else None

    def stats(self) -> Dict:
        return {
            "conversation_id": self.conversation_id,
            "state": self.state_machine.state.value,
            "checkpoints": len(self._checkpoints),
            "messages_since_checkpoint": self._messages_since_checkpoint,
            "retry_counts": dict(self._retry_counts),
            "transitions": len(self.state_machine._history),
        }
