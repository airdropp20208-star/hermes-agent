"""
Session Manager — multi-session, branching, checkpointing.
Supports: session trees, snapshots, restore, export/import.
"""
import json
import time
import uuid
import copy
import sqlite3
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """A session checkpoint (snapshot)."""
    checkpoint_id: str
    session_id: str
    messages: List[Dict]
    metadata: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    description: str = ""


@dataclass
class Session:
    """A conversation session."""
    session_id: str
    title: str = ""
    parent_id: Optional[str] = None  # For branching
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[Checkpoint] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    state: str = "active"  # active | paused | archived

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def age(self) -> float:
        return time.time() - self.created_at


class SessionManager:
    """
    Session management with:
    - Multi-session support
    - Session branching (fork from any point)
    - Checkpointing and restore
    - SQLite persistence
    - Search across sessions
    - Export/import (JSON)
    - Session merging
    - Auto-cleanup of old sessions
    """

    def __init__(self, db_path: Optional[str] = None):
        self._sessions: Dict[str, Session] = {}
        self._active_session: Optional[str] = None
        self._db_path = db_path
        self._db: Optional[sqlite3.Connection] = None

        if db_path:
            self._init_db(db_path)

    def _init_db(self, path: str):
        """Initialize SQLite storage."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                parent_id TEXT,
                created_at REAL,
                updated_at REAL,
                metadata TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                state TEXT DEFAULT 'active'
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp REAL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                session_id TEXT,
                description TEXT,
                messages_snapshot TEXT,
                metadata TEXT DEFAULT '{}',
                created_at REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_cp_session ON checkpoints(session_id)")
        self._db.commit()
        self._load_sessions()

    def _load_sessions(self):
        """Load sessions from SQLite."""
        if not self._db:
            return
        cursor = self._db.execute("SELECT * FROM sessions WHERE state != 'archived'")
        for row in cursor.fetchall():
            session = Session(
                session_id=row[0], title=row[1] or "", parent_id=row[2],
                created_at=row[3], updated_at=row[4],
                metadata=json.loads(row[5] or "{}"),
                tags=json.loads(row[6] or "[]"), state=row[7],
            )
            # Load messages
            msg_cursor = self._db.execute(
                "SELECT role, content, timestamp, metadata FROM messages WHERE session_id=? ORDER BY id",
                (session.session_id,)
            )
            session.messages = [{
                "role": r[0], "content": r[1], "timestamp": r[2],
                "metadata": json.loads(r[3] or "{}")
            } for r in msg_cursor.fetchall()]

            self._sessions[session.session_id] = session

    def create(self, title: str = "", parent_id: str = None,
               metadata: Dict = None) -> Session:
        """Create a new session."""
        sid = str(uuid.uuid4())[:12]
        session = Session(
            session_id=sid, title=title or f"Session {sid[:6]}",
            parent_id=parent_id, metadata=metadata or {},
        )
        self._sessions[sid] = session
        self._active_session = sid

        if self._db:
            self._db.execute(
                "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?)",
                (sid, session.title, parent_id, session.created_at,
                 session.updated_at, json.dumps(metadata or {}),
                 json.dumps([]), "active")
            )
            self._db.commit()

        return session

    def get(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    @property
    def active(self) -> Optional[Session]:
        """Get the currently active session."""
        if self._active_session:
            return self._sessions.get(self._active_session)
        return None

    def set_active(self, session_id: str) -> bool:
        """Set the active session."""
        if session_id in self._sessions:
            self._active_session = session_id
            return True
        return False

    def add_message(self, session_id: str, role: str, content: str,
                    metadata: Dict = None):
        """Add a message to a session."""
        session = self._sessions.get(session_id)
        if not session:
            return

        msg = {
            "role": role, "content": content,
            "timestamp": time.time(), "metadata": metadata or {},
        }
        session.messages.append(msg)
        session.updated_at = time.time()

        if self._db:
            self._db.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, metadata) VALUES (?,?,?,?,?)",
                (session_id, role, content, msg["timestamp"], json.dumps(metadata or {}))
            )
            self._db.execute(
                "UPDATE sessions SET updated_at=? WHERE session_id=?",
                (msg["timestamp"], session_id)
            )
            self._db.commit()

    def checkpoint(self, session_id: str, description: str = "") -> Optional[Checkpoint]:
        """Create a checkpoint (snapshot) of a session."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        cp = Checkpoint(
            checkpoint_id=str(uuid.uuid4())[:8],
            session_id=session_id,
            messages=copy.deepcopy(session.messages),
            metadata=copy.deepcopy(session.metadata),
            description=description,
        )
        session.checkpoints.append(cp)

        if self._db:
            self._db.execute(
                "INSERT INTO checkpoints VALUES (?,?,?,?,?,?)",
                (cp.checkpoint_id, session_id, description,
                 json.dumps(cp.messages), json.dumps(cp.metadata), cp.created_at)
            )
            self._db.commit()

        return cp

    def restore(self, session_id: str, checkpoint_id: str) -> bool:
        """Restore session to a checkpoint."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        for cp in session.checkpoints:
            if cp.checkpoint_id == checkpoint_id:
                session.messages = copy.deepcopy(cp.messages)
                session.metadata = copy.deepcopy(cp.metadata)
                session.updated_at = time.time()
                return True
        return False

    def branch(self, session_id: str, from_message: int = -1,
               title: str = "") -> Optional[Session]:
        """Branch (fork) a session from a specific point."""
        parent = self._sessions.get(session_id)
        if not parent:
            return None

        child = self.create(
            title=title or f"Branch of {parent.title}",
            parent_id=session_id,
        )
        # Copy messages up to the branch point
        if from_message < 0:
            from_message = len(parent.messages) + from_message
        child.messages = copy.deepcopy(parent.messages[:from_message + 1])
        return child

    def merge(self, source_id: str, target_id: str) -> bool:
        """Merge source session's messages into target."""
        source = self._sessions.get(source_id)
        target = self._sessions.get(target_id)
        if not source or not target:
            return False

        # Add merge marker
        target.messages.append({
            "role": "system",
            "content": f"--- Merged from session {source.title} ({source_id}) ---",
            "timestamp": time.time(),
            "metadata": {"merge": True, "source": source_id},
        })
        target.messages.extend(copy.deepcopy(source.messages))
        target.updated_at = time.time()
        return True

    def search_messages(self, query: str, session_id: str = None) -> List[Dict]:
        """Search messages across sessions."""
        results = []
        sessions = [self._sessions[session_id]] if session_id else self._sessions.values()

        for session in sessions:
            for i, msg in enumerate(session.messages):
                if query.lower() in msg.get("content", "").lower():
                    results.append({
                        "session_id": session.session_id,
                        "session_title": session.title,
                        "message_index": i,
                        "role": msg["role"],
                        "content": msg["content"][:500],
                        "timestamp": msg.get("timestamp", 0),
                    })
        return results

    def list_sessions(self, state: str = None) -> List[Dict]:
        """List all sessions."""
        sessions = []
        for s in self._sessions.values():
            if state and s.state != state:
                continue
            sessions.append({
                "session_id": s.session_id,
                "title": s.title,
                "parent_id": s.parent_id,
                "messages": s.message_count,
                "state": s.state,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
                "tags": s.tags,
            })
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    def export_session(self, session_id: str) -> Optional[str]:
        """Export session as JSON."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        return json.dumps({
            "session_id": session.session_id,
            "title": session.title,
            "messages": session.messages,
            "metadata": session.metadata,
            "tags": session.tags,
            "created_at": session.created_at,
        }, indent=2)

    def import_session(self, json_data: str) -> Optional[Session]:
        """Import session from JSON."""
        try:
            data = json.loads(json_data)
            session = self.create(
                title=data.get("title", "Imported"),
                metadata=data.get("metadata", {}),
            )
            session.messages = data.get("messages", [])
            session.tags = data.get("tags", [])
            return session
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return None

    def archive(self, session_id: str) -> bool:
        """Archive a session."""
        session = self._sessions.get(session_id)
        if session:
            session.state = "archived"
            if self._db:
                self._db.execute("UPDATE sessions SET state='archived' WHERE session_id=?", (session_id,))
                self._db.commit()
            return True
        return False

    def cleanup(self, max_age_days: int = 30) -> int:
        """Archive sessions older than max_age_days."""
        cutoff = time.time() - (max_age_days * 86400)
        count = 0
        for s in self._sessions.values():
            if s.state == "active" and s.updated_at < cutoff:
                self.archive(s.session_id)
                count += 1
        return count

    def stats(self) -> Dict:
        """Get session manager stats."""
        states = {}
        for s in self._sessions.values():
            states[s.state] = states.get(s.state, 0) + 1
        return {
            "total_sessions": len(self._sessions),
            "by_state": states,
            "active_session": self._active_session,
            "total_messages": sum(s.message_count for s in self._sessions.values()),
        }
