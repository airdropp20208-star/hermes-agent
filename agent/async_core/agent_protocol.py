"""
Agent Protocol — agent-to-agent messaging, swarm coordination, consensus.
Multiple agents can collaborate, debate, and reach consensus.
"""
import time
import uuid
import json
import logging
import asyncio
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MessageType(Enum):
    REQUEST = "request"       # Ask another agent to do something
    RESPONSE = "response"     # Answer to a request
    BROADCAST = "broadcast"   # Send to all agents
    PROPOSAL = "proposal"     # Propose a solution
    VOTE = "vote"            # Vote on a proposal
    DELEGATION = "delegation" # Delegate a task
    FEEDBACK = "feedback"     # Give feedback on work
    HEARTBEAT = "heartbeat"   # Health check


@dataclass
class AgentMessage:
    """A message between agents."""
    id: str
    sender_id: str
    receiver_id: str  # "" for broadcast
    msg_type: MessageType
    content: Any
    in_reply_to: str = ""
    priority: float = 0.5
    ttl: float = 300  # seconds until expiry
    created_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


@dataclass
class AgentProfile:
    """Profile of a registered agent."""
    agent_id: str
    name: str
    capabilities: List[str] = field(default_factory=list)
    role: str = "worker"  # worker, reviewer, coordinator, specialist
    status: str = "online"  # online, busy, offline
    max_tasks: int = 3
    current_tasks: int = 0
    trust_score: float = 0.5
    total_contributions: int = 0


@dataclass
class Proposal:
    """A proposal for group decision-making."""
    id: str
    proposer_id: str
    description: str
    options: List[str] = field(default_factory=list)
    votes: Dict[str, str] = field(default_factory=dict)  # agent_id -> option
    status: str = "open"  # open, accepted, rejected
    deadline: float = 0
    created_at: float = field(default_factory=time.time)

    @property
    def result(self) -> Optional[str]:
        if not self.votes:
            return None
        counts = {}
        for vote in self.votes.values():
            counts[vote] = counts.get(vote, 0) + 1
        return max(counts, key=counts.get) if counts else None


class AgentProtocol:
    """
    Agent-to-agent communication with:
    - Direct messaging between agents
    - Broadcast to all agents
    - Task delegation and tracking
    - Proposal/voting for consensus
    - Capability-based routing
    - Trust scoring
    - Message queue with priority
    - Heartbeat monitoring
    """

    def __init__(self):
        self._agents: Dict[str, AgentProfile] = {}
        self._message_queue: List[AgentMessage] = []
        self._proposals: Dict[str, Proposal] = {}
        self._handlers: Dict[str, Callable] = {}
        self._message_history: List[AgentMessage] = []

    def register_agent(self, agent_id: str, name: str,
                       capabilities: List[str] = None,
                       role: str = "worker") -> AgentProfile:
        """Register an agent in the protocol."""
        profile = AgentProfile(
            agent_id=agent_id, name=name,
            capabilities=capabilities or [], role=role,
        )
        self._agents[agent_id] = profile
        return profile

    def unregister_agent(self, agent_id: str):
        """Remove an agent."""
        self._agents.pop(agent_id, None)

    def send(self, sender_id: str, receiver_id: str,
             msg_type: MessageType, content: Any,
             priority: float = 0.5) -> AgentMessage:
        """Send a message to another agent."""
        msg = AgentMessage(
            id="m_" + str(uuid.uuid4())[:8],
            sender_id=sender_id, receiver_id=receiver_id,
            msg_type=msg_type, content=content, priority=priority,
        )
        self._message_queue.append(msg)
        self._message_history.append(msg)
        return msg

    def broadcast(self, sender_id: str, msg_type: MessageType,
                  content: Any) -> List[AgentMessage]:
        """Broadcast a message to all agents."""
        messages = []
        for agent_id in self._agents:
            if agent_id != sender_id:
                msg = self.send(sender_id, agent_id, msg_type, content)
                messages.append(msg)
        return messages

    def get_messages(self, agent_id: str, consume: bool = True) -> List[AgentMessage]:
        """Get pending messages for an agent."""
        messages = [m for m in self._message_queue
                   if m.receiver_id == agent_id or m.receiver_id == ""]
        if consume:
            self._message_queue = [m for m in self._message_queue
                                  if m not in messages]
        return sorted(messages, key=lambda m: m.priority, reverse=True)

    def propose(self, proposer_id: str, description: str,
                options: List[str], deadline: float = 60) -> Proposal:
        """Create a proposal for group voting."""
        proposal = Proposal(
            id="pr_" + str(uuid.uuid4())[:8],
            proposer_id=proposer_id, description=description,
            options=options, deadline=time.time() + deadline,
        )
        self._proposals[proposal.id] = proposal

        # Broadcast proposal
        self.broadcast(proposer_id, MessageType.PROPOSAL, {
            "proposal_id": proposal.id,
            "description": description,
            "options": options,
        })

        return proposal

    def vote(self, agent_id: str, proposal_id: str, option: str) -> bool:
        """Vote on a proposal."""
        proposal = self._proposals.get(proposal_id)
        if not proposal or proposal.status != "open":
            return False
        if option not in proposal.options:
            return False
        proposal.votes[agent_id] = option

        # Check if all agents voted
        if len(proposal.votes) >= len(self._agents) - 1:
            self._finalize_proposal(proposal)

        return True

    def _finalize_proposal(self, proposal: Proposal):
        """Finalize a proposal based on votes."""
        result = proposal.result
        if result:
            proposal.status = "accepted"
            # Notify all agents
            self.broadcast("system", MessageType.BROADCAST, {
                "proposal_id": proposal.id,
                "result": result,
                "status": "accepted",
            })
        else:
            proposal.status = "rejected"

    def delegate_task(self, delegator_id: str, task: str,
                      required_capabilities: List[str] = None) -> Optional[AgentMessage]:
        """Delegate a task to the best available agent."""
        candidates = []
        for agent in self._agents.values():
            if agent.agent_id == delegator_id:
                continue
            if agent.status != "online":
                continue
            if agent.current_tasks >= agent.max_tasks:
                continue
            if required_capabilities:
                if not any(cap in agent.capabilities for cap in required_capabilities):
                    continue
            candidates.append(agent)

        if not candidates:
            return None

        # Pick best candidate by trust score and availability
        candidates.sort(key=lambda a: (a.trust_score, a.max_tasks - a.current_tasks),
                       reverse=True)
        chosen = candidates[0]

        msg = self.send(delegator_id, chosen.agent_id, MessageType.DELEGATION, {
            "task": task,
            "required_capabilities": required_capabilities or [],
        })
        chosen.current_tasks += 1
        return msg

    def update_trust(self, agent_id: str, success: bool):
        """Update agent trust score based on task outcome."""
        agent = self._agents.get(agent_id)
        if not agent:
            return
        agent.total_contributions += 1
        if success:
            agent.trust_score = min(1.0, agent.trust_score + 0.05)
        else:
            agent.trust_score = max(0, agent.trust_score - 0.1)

    def find_agents_by_capability(self, capability: str) -> List[AgentProfile]:
        """Find agents with a specific capability."""
        return [a for a in self._agents.values()
                if capability in a.capabilities and a.status == "online"]

    def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        return self._agents.get(agent_id)

    def list_agents(self) -> List[Dict]:
        return [{
            "id": a.agent_id, "name": a.name,
            "role": a.role, "status": a.status,
            "capabilities": a.capabilities,
            "trust": round(a.trust_score, 2),
            "tasks": a.current_tasks,
        } for a in self._agents.values()]

    def get_proposal(self, proposal_id: str) -> Optional[Proposal]:
        return self._proposals.get(proposal_id)

    def stats(self) -> Dict:
        return {
            "agents": len(self._agents),
            "online": sum(1 for a in self._agents.values() if a.status == "online"),
            "pending_messages": len(self._message_queue),
            "total_messages": len(self._message_history),
            "proposals": len(self._proposals),
            "accepted_proposals": sum(1 for p in self._proposals.values()
                                     if p.status == "accepted"),
        }
