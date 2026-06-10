"""
Knowledge Graph — structured memory with entities, relationships, graph queries.
Agent can reason over structured knowledge, not just flat text.
"""
import time
import uuid
import json
import logging
import sqlite3
from typing import Optional, Dict, Any, List, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A knowledge graph entity (node)."""
    id: str
    name: str
    entity_type: str  # person, concept, tool, file, project, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    confidence: float = 1.0  # 0-1, how confident we are about this entity
    source: str = ""  # where we learned about this entity
    mention_count: int = 1

    @property
    def display(self) -> str:
        return "%s (%s)" % (self.name, self.entity_type)


@dataclass
class Relationship:
    """A directed relationship (edge) between two entities."""
    id: str
    source_id: str
    target_id: str
    rel_type: str  # uses, depends_on, created_by, related_to, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # strength of relationship
    created_at: float = field(default_factory=time.time)
    evidence: List[str] = field(default_factory=list)  # supporting facts


@dataclass
class GraphQuery:
    """Result of a graph query."""
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    paths: List[List[str]] = field(default_factory=list)  # entity id paths
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """
    Knowledge graph with:
    - Entity management (create, update, merge)
    - Relationship tracking with evidence
    - Graph traversal (BFS, DFS, shortest path)
    - Pattern matching queries
    - Subgraph extraction
    - Entity resolution (dedup similar entities)
    - Confidence scoring
    - Temporal queries (when did we learn X?)
    - SQLite persistence
    - LLM-friendly context generation
    """

    def __init__(self, db_path: str = None):
        self._entities: Dict[str, Entity] = {}
        self._relationships: Dict[str, Relationship] = {}
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> set of rel_ids
        self._type_index: Dict[str, Set[str]] = defaultdict(set)  # type -> entity_ids
        self._name_index: Dict[str, str] = {}  # lowercase name -> entity_id
        self._db_path = db_path
        self._db = None

        if db_path:
            self._init_db(db_path)

    def _init_db(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY, name TEXT, entity_type TEXT,
                properties TEXT, created_at REAL, updated_at REAL,
                confidence REAL, source TEXT, mention_count INTEGER
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY, source_id TEXT, target_id TEXT,
                rel_type TEXT, properties TEXT, weight REAL,
                created_at REAL, evidence TEXT
            )
        """)
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_ent_type ON entities(entity_type)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_ent_name ON entities(name)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_rel_src ON relationships(source_id)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_rel_tgt ON relationships(target_id)")
        self._db.commit()
        self._load_from_db()

    def _load_from_db(self):
        if not self._db:
            return
        for row in self._db.execute("SELECT * FROM entities").fetchall():
            e = Entity(id=row[0], name=row[1], entity_type=row[2],
                       properties=json.loads(row[3] or "{}"),
                       created_at=row[4], updated_at=row[5],
                       confidence=row[6], source=row[7] or "", mention_count=row[8] or 1)
            self._entities[e.id] = e
            self._type_index[e.entity_type].add(e.id)
            self._name_index[e.name.lower()] = e.id

        for row in self._db.execute("SELECT * FROM relationships").fetchall():
            r = Relationship(id=row[0], source_id=row[1], target_id=row[2],
                             rel_type=row[3], properties=json.loads(row[4] or "{}"),
                             weight=row[5], created_at=row[6],
                             evidence=json.loads(row[7] or "[]"))
            self._relationships[r.id] = r
            self._adjacency[r.source_id].add(r.id)
            self._adjacency[r.target_id].add(r.id)

    def add_entity(self, name: str, entity_type: str = "concept",
                   properties: Dict = None, source: str = "",
                   confidence: float = 1.0) -> Entity:
        """Add or update an entity."""
        # Check for existing entity
        existing = self.find_entity(name)
        if existing:
            existing.mention_count += 1
            existing.updated_at = time.time()
            if properties:
                existing.properties.update(properties)
            existing.confidence = max(existing.confidence, confidence)
            if self._db:
                self._db.execute(
                    "UPDATE entities SET mention_count=?, updated_at=?, properties=?, confidence=? WHERE id=?",
                    (existing.mention_count, existing.updated_at,
                     json.dumps(existing.properties), existing.confidence, existing.id))
                self._db.commit()
            return existing

        eid = "e_" + str(uuid.uuid4())[:8]
        entity = Entity(id=eid, name=name, entity_type=entity_type,
                        properties=properties or {}, source=source,
                        confidence=confidence)
        self._entities[eid] = entity
        self._type_index[entity_type].add(eid)
        self._name_index[name.lower()] = eid

        if self._db:
            self._db.execute(
                "INSERT INTO entities VALUES (?,?,?,?,?,?,?,?,?)",
                (eid, name, entity_type, json.dumps(properties or {}),
                 entity.created_at, entity.updated_at, confidence, source, 1))
            self._db.commit()
        return entity

    def add_relationship(self, source_id: str, target_id: str,
                         rel_type: str, weight: float = 1.0,
                         properties: Dict = None, evidence: str = "") -> Relationship:
        """Add a relationship between two entities."""
        # Check for existing
        for rid in self._adjacency.get(source_id, set()):
            r = self._relationships.get(rid)
            if r and r.source_id == source_id and r.target_id == target_id and r.rel_type == rel_type:
                r.weight = max(r.weight, weight)
                if evidence:
                    r.evidence.append(evidence)
                return r

        rid = "r_" + str(uuid.uuid4())[:8]
        rel = Relationship(id=rid, source_id=source_id, target_id=target_id,
                           rel_type=rel_type, weight=weight,
                           properties=properties or {},
                           evidence=[evidence] if evidence else [])
        self._relationships[rid] = rel
        self._adjacency[source_id].add(rid)
        self._adjacency[target_id].add(rid)

        if self._db:
            self._db.execute(
                "INSERT INTO relationships VALUES (?,?,?,?,?,?,?,?)",
                (rid, source_id, target_id, rel_type,
                 json.dumps(properties or {}), weight, rel.created_at,
                 json.dumps(rel.evidence)))
            self._db.commit()
        return rel

    def find_entity(self, name: str) -> Optional[Entity]:
        """Find entity by name (case-insensitive)."""
        eid = self._name_index.get(name.lower())
        return self._entities.get(eid) if eid else None

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def get_neighbors(self, entity_id: str, rel_type: str = None,
                      direction: str = "both") -> List[Tuple[Entity, Relationship]]:
        """Get neighboring entities connected to this entity."""
        results = []
        for rid in self._adjacency.get(entity_id, set()):
            r = self._relationships.get(rid)
            if not r:
                continue
            if rel_type and r.rel_type != rel_type:
                continue

            neighbor_id = None
            if direction in ("out", "both") and r.source_id == entity_id:
                neighbor_id = r.target_id
            elif direction in ("in", "both") and r.target_id == entity_id:
                neighbor_id = r.source_id

            if neighbor_id and neighbor_id in self._entities:
                results.append((self._entities[neighbor_id], r))

        return results

    def find_path(self, start_id: str, end_id: str, max_depth: int = 5) -> List[List[str]]:
        """BFS shortest path between two entities."""
        if start_id == end_id:
            return [[start_id]]

        visited = {start_id}
        queue = [[start_id]]
        paths = []

        while queue and len(paths) < 3:
            path = queue.pop(0)
            if len(path) > max_depth:
                break
            current = path[-1]

            for rid in self._adjacency.get(current, set()):
                r = self._relationships.get(rid)
                if not r:
                    continue
                neighbor = r.target_id if r.source_id == current else r.source_id
                if neighbor == end_id:
                    paths.append(path + [neighbor])
                elif neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return paths

    def query_pattern(self, source_type: str = None, rel_type: str = None,
                      target_type: str = None) -> List[Tuple[Entity, Relationship, Entity]]:
        """Pattern match: (entity_type_A) -[rel_type]-> (entity_type_B)."""
        results = []
        for r in self._relationships.values():
            src = self._entities.get(r.source_id)
            tgt = self._entities.get(r.target_id)
            if not src or not tgt:
                continue
            if source_type and src.entity_type != source_type:
                continue
            if rel_type and r.rel_type != rel_type:
                continue
            if target_type and tgt.entity_type != target_type:
                continue
            results.append((src, r, tgt))
        return results

    def get_subgraph(self, entity_ids: Set[str], depth: int = 1) -> 'KnowledgeGraph':
        """Extract a subgraph around given entities."""
        included = set(entity_ids)
        for _ in range(depth):
            for eid in list(included):
                for rid in self._adjacency.get(eid, set()):
                    r = self._relationships.get(rid)
                    if r:
                        included.add(r.source_id)
                        included.add(r.target_id)

        sub = KnowledgeGraph()
        for eid in included:
            e = self._entities.get(eid)
            if e:
                sub._entities[eid] = e
                sub._type_index[e.entity_type].add(eid)
                sub._name_index[e.name.lower()] = eid

        for rid, r in self._relationships.items():
            if r.source_id in included and r.target_id in included:
                sub._relationships[rid] = r
                sub._adjacency[r.source_id].add(rid)
                sub._adjacency[r.target_id].add(rid)

        return sub

    def merge_entities(self, id1: str, id2: str) -> Entity:
        """Merge two entities (keep id1, redirect id2's relationships)."""
        e1 = self._entities.get(id1)
        e2 = self._entities.get(id2)
        if not e1 or not e2:
            return e1 or e2

        # Merge properties
        e1.properties.update(e2.properties)
        e1.mention_count += e2.mention_count
        e1.confidence = max(e1.confidence, e2.confidence)
        e1.updated_at = time.time()

        # Redirect relationships
        for rid in list(self._adjacency.get(id2, set())):
            r = self._relationships.get(rid)
            if not r:
                continue
            if r.source_id == id2:
                r.source_id = id1
            if r.target_id == id2:
                r.target_id = id1
            self._adjacency[id1].add(rid)

        # Remove e2
        del self._entities[id2]
        self._adjacency.pop(id2, None)
        self._type_index.get(e2.entity_type, set()).discard(id2)
        self._name_index.pop(e2.name.lower(), None)

        return e1

    def to_context(self, max_tokens: int = 2000) -> str:
        """Generate LLM-friendly context from the knowledge graph."""
        # Sort by mention count (most important first)
        entities = sorted(self._entities.values(), key=lambda e: e.mention_count, reverse=True)
        parts = []
        total = 0

        for entity in entities[:50]:
            neighbors = self.get_neighbors(entity.id)
            line = "%s [%s]" % (entity.name, entity.entity_type)
            for neighbor, rel in neighbors[:5]:
                line += " --%s--> %s" % (rel.rel_type, neighbor.name)

            if total + len(line) > max_tokens * 4:
                break
            parts.append(line)
            total += len(line)

        return "\n".join(parts)

    def stats(self) -> Dict:
        type_counts = {t: len(ids) for t, ids in self._type_index.items()}
        rel_counts = defaultdict(int)
        for r in self._relationships.values():
            rel_counts[r.rel_type] += 1
        return {
            "entities": len(self._entities),
            "relationships": len(self._relationships),
            "entity_types": dict(type_counts),
            "relationship_types": dict(rel_counts),
        }
