"""Private durable admission, execution receipts, canonical facts and delivery."""

import fcntl
import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from pydantic import ValidationError

from lux_agent.protocol import (
    MAX_EXECUTION_FACT_BYTES,
    AgentItemSnapshot,
    Command,
    ExecutionFact,
    InputAccepted,
    ItemTextDelta,
    ModelResponseCompleted,
    ObservedTextFragment,
    OutboundFactIntent,
    QuerySessionExecution,
    ReplaySessionEvents,
    ResourceId,
    RunSnapshot,
    SessionExecutionSync,
    TurnSnapshot,
    WireModel,
    decode_command,
    decode_fact,
)

if TYPE_CHECKING:
    from lux_agent.reconciliation import ModelEvidenceView, OriginalModelEvidence


COMMAND_TOPIC = "lux.agent.commands.v2"
EVENT_TOPIC = "lux.agent.events.v2"


class EnvelopeBudgetExceeded(ValueError):
    """A retained source fact needs coordinated transport capacity repair."""


def now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(raw: str | bytes) -> str:
    """Hash original JSON values, preserving string/time and array representations."""
    return json.dumps(
        json.loads(raw),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(raw: str | bytes) -> str:
    return hashlib.sha256(canonical_json(raw).encode()).hexdigest()


@dataclass(frozen=True)
class InboundRecord:
    topic: str
    partition: int
    offset: int
    key: bytes | None
    value: bytes | None


@dataclass(frozen=True)
class StoredCommand:
    ordinal: int
    command: Command
    envelope: str
    application_state: str


@dataclass(frozen=True)
class Publication:
    id: int
    event: ExecutionFact
    envelope: str
    partition_key: str
    attempts: int
    dispatch_generation: int


class ExecutionPreparation(WireModel):
    execution_id: ResourceId
    kind: Literal["MODEL", "TOOL"]
    user_id: ResourceId
    session_id: ResourceId
    run_id: ResourceId
    turn_id: ResourceId
    tool_call_item_id: ResourceId | None = None
    operation_id: ResourceId | None = None
    # The store archives this named runtime/adapter-owned request without decoding it.
    request_snapshot: str


@dataclass(frozen=True)
class ExecutionReceipt:
    preparation: ExecutionPreparation
    state: str
    response_snapshot: str | None
    confirmed_at: str | None

    @property
    def operation_id(self) -> str | None:
        return self.preparation.operation_id


class ModelProviderBinding(WireModel):
    execution_id: ResourceId
    user_id: ResourceId
    session_id: ResourceId
    request_hash: str
    provider: Literal["openai"] = "openai"
    account_ref: str
    model: str
    endpoint: str
    organization_id: str | None = None
    project_id: str | None = None


class ModelProviderReceipt(WireModel):
    binding: ModelProviderBinding
    response_id: str | None = None
    dispatch_started: bool = True


@dataclass(frozen=True)
class FactTargets:
    run_id: str
    turn_id: str | None
    item_id: str | None


def targets(fact: OutboundFactIntent) -> FactTargets:
    body = fact.body
    if isinstance(body, RunSnapshot):
        return FactTargets(body.run.id, None, None)
    if isinstance(body, (TurnSnapshot, ModelResponseCompleted)):
        return FactTargets(body.turn.run.id, body.turn.id, None)
    return FactTargets(body.item.turn.run.id, body.item.turn.id, body.item.id)


_SCHEMA = """
CREATE TABLE command (
 ordinal INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
 envelope TEXT NOT NULL, schema_version INTEGER NOT NULL, event_type TEXT NOT NULL,
 user_id TEXT NOT NULL, session_id TEXT NOT NULL, item_id TEXT UNIQUE,
 input_sequence_no INTEGER, event_hash TEXT NOT NULL, received_at TEXT NOT NULL,
 application_state TEXT NOT NULL DEFAULT 'RECEIVED',  applied_checkpoint_id TEXT,  error
 TEXT,
 UNIQUE(session_id, input_sequence_no),  CHECK(application_state IN ('RECEIVED',
 'APPLIED', 'REJECTED'))
);
CREATE INDEX pending_command ON command(session_id,application_state,ordinal);
CREATE TABLE receipt (
 topic TEXT NOT NULL,  partition INTEGER NOT NULL,  offset INTEGER NOT NULL,  key BLOB,
 value BLOB,
 state TEXT NOT NULL, error TEXT, received_at TEXT NOT NULL, event_id TEXT,
 PRIMARY KEY(topic, partition, offset),  CHECK(state IN ('ACCEPTED', 'DUPLICATE',
 'REJECTED'))
);
CREATE TABLE run (run_id TEXT PRIMARY KEY,  user_id TEXT NOT NULL,  session_id TEXT NOT
 NULL,  created_at TEXT NOT NULL);
CREATE INDEX run_session ON run(user_id,session_id,run_id);
CREATE TABLE session_event_counter (
 session_id TEXT PRIMARY KEY,  user_id TEXT NOT NULL,  last_event_seq INTEGER NOT NULL
 DEFAULT 0,
 CHECK(last_event_seq>=0)
);
CREATE TABLE event (
 log_ordinal INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
 user_id TEXT NOT NULL,  session_id TEXT NOT NULL,  run_id TEXT,  turn_id TEXT,  item_id
 TEXT,
 event_type TEXT NOT NULL, event_seq INTEGER NOT NULL, stream_seq INTEGER,
 occurred_at TEXT NOT NULL,  intent_hash TEXT NOT NULL,  event_hash TEXT NOT NULL,
 envelope TEXT NOT NULL,
 UNIQUE(session_id,event_seq), UNIQUE(item_id,stream_seq), CHECK(event_seq>0),
 CHECK((event_type='ITEM_TEXT_DELTA' AND stream_seq>0) OR (event_type!='ITEM_TEXT_DELTA'
 AND stream_seq IS NULL))
);
CREATE INDEX session_replay ON event(session_id,event_seq);
CREATE INDEX item_fragments ON event(item_id,stream_seq);
CREATE TABLE publication (
 id INTEGER PRIMARY KEY AUTOINCREMENT,  event_id TEXT NOT NULL UNIQUE REFERENCES
 event(event_id),
 partition_key TEXT NOT NULL,  delivered INTEGER NOT NULL DEFAULT 0,  attempts INTEGER
 NOT NULL DEFAULT 0,
 next_attempt REAL NOT NULL DEFAULT 0, dispatch_generation INTEGER NOT NULL DEFAULT 1,
 acknowledged_at TEXT, last_error TEXT,
 CHECK(delivered IN (0,1)), CHECK(attempts>=0), CHECK(dispatch_generation>0)
);
CREATE INDEX pending_publication ON publication(delivered,partition_key,id);
CREATE TABLE execution_receipt (
 execution_id TEXT PRIMARY KEY,  kind TEXT NOT NULL,  user_id TEXT NOT NULL,  session_id
 TEXT NOT NULL,
 run_id TEXT NOT NULL REFERENCES run(run_id),  turn_id TEXT NOT NULL,  tool_call_item_id
 TEXT,  operation_id TEXT,
 request_hash TEXT NOT NULL, request_snapshot TEXT NOT NULL, state TEXT NOT NULL,
 response_snapshot TEXT,  created_at TEXT NOT NULL,  updated_at TEXT NOT NULL,
 confirmed_at TEXT,
 CHECK(kind IN ('MODEL', 'TOOL')),  CHECK(state IN ('PREPARED', 'IN_FLIGHT',
 'CONFIRMED', 'FAILED', 'UNKNOWN')),
 CHECK((kind='MODEL' AND tool_call_item_id IS NULL AND operation_id IS NULL) OR
 (kind='TOOL' AND tool_call_item_id IS NOT NULL))
);
CREATE UNIQUE INDEX model_turn ON execution_receipt(turn_id) WHERE kind='MODEL';
CREATE UNIQUE INDEX tool_call ON execution_receipt(tool_call_item_id) WHERE kind='TOOL';
CREATE UNIQUE INDEX execution_operation ON execution_receipt(user_id, operation_id)
 WHERE operation_id IS NOT NULL;
PRAGMA user_version=2;
"""


class AgentStore:
    """An OS lock covers this store and its paired native checkpointer lifecycle."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self._closed = False
        self._driver_active = False
        self._lock_fd = os.open(
            f"{path}.lock", os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600
        )
        connection: sqlite3.Connection | None = None
        try:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "Agent database is already owned by a process"
                ) from error
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(descriptor)
            connection = sqlite3.connect(
                path, isolation_level=None, timeout=1, check_same_thread=False
            )
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if version not in (0, 2) or (version == 0 and tables):
                raise RuntimeError(
                    "Incompatible legacy Agent schema; select a fresh local store"
                )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            if version == 0:
                connection.executescript("BEGIN IMMEDIATE;\n" + _SCHEMA + "COMMIT;")
            # Additive adapter receipt storage also opens existing owned v2 stores.
            connection.execute(
                "CREATE TABLE IF NOT EXISTS model_provider_receipt ("
                "execution_id TEXT PRIMARY KEY "
                "REFERENCES execution_receipt(execution_id),"
                "binding TEXT NOT NULL, response_id TEXT, created_at TEXT NOT NULL,"
                "dispatch_started INTEGER NOT NULL DEFAULT 1 "
                "CHECK(dispatch_started IN (0,1)))"
            )
            columns = connection.execute(
                "PRAGMA table_info(model_provider_receipt)"
            ).fetchall()
            if not any(column["name"] == "dispatch_started" for column in columns):
                # Old receipts surrounded the provider call; retain that uncertainty.
                connection.execute(
                    "ALTER TABLE model_provider_receipt ADD COLUMN dispatch_started "
                    "INTEGER NOT NULL DEFAULT 1 CHECK(dispatch_started IN (0,1))"
                )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS model_reconciliation_audit ("
                "execution_id TEXT PRIMARY KEY "
                "REFERENCES execution_receipt(execution_id),"
                "operator_id TEXT NOT NULL, evidence_digest TEXT NOT NULL,"
                "attachment TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            self._connection = connection
        except BaseException:
            if connection is not None:
                connection.close()
            os.close(self._lock_fd)
            raise

    @contextmanager
    def driver_lifecycle(self) -> Iterator[None]:
        with self._lock:
            if self._driver_active:
                raise RuntimeError("This store already has a Session driver")
            self._driver_active = True
        try:
            yield
        finally:
            with self._lock:
                self._driver_active = False

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
                self._connection.execute("COMMIT")
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise

    def _owner(self, connection: sqlite3.Connection, owner: str, session: str) -> None:
        row = connection.execute(
            "SELECT user_id FROM command WHERE session_id=? UNION SELECT user_id "
            "FROM run WHERE session_id=? "
            "UNION SELECT user_id FROM session_event_counter WHERE session_id=?",
            (session, session, session),
        ).fetchone()
        if row is not None and row[0] != owner:
            raise ValueError("Session owner cannot change")

    def _admit(
        self, connection: sqlite3.Connection, raw: str, command: Command
    ) -> StoredCommand:
        self._owner(connection, command.user_id, command.session_id)
        digest = content_hash(raw)
        existing = connection.execute(
            "SELECT * FROM command WHERE event_id=?", (command.event_id,)
        ).fetchone()
        if existing is not None:
            if existing["event_hash"] != digest:
                raise ValueError(
                    "Conflicting event identity; original content preserved"
                )
            return StoredCommand(
                existing["ordinal"],
                decode_command(existing["envelope"]),
                existing["envelope"],
                existing["application_state"],
            )
        item_id = command.item.id if isinstance(command, InputAccepted) else None
        sequence = (
            command.item.input_sequence_no
            if isinstance(command, InputAccepted)
            else None
        )
        cursor = connection.execute(
            "INSERT INTO command(event_id,envelope,schema_version,event_type,user_id,"
            "session_id,item_id,"
            "input_sequence_no,event_hash,received_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                command.event_id,
                raw,
                2,
                command.event_type,
                command.user_id,
                command.session_id,
                item_id,
                sequence,
                digest,
                now(),
            ),
        )
        return StoredCommand(cursor.lastrowid, command, raw, "RECEIVED")

    def admit(self, raw: str | bytes) -> StoredCommand:
        envelope = raw.decode() if isinstance(raw, bytes) else raw
        command = decode_command(envelope)
        try:
            with self._transaction() as connection:
                return self._admit(connection, envelope, command)
        except (ValueError, sqlite3.IntegrityError) as error:
            # HTTP repair has no Kafka receipt; keep diagnostics
            # on the original command.
            with self._transaction() as connection:
                connection.execute(
                    "UPDATE command SET error=? WHERE event_id=?",
                    (str(error), command.event_id),
                )
            raise ValueError(str(error)) from error

    def receive(self, record: InboundRecord) -> str:
        with self._transaction() as connection:
            previous = connection.execute(
                "SELECT * FROM receipt WHERE topic=? AND partition=? AND offset=?",
                (record.topic, record.partition, record.offset),
            ).fetchone()
            if previous is not None:
                if previous["key"] != record.key or previous["value"] != record.value:
                    raise ValueError("Conflicting physical receipt")
                return previous["state"]
            state, error, event_id = "REJECTED", None, None
            try:
                if record.topic != COMMAND_TOPIC or record.value is None:
                    raise ValueError("Unexpected command topic or empty record")
                command = decode_command(record.value)
                event_id = command.event_id
                if record.key != command.partition_key.encode():
                    raise ValueError("Partition key disagrees with owner/Session")
                duplicate = connection.execute(
                    "SELECT 1 FROM command WHERE event_id=?", (event_id,)
                ).fetchone()
                self._admit(connection, record.value.decode(), command)
                state = "DUPLICATE" if duplicate else "ACCEPTED"
            except (
                ValidationError,
                ValueError,
                sqlite3.IntegrityError,
                UnicodeError,
            ) as invalid:
                error = str(invalid)[:1000]
            connection.execute(
                "INSERT INTO receipt VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    record.topic,
                    record.partition,
                    record.offset,
                    record.key,
                    record.value,
                    state,
                    error,
                    now(),
                    event_id,
                ),
            )
            return state

    def receipt_error(self, record: InboundRecord) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT error FROM receipt WHERE topic=? AND partition=? AND offset=?",
                (record.topic, record.partition, record.offset),
            ).fetchone()
            return row[0] if row else None

    def commands(
        self, session_id: str, after_ordinal: int = 0
    ) -> tuple[StoredCommand, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM command WHERE session_id=? AND ordinal>? "
                "AND application_state!='REJECTED' ORDER BY ordinal",
                (session_id, after_ordinal),
            ).fetchall()
            return tuple(
                StoredCommand(
                    r["ordinal"],
                    decode_command(r["envelope"]),
                    r["envelope"],
                    r["application_state"],
                )
                for r in rows
            )

    def session_ids(self) -> tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT session_id FROM command UNION SELECT session_id FROM run "
                "UNION SELECT session_id FROM session_event_counter"
            ).fetchall()
            return tuple(row[0] for row in rows)

    def command_state(self, event_id: str) -> str:
        with self._lock:
            return self._connection.execute(
                "SELECT application_state FROM command WHERE event_id=?", (event_id,)
            ).fetchone()[0]

    def mark_applied(self, event_ids: tuple[str, ...], checkpoint_id: str) -> None:
        with self._transaction() as connection:
            connection.executemany(
                "UPDATE command SET application_state='APPLIED',"
                "applied_checkpoint_id=? "
                "WHERE event_id=? AND application_state='RECEIVED'",
                ((checkpoint_id, identity) for identity in event_ids),
            )

    def register_run(self, owner: str, session: str, run_id: str) -> None:
        with self._transaction() as connection:
            self._owner(connection, owner, session)
            row = connection.execute(
                "SELECT user_id,session_id FROM run WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is not None and tuple(row) != (owner, session):
                raise ValueError("Conflicting Run identity")
            connection.execute(
                "INSERT OR IGNORE INTO run VALUES (?,?,?,?)",
                (run_id, owner, session, now()),
            )

    def _existing(
        self, connection: sqlite3.Connection, identity: str, digest: str
    ) -> ExecutionFact | None:
        row = connection.execute(
            "SELECT intent_hash,envelope FROM event WHERE event_id=?", (identity,)
        ).fetchone()
        if row is None:
            return None
        if row["intent_hash"] != digest:
            raise ValueError("Conflicting event identity/content")
        return decode_fact(row["envelope"])

    def _append(
        self, connection: sqlite3.Connection, intent: OutboundFactIntent, digest: str
    ) -> ExecutionFact:
        existing = self._existing(connection, intent.event_id, digest)
        if existing is not None:
            return existing
        self._owner(connection, intent.user_id, intent.session_id)
        connection.execute(
            "INSERT OR IGNORE INTO session_event_counter VALUES (?,?,0)",
            (intent.session_id, intent.user_id),
        )
        connection.execute(
            "UPDATE session_event_counter SET last_event_seq=last_event_seq+1 WHERE "
            "session_id=?",
            (intent.session_id,),
        )
        sequence = connection.execute(
            "SELECT last_event_seq FROM session_event_counter WHERE session_id=?",
            (intent.session_id,),
        ).fetchone()[0]
        fact = ExecutionFact(**intent.model_dump(), event_seq=sequence)
        raw = fact.model_dump_json()
        if len(raw.encode("utf-8")) > MAX_EXECUTION_FACT_BYTES:
            raise EnvelopeBudgetExceeded(
                "Execution fact envelope exceeds the serialized UTF-8 budget"
            )
        routing = targets(intent)
        stream = fact.body.stream_seq if isinstance(fact.body, ItemTextDelta) else None
        connection.execute(
            "INSERT INTO event(event_id,user_id,session_id,run_id,turn_id,item_id,"
            "event_type,"
            "event_seq,stream_seq,occurred_at,intent_hash,event_hash,envelope) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                fact.event_id,
                fact.user_id,
                fact.session_id,
                routing.run_id,
                routing.turn_id,
                routing.item_id,
                fact.event_type,
                sequence,
                stream,
                fact.occurred_at.isoformat(),
                digest,
                content_hash(raw),
                raw,
            ),
        )
        connection.execute(
            "INSERT INTO publication(event_id,partition_key) VALUES (?,?)",
            (fact.event_id, fact.partition_key),
        )
        return fact

    def append_fact(self, intent: OutboundFactIntent) -> ExecutionFact:
        with self._transaction() as connection:
            return self._append(
                connection, intent, content_hash(intent.model_dump_json())
            )

    def append_fragment(self, fragment: ObservedTextFragment) -> ExecutionFact:
        digest = content_hash(fragment.model_dump_json())
        with self._transaction() as connection:
            existing = self._existing(connection, fragment.event_id, digest)
            if existing is not None:
                return existing
            initial = connection.execute(
                "SELECT envelope FROM event WHERE item_id=? "
                "AND event_type='AGENT_ITEM_SNAPSHOT' ORDER BY event_seq LIMIT 1",
                (fragment.item.id,),
            ).fetchone()
            if initial is None or decode_fact(initial[0]).body.item != fragment.item:
                raise ValueError("Fragment requires its actual initial Item snapshot")
            stream = connection.execute(
                "SELECT COALESCE(MAX(stream_seq),0)+1 FROM event WHERE item_id=?",
                (fragment.item.id,),
            ).fetchone()[0]
            intent = OutboundFactIntent(
                **fragment.model_dump(exclude={"item", "text"}),
                event_type="ITEM_TEXT_DELTA",
                body=ItemTextDelta(
                    item=fragment.item, stream_seq=stream, text=fragment.text
                ),
            )
            return self._append(connection, intent, digest)

    def high_water(self, session: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT last_event_seq FROM session_event_counter WHERE session_id=?",
                (session,),
            ).fetchone()
            return row[0] if row else 0

    def events(
        self,
        session: str,
        after: int = 0,
        through: int | None = None,
        limit: int = 1000,
    ) -> tuple[ExecutionFact, ...]:
        with self._lock:
            end = self.high_water(session) if through is None else through
            rows = self._connection.execute(
                "SELECT envelope FROM event WHERE session_id=? AND event_seq>? "
                "AND event_seq<=? ORDER BY event_seq LIMIT ?",
                (session, after, end, limit),
            ).fetchall()
            return tuple(decode_fact(row[0]) for row in rows)

    def pending_publications(self, at: float | None = None) -> tuple[Publication, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT p.*,e.envelope FROM publication p JOIN event e ON "
                "e.event_id=p.event_id "
                "WHERE p.delivered=0 AND p.next_attempt<=? AND NOT EXISTS "
                "(SELECT 1 FROM publication earlier WHERE earlier.delivered=0 "
                "AND earlier.partition_key=p.partition_key AND earlier.id<p.id) "
                "ORDER BY p.id LIMIT 100",
                (time.time() if at is None else at,),
            ).fetchall()
            return tuple(
                Publication(
                    r["id"],
                    decode_fact(r["envelope"]),
                    r["envelope"],
                    r["partition_key"],
                    r["attempts"],
                    r["dispatch_generation"],
                )
                for r in rows
            )

    def publication_delivered(self, identity: int, generation: int) -> None:
        with self._transaction() as connection:
            connection.execute(
                "UPDATE publication SET delivered=1,acknowledged_at=?,last_error=NULL "
                "WHERE id=? AND dispatch_generation=?",
                (now(), identity, generation),
            )

    def publication_failed(
        self, identity: int, generation: int, error: str = "Publication failed"
    ) -> None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT attempts FROM publication WHERE id=? AND dispatch_generation=?",
                (identity, generation),
            ).fetchone()
            if row is not None:
                delay = min(60, 0.25 * 2 ** min(row[0], 8))
                connection.execute(
                    "UPDATE publication SET attempts=attempts+1,next_attempt=?,"
                    "last_error=? "
                    "WHERE id=? AND dispatch_generation=?",
                    (time.time() + delay, error[:1000], identity, generation),
                )

    def replay(self, command: ReplaySessionEvents) -> None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT envelope,application_state FROM command WHERE event_id=?",
                (command.event_id,),
            ).fetchone()
            if row is None or decode_command(row["envelope"]) != command:
                raise ValueError("Replay requires original durable command")
            if row["application_state"] != "RECEIVED":
                return
            connection.execute(
                "UPDATE publication SET dispatch_generation=dispatch_generation+1,"
                "delivered=0,"
                "acknowledged_at=NULL,next_attempt=0,attempts=0,last_error=NULL "
                "WHERE event_id IN "
                "(SELECT event_id FROM event WHERE session_id=? AND event_seq>? AND "
                "event_seq<=? "
                "ORDER BY event_seq LIMIT ?)",
                (
                    command.session_id,
                    command.after_event_seq,
                    command.through_event_seq,
                    command.limit,
                ),
            )
            connection.execute(
                "UPDATE command SET application_state='APPLIED' WHERE event_id=?",
                (command.event_id,),
            )

    def observe(self, command: QuerySessionExecution) -> None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT envelope,application_state FROM command WHERE event_id=?",
                (command.event_id,),
            ).fetchone()
            if row is None or decode_command(row["envelope"]) != command:
                raise ValueError("Query requires original durable command")
            if row["application_state"] != "RECEIVED":
                return
            connection.execute(
                "UPDATE publication SET dispatch_generation=dispatch_generation+1,"
                "delivered=0,"
                "acknowledged_at=NULL,next_attempt=0,attempts=0,last_error=NULL "
                "WHERE event_id IN "
                "(SELECT event_id FROM event WHERE session_id=? ORDER BY event_seq "
                "DESC LIMIT ?)",
                (command.session_id, command.limit),
            )
            connection.execute(
                "UPDATE command SET application_state='APPLIED' WHERE event_id=?",
                (command.event_id,),
            )

    def sync(
        self,
        owner: str,
        session: str,
        after: int = 0,
        through: int | None = None,
        limit: int = 100,
    ) -> SessionExecutionSync:
        if (
            not 1 <= limit <= 1000
            or after < 0
            or (through is not None and through < after)
        ):
            raise ValueError("Invalid source sync range")
        # A short local transaction fixes both the committed head and source views.
        with self._transaction() as connection:
            self._owner(connection, owner, session)
            row = connection.execute(
                "SELECT last_event_seq FROM session_event_counter WHERE session_id=?",
                (session,),
            ).fetchone()
            head = row[0] if row else 0
            end = min(head, through) if through is not None else head
            rows = connection.execute(
                "SELECT envelope FROM event WHERE session_id=? AND event_seq<=? "
                "ORDER BY event_seq",
                (session, head),
            ).fetchall()
            facts = tuple(decode_fact(row[0]) for row in rows)
            runs: dict[str, RunSnapshot] = {}
            turns: dict[str, TurnSnapshot] = {}
            items: dict[str, AgentItemSnapshot] = {}

            def retain[T: RunSnapshot | TurnSnapshot | AgentItemSnapshot](
                views: dict[str, T], key: str, value: T
            ) -> None:
                previous = views.get(key)
                if previous is None or value.source_revision > previous.source_revision:
                    views[key] = value
                elif (
                    value.source_revision == previous.source_revision
                    and value != previous
                ):
                    raise ValueError("Conflicting canonical source revision")

            for fact in facts:
                body = fact.body
                if isinstance(body, RunSnapshot):
                    retain(runs, body.run.id, body)
                elif isinstance(body, TurnSnapshot):
                    retain(turns, body.turn.id, body)
                elif isinstance(body, AgentItemSnapshot):
                    retain(items, body.item.id, body)
                elif isinstance(body, ModelResponseCompleted):
                    for member in body.items:
                        retain(items, member.item.id, member)
            page = tuple(fact for fact in facts if after < fact.event_seq <= end)[
                :limit
            ]
            cursor = page[-1].event_seq if page and page[-1].event_seq < end else None
            return SessionExecutionSync(
                user_id=owner,
                session_id=session,
                event_high_watermark=head,
                events=page,
                runs=tuple(runs.values()),
                turns=tuple(turns.values()),
                items=tuple(items.values()),
                next_after_event_seq=cursor,
            )

    def prepare_execution(self, prepared: ExecutionPreparation) -> ExecutionReceipt:
        if (
            prepared.kind == "MODEL"
            and (
                prepared.tool_call_item_id is not None
                or prepared.operation_id is not None
            )
        ) or (prepared.kind == "TOOL" and prepared.tool_call_item_id is None):
            raise ValueError("Invalid execution binding")
        if (
            prepared.operation_id is not None
            and prepared.operation_id != prepared.tool_call_item_id
        ):
            raise ValueError("Submission operation must equal original call Item")
        with self._transaction() as connection:
            run = connection.execute(
                "SELECT user_id,session_id FROM run WHERE run_id=?", (prepared.run_id,)
            ).fetchone()
            if run is None or tuple(run) != (prepared.user_id, prepared.session_id):
                raise ValueError("Execution requires original Run registration")
            existing = self.execution(prepared.execution_id)
            if existing is not None:
                if existing.preparation != prepared:
                    raise ValueError("Conflicting execution identity/request")
                return existing
            connection.execute(
                "INSERT INTO execution_receipt VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    prepared.execution_id,
                    prepared.kind,
                    prepared.user_id,
                    prepared.session_id,
                    prepared.run_id,
                    prepared.turn_id,
                    prepared.tool_call_item_id,
                    prepared.operation_id,
                    content_hash(prepared.request_snapshot),
                    prepared.request_snapshot,
                    "PREPARED",
                    None,
                    now(),
                    now(),
                    None,
                ),
            )
            return ExecutionReceipt(prepared, "PREPARED", None, None)

    def execution(self, identity: str) -> ExecutionReceipt | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM execution_receipt WHERE execution_id=?", (identity,)
            ).fetchone()
            if row is None:
                return None
            prepared = ExecutionPreparation(
                execution_id=row["execution_id"],
                kind=row["kind"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                run_id=row["run_id"],
                turn_id=row["turn_id"],
                tool_call_item_id=row["tool_call_item_id"],
                operation_id=row["operation_id"],
                request_snapshot=row["request_snapshot"],
            )
            return ExecutionReceipt(
                prepared, row["state"], row["response_snapshot"], row["confirmed_at"]
            )

    def execution_started(self, identity: str) -> None:
        with self._transaction() as connection:
            connection.execute(
                "UPDATE execution_receipt SET state='IN_FLIGHT',updated_at=? "
                "WHERE execution_id=? AND state='PREPARED'",
                (now(), identity),
            )

    def model_provider_receipt(self, identity: str) -> ModelProviderReceipt | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT binding,response_id,dispatch_started "
                "FROM model_provider_receipt "
                "WHERE execution_id=?",
                (identity,),
            ).fetchone()
            if row is None:
                return None
            return ModelProviderReceipt(
                binding=ModelProviderBinding.model_validate_json(row["binding"]),
                response_id=row["response_id"],
                dispatch_started=bool(row["dispatch_started"]),
            )

    def prepare_model_provider(
        self, binding: ModelProviderBinding, *, dispatch_started: bool = True
    ) -> None:
        with self._transaction() as connection:
            execution = self.execution(binding.execution_id)
            if execution is None or execution.preparation.kind != "MODEL":
                raise ValueError("Provider binding requires original model execution")
            prepared = execution.preparation
            if (
                prepared.user_id != binding.user_id
                or prepared.session_id != binding.session_id
                or content_hash(prepared.request_snapshot) != binding.request_hash
            ):
                raise ValueError("Conflicting model provider request binding")
            if execution.state != "IN_FLIGHT":
                raise ValueError("Provider submission requires IN_FLIGHT execution")
            saved = self.model_provider_receipt(binding.execution_id)
            if saved is not None:
                raise ValueError("Provider submission binding already exists")
            connection.execute(
                "INSERT INTO model_provider_receipt "
                "(execution_id,binding,response_id,created_at,dispatch_started) "
                "VALUES (?,?,NULL,?,?)",
                (
                    binding.execution_id,
                    binding.model_dump_json(),
                    now(),
                    int(dispatch_started),
                ),
            )

    def start_model_provider(self, binding: ModelProviderBinding) -> None:
        """Durably cross the dispatch boundary once, after local media hydration."""
        with self._transaction() as connection:
            saved = self.model_provider_receipt(binding.execution_id)
            receipt = self.execution(binding.execution_id)
            if (
                saved is None
                or saved.binding != binding
                or saved.dispatch_started
                or saved.response_id is not None
                or receipt is None
                or receipt.state != "IN_FLIGHT"
            ):
                raise ValueError("Conflicting model provider dispatch boundary")
            connection.execute(
                "UPDATE model_provider_receipt SET dispatch_started=1 "
                "WHERE execution_id=?",
                (binding.execution_id,),
            )

    def observe_model_provider(
        self, binding: ModelProviderBinding, response_id: str
    ) -> None:
        if not response_id:
            raise ValueError("Provider response identity is required")
        with self._transaction() as connection:
            saved = self.model_provider_receipt(binding.execution_id)
            if saved is None or saved.binding != binding or not saved.dispatch_started:
                raise ValueError("Conflicting model provider binding")
            if saved.response_id not in (None, response_id):
                raise ValueError("Conflicting original provider response identity")
            connection.execute(
                "UPDATE model_provider_receipt SET response_id=? WHERE execution_id=?",
                (response_id, binding.execution_id),
            )

    def model_evidence_view(
        self, identity: str, accounts: tuple[str, ...]
    ) -> ModelEvidenceView | None:
        from lux_agent.reconciliation import ModelEvidenceView

        with self._lock:
            saved = self.model_provider_receipt(identity)
            receipt = self.execution(identity)
            if (
                saved is None
                or receipt is None
                or saved.binding.account_ref not in accounts
            ):
                return None
            return ModelEvidenceView(
                binding=saved.binding,
                run_id=receipt.preparation.run_id,
                turn_id=receipt.preparation.turn_id,
                state=receipt.state,
                response_id=saved.response_id,
            )

    def unresolved_model_evidence(
        self, accounts: tuple[str, ...]
    ) -> tuple[ModelEvidenceView, ...]:
        if not accounts:
            return ()
        placeholders = ",".join("?" for _ in accounts)
        with self._lock:
            rows = self._connection.execute(
                "SELECT m.execution_id FROM model_provider_receipt m JOIN "
                "execution_receipt e ON e.execution_id=m.execution_id WHERE "
                "e.state IN ('UNKNOWN','IN_FLIGHT') AND m.response_id IS NULL "
                "AND m.dispatch_started=1 "
                f"AND json_extract(m.binding,'$.account_ref') IN ({placeholders}) "
                "ORDER BY m.created_at LIMIT 128",
                accounts,
            ).fetchall()
            return tuple(
                view
                for row in rows
                if (view := self.model_evidence_view(row[0], accounts)) is not None
            )

    def attach_model_evidence(
        self, evidence: OriginalModelEvidence, operator: str, accounts: tuple[str, ...]
    ) -> ModelEvidenceView:
        if (
            not operator.strip()
            or not evidence.evidence_reference.strip()
            or not evidence.reason.strip()
        ):
            raise ValueError("Trusted operator and original evidence are required")
        binding = evidence.binding
        encoded = evidence.model_dump_json()
        digest = content_hash(encoded)
        with self._transaction() as connection:
            saved = self.model_provider_receipt(binding.execution_id)
            receipt = self.execution(binding.execution_id)
            if (
                saved is None
                or receipt is None
                or saved.binding != binding
                or binding.account_ref not in accounts
                or not saved.dispatch_started
            ):
                raise ValueError("Conflicting original provider binding")
            prepared = receipt.preparation
            if (
                prepared.run_id != evidence.run_id
                or prepared.turn_id != evidence.turn_id
                or content_hash(prepared.request_snapshot) != binding.request_hash
            ):
                raise ValueError("Conflicting original execution binding")
            audit = connection.execute(
                "SELECT evidence_digest FROM model_reconciliation_audit "
                "WHERE execution_id=?",
                (binding.execution_id,),
            ).fetchone()
            if audit is not None:
                if audit[0] != digest or saved.response_id != evidence.response_id:
                    raise ValueError("Original evidence is immutable")
                return self.model_evidence_view(binding.execution_id, accounts)
            if (
                receipt.state not in ("UNKNOWN", "IN_FLIGHT")
                or saved.response_id is not None
            ):
                raise ValueError(
                    "Only missing original response evidence can be attached"
                )
            connection.execute(
                "UPDATE model_provider_receipt SET response_id=? WHERE execution_id=?",
                (evidence.response_id, binding.execution_id),
            )
            connection.execute(
                "INSERT INTO model_reconciliation_audit VALUES (?,?,?,?,?)",
                (binding.execution_id, operator, digest, encoded, now()),
            )
            return self.model_evidence_view(binding.execution_id, accounts)

    def execution_unknown(self, identity: str) -> None:
        with self._transaction() as connection:
            connection.execute(
                "UPDATE execution_receipt SET state='UNKNOWN',updated_at=? "
                "WHERE execution_id=? AND state='IN_FLIGHT'",
                (now(), identity),
            )

    def execution_confirmed(
        self, identity: str, response: str, failed: bool = False
    ) -> None:
        canonical_json(response)
        with self._transaction() as connection:
            receipt = self.execution(identity)
            if receipt is None:
                raise ValueError("Execution was not prepared")
            if receipt.state in ("CONFIRMED", "FAILED"):
                if (
                    content_hash(receipt.response_snapshot) != content_hash(response)
                    or (receipt.state == "FAILED") != failed
                ):
                    raise ValueError("Conflicting confirmed execution outcome")
                return
            connection.execute(
                "UPDATE execution_receipt SET state=?,response_snapshot=?,"
                "updated_at=?,confirmed_at=? "
                "WHERE execution_id=?",
                ("FAILED" if failed else "CONFIRMED", response, now(), now(), identity),
            )

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                os.close(self._lock_fd)
                self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
