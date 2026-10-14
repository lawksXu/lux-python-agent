import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from test_protocol_v2 import SESSION, USER, input_json

from lux_agent import protocol


def store_at(path):
    from lux_agent.store import AgentStore

    return AgentStore(path)


def intent(body, event_type="RUN_SNAPSHOT", event_id=None):
    return protocol.OutboundFactIntent(
        schema_version=2,
        event_id=event_id or str(uuid4()),
        event_type=event_type,
        occurred_at=datetime.now(UTC),
        user_id=USER,
        session_id=SESSION,
        body=body,
    )


def running():
    return protocol.RunSnapshot(
        run=protocol.RunIdentity(id=str(uuid4()), run_index=1),
        source_revision=1,
        status="RUNNING",
        started_at=None,
        finished_at=None,
        final_item=None,
        error_code=None,
        error_message=None,
    )


def test_logical_input_dedup_keeps_original_time_and_admission_ordinal(tmp_path):
    with store_at(tmp_path / "agent.db") as store:
        raw = input_json()
        first = store.admit(raw)
        duplicate = store.admit(raw)
        assert first.ordinal == duplicate.ordinal
        assert first.envelope == raw
        assert first.application_state == "RECEIVED"
        changed = json.loads(raw)
        changed["item"]["content"]["blocks"][0]["text"] = "changed"
        with pytest.raises(ValueError, match="Conflicting"):
            store.admit(json.dumps(changed))
        assert store.commands(SESSION)[0].envelope == raw
        assert store.command_state(first.command.event_id) == "RECEIVED"


def test_receipt_is_durable_before_ack_and_duplicate_offsets_do_not_duplicate_commands(
    tmp_path,
):
    from lux_agent.store import InboundRecord

    record = InboundRecord(
        "lux.agent.commands.v2",
        0,
        8,
        f"{USER}:{SESSION}".encode(),
        input_json().encode(),
    )
    path = tmp_path / "agent.db"
    with store_at(path) as store:
        assert store.receive(record) == "ACCEPTED"
        assert store.receive(record) == "ACCEPTED"
    with store_at(path) as store:
        assert len(store.commands(SESSION)) == 1
        bad = InboundRecord(record.topic, 0, 9, b"wrong-key", record.value)
        assert store.receive(bad) == "REJECTED"
        assert store.receipt_error(bad)


def test_canonical_duplicates_and_conflicts_do_not_allocate_new_session_positions(
    tmp_path,
):
    with store_at(tmp_path / "agent.db") as store:
        original = intent(running())
        assert store.append_fact(original).event_seq == 1
        assert store.append_fact(original).event_seq == 1
        changed = original.model_copy(update={"occurred_at": datetime.now(UTC)})
        with pytest.raises(ValueError, match="Conflicting"):
            store.append_fact(changed)
        assert store.append_fact(intent(running())).event_seq == 2
        assert store.high_water(SESSION) == 2


def test_fragments_have_separate_contiguous_sequences_and_keep_repeated_text(tmp_path):
    with store_at(tmp_path / "agent.db") as store:
        turn = protocol.TurnIdentity(id=str(uuid4()), turn_index=1, run=running().run)
        item = protocol.AgentItemIdentity(id=str(uuid4()), item_index=1, turn=turn)
        initial = protocol.AgentItemSnapshot(
            item=item,
            source_revision=1,
            kind="MESSAGE",
            status="PARTIAL",
            payload=protocol.OrderedContent(blocks=()),
            completed_at=None,
            call_id=None,
            tool_name=None,
        )
        store.append_fact(intent(initial, "AGENT_ITEM_SNAPSHOT"))
        fragment = protocol.ObservedTextFragment(
            schema_version=2,
            event_id=str(uuid4()),
            occurred_at=datetime.now(UTC),
            user_id=USER,
            session_id=SESSION,
            item=item,
            text=" ",
        )
        first = store.append_fragment(fragment)
        assert store.append_fragment(fragment) == first
        second = store.append_fragment(
            fragment.model_copy(update={"event_id": str(uuid4())})
        )
        assert (first.body.stream_seq, second.body.stream_seq) == (1, 2)
        assert (first.event_seq, second.event_seq) == (2, 3)


def test_explicit_replay_generation_protects_against_old_publish_ack(tmp_path):
    with store_at(tmp_path / "agent.db") as store:
        store.append_fact(intent(running()))
        original = store.pending_publications()[0]
        command = protocol.ReplaySessionEvents(
            schema_version=2,
            event_id=str(uuid4()),
            occurred_at=datetime.now(UTC),
            user_id=USER,
            session_id=SESSION,
            after_event_seq=0,
            through_event_seq=1,
            limit=100,
        )
        store.admit(command.model_dump_json())
        store.replay(command)
        store.publication_delivered(original.id, original.dispatch_generation)
        current = store.pending_publications()[0]
        assert current.dispatch_generation == 2
        store.replay(command)
        assert store.pending_publications()[0].dispatch_generation == 2
        store.publication_delivered(current.id, current.dispatch_generation)
        assert not store.pending_publications()


def test_confirmed_execution_outcome_is_immutable_and_original_operation_is_retained(
    tmp_path,
):
    from lux_agent.store import ExecutionPreparation

    with store_at(tmp_path / "agent.db") as store:
        run = running().run
        store.register_run(USER, SESSION, run.id)
        call = str(uuid4())
        prepared = ExecutionPreparation(
            execution_id=str(uuid4()),
            kind="TOOL",
            user_id=USER,
            session_id=SESSION,
            run_id=run.id,
            turn_id=str(uuid4()),
            tool_call_item_id=call,
            operation_id=call,
            request_snapshot='{"request":"original"}',
        )
        store.prepare_execution(prepared)
        store.execution_started(prepared.execution_id)
        store.execution_confirmed(prepared.execution_id, '{"result":"accepted"}')
        with pytest.raises(ValueError, match="Conflicting"):
            store.execution_confirmed(prepared.execution_id, '{"result":"changed"}')
        assert store.execution(prepared.execution_id).operation_id == call


def test_exclusive_store_and_legacy_database_are_not_silently_taken_over(tmp_path):
    path = tmp_path / "agent.db"
    with store_at(path):
        with pytest.raises(RuntimeError, match="owned"):
            store_at(path)
    legacy = tmp_path / "legacy.db"
    connection = sqlite3.connect(legacy)
    connection.execute("CREATE TABLE original(value TEXT)")
    connection.execute("INSERT INTO original VALUES ('keep')")
    connection.execute("PRAGMA user_version=1")
    connection.commit()
    connection.close()
    with pytest.raises(RuntimeError, match="legacy|schema"):
        store_at(legacy)
    with closing(sqlite3.connect(legacy)) as connection:
        assert connection.execute("SELECT value FROM original").fetchone() == ("keep",)


def test_source_sync_reads_nested_observed_snapshots_and_requeues_query_once(tmp_path):
    from lux_agent.protocol import ModelResponseCompleted, QuerySessionExecution
    from lux_agent.store import AgentStore

    with AgentStore(tmp_path / "agent.db") as store:
        first = store.append_fact(intent(running()))
        turn = protocol.TurnIdentity(id=str(uuid4()), turn_index=1, run=first.body.run)
        item = protocol.AgentItemSnapshot(
            item=protocol.AgentItemIdentity(id=str(uuid4()), item_index=1, turn=turn),
            source_revision=1,
            kind="MESSAGE",
            status="COMPLETED",
            payload=protocol.OrderedContent(blocks=(protocol.TextBlock(text="done"),)),
            completed_at=datetime.now(UTC),
            call_id=None,
            tool_name=None,
        )
        completed = protocol.OutboundFactIntent(
            schema_version=2,
            event_id=str(uuid4()),
            occurred_at=datetime.now(UTC),
            user_id=USER,
            session_id=SESSION,
            event_type="MODEL_RESPONSE_COMPLETED",
            body=ModelResponseCompleted(
                model_execution_id="model-exec", turn=turn, items=(item,)
            ),
        )
        store.append_fact(completed)
        synced = store.sync(USER, SESSION, 0, None, 1)
        assert synced.event_high_watermark == 2
        assert len(synced.events) == 1 and synced.next_after_event_seq == 1
        assert synced.items == (item,)
        for publication in store.pending_publications():
            store.publication_delivered(publication.id, publication.dispatch_generation)
        query = QuerySessionExecution(
            schema_version=2,
            event_id=str(uuid4()),
            occurred_at=datetime.now(UTC),
            user_id=USER,
            session_id=SESSION,
            limit=100,
        )
        store.admit(query.model_dump_json())
        store.observe(query)
        generation = store.pending_publications()[0].dispatch_generation
        store.observe(query)
        assert store.pending_publications()[0].dispatch_generation == generation
        with pytest.raises(ValueError, match="owner"):
            store.sync(str(uuid4()), SESSION, 0, None, 100)


def test_envelope_budget_rejection_rolls_back_counter_event_and_publication(
    monkeypatch, tmp_path
):
    from lux_agent import store as storage
    from lux_agent.store import ExecutionPreparation

    with store_at(tmp_path / "agent.db") as store:
        initial = intent(running())
        store.register_run(USER, SESSION, initial.body.run.id)
        prepared = ExecutionPreparation(
            execution_id=str(uuid4()),
            kind="MODEL",
            user_id=USER,
            session_id=SESSION,
            run_id=initial.body.run.id,
            turn_id=str(uuid4()),
            request_snapshot='{"prompt":"original"}',
        )
        store.prepare_execution(prepared)
        store.execution_confirmed(
            prepared.execution_id, '{"response":"original observed response"}'
        )
        expected = protocol.ExecutionFact(**initial.model_dump(), event_seq=1)
        budget = len(expected.model_dump_json().encode("utf-8"))
        monkeypatch.setattr(storage, "MAX_EXECUTION_FACT_BYTES", budget, raising=False)
        store.append_fact(initial)
        oversized = initial.model_copy(
            update={
                "event_id": str(uuid4()),
                "body": initial.body.model_copy(update={"error_message": "éé"}),
            }
        )
        with pytest.raises(ValueError, match="envelope"):
            store.append_fact(oversized)
        assert store.high_water(SESSION) == 1
        assert len(store.events(SESSION)) == len(store.pending_publications()) == 1
        assert (
            store.execution(prepared.execution_id).response_snapshot
            == '{"response":"original observed response"}'
        )
