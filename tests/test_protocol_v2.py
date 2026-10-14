import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from lux_agent import protocol

USER = "00000000-0000-0000-0000-000000000001"
SESSION = "00000000-0000-0000-0000-000000000002"


def input_json(sequence: int = 1, text: str = " 你好\n ") -> str:
    identity = str(uuid4())
    return json.dumps(
        {
            "schema_version": 2,
            "event_id": identity,
            "event_type": "INPUT_ACCEPTED",
            "occurred_at": "2026-10-12T01:02:03.120000Z",
            "user_id": USER,
            "session_id": SESSION,
            "item": {
                "id": identity,
                "sequence_no": sequence,
                "input_sequence_no": sequence,
                "source": "USER",
                "kind": "MESSAGE",
                "content": {"blocks": [{"type": "text", "text": text}]},
            },
        },
        ensure_ascii=False,
    )


def test_session_input_has_no_java_selected_execution_and_preserves_content() -> None:
    command = protocol.decode_command(input_json().encode())
    assert command.event_type == "INPUT_ACCEPTED"
    assert command.event_id == command.item.id
    assert command.item.content.blocks[0].text == " 你好\n "
    assert "run_id" not in command.model_dump()


@pytest.mark.parametrize(
    "field,value", [("schema_version", True), ("run_id", str(uuid4()))]
)
def test_session_input_rejects_legacy_assignment_and_noninteger_version(
    field, value
) -> None:
    value_tree = json.loads(input_json())
    value_tree[field] = value
    with pytest.raises(ValidationError):
        protocol.decode_command(json.dumps(value_tree).encode())


def test_input_identity_is_its_event_identity() -> None:
    value_tree = json.loads(input_json())
    value_tree["item"]["id"] = str(uuid4())
    with pytest.raises(ValidationError):
        protocol.decode_command(json.dumps(value_tree).encode())


def test_fact_discriminator_selects_body_without_duplicate_body_type() -> None:
    raw = json.dumps(
        {
            "schema_version": 2,
            "event_id": str(uuid4()),
            "event_type": "RUN_SNAPSHOT",
            "occurred_at": "2026-10-12T01:02:03Z",
            "user_id": USER,
            "session_id": SESSION,
            "event_seq": 1,
            "body": {
                "run": {"id": str(uuid4()), "run_index": 1},
                "source_revision": 1,
                "status": "RUNNING",
                "started_at": None,
                "finished_at": None,
                "final_item": None,
                "error_code": None,
                "error_message": None,
            },
        }
    )
    fact = protocol.decode_fact(raw.encode())
    assert fact.body.status == "RUNNING"
    assert "type" not in fact.body.model_dump()
    with pytest.raises(ValidationError):
        protocol.decode_fact(
            raw.replace('"RUN_SNAPSHOT"', '"ITEM_TEXT_DELTA"').encode()
        )


def test_shared_fixtures_preserve_original_json_digests():
    from pathlib import Path

    from lux_agent.store import content_hash

    fixtures = Path(__file__).resolve().parents[2] / "lux-java-backend/docs/fixtures/v2"
    digests = json.loads((fixtures / "digests.json").read_text())
    for entry in digests:
        name, digest = entry["filename"], entry["sha256"]
        raw = (fixtures / name).read_text()
        assert content_hash(raw) == digest
        decoded = (
            protocol.decode_command(raw)
            if name == "input-accepted.json"
            else protocol.decode_fact(raw)
        )
        assert decoded.schema_version == 2


def test_source_times_reject_nonzero_submicrosecond_precision():
    from pathlib import Path

    raw = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "lux-java-backend/docs/fixtures/v2/run-snapshot.json"
        ).read_text()
    )
    raw["body"]["started_at"] = "2026-10-12T00:00:00.1234567Z"
    with pytest.raises(ValidationError):
        protocol.decode_fact(json.dumps(raw))
    raw["body"]["started_at"] = "2026-10-12T00:00:00.123456000Z"
    assert protocol.decode_fact(json.dumps(raw)).body.started_at.microsecond == 123456


@pytest.mark.parametrize("version", [0, True])
def test_task_query_version_requires_positive_integer(version):
    with pytest.raises(ValidationError):
        protocol.TaskQuerySnapshot(
            task_id=str(uuid4()),
            version=version,
            status="CREATED",
            assets=(),
            failure=None,
        )


@pytest.mark.parametrize("index", [-1, 2**31, True])
def test_asset_source_position_matches_java_int_range(index):
    with pytest.raises(ValidationError):
        protocol.TaskAssetReference(asset_id=str(uuid4()), source_index=index)


@pytest.mark.parametrize("model", ["RunSnapshot", "TurnSnapshot"])
def test_lifecycle_error_code_matches_java_storage_bound(model):
    run = protocol.RunIdentity(id=str(uuid4()), run_index=1)
    common = {
        "source_revision": 1,
        "status": "RUNNING",
        "started_at": None,
        "finished_at": None,
        "error_code": "x" * 65,
        "error_message": None,
    }
    arguments = (
        {"run": run, "final_item": None}
        if model == "RunSnapshot"
        else {"turn": protocol.TurnIdentity(id=str(uuid4()), turn_index=1, run=run)}
    )
    with pytest.raises(ValidationError):
        getattr(protocol, model)(**common, **arguments)


@pytest.mark.parametrize("code", ["", "  ", "\n\t"])
def test_tool_error_code_requires_nonblank_value(code):
    with pytest.raises(ValidationError):
        protocol.ToolError(code=code, message="failure")


@pytest.mark.parametrize("model", ["RunSnapshot", "TurnSnapshot"])
def test_lifecycle_error_message_enforces_utf8_text_capacity(model):
    run = protocol.RunIdentity(id=str(uuid4()), run_index=1)
    common = {
        "source_revision": 1,
        "status": "RUNNING",
        "started_at": None,
        "finished_at": None,
        "error_code": None,
        "error_message": "你" * 21846,
    }
    arguments = (
        {"run": run, "final_item": None}
        if model == "RunSnapshot"
        else {"turn": protocol.TurnIdentity(id=str(uuid4()), turn_index=1, run=run)}
    )
    with pytest.raises(ValidationError):
        getattr(protocol, model)(**common, **arguments)
    common["error_message"] = "a" * 65535
    assert getattr(protocol, model)(**common, **arguments).error_message == "a" * 65535


@pytest.mark.parametrize(
    "model", ["ExecutionFact", "OutboundFactIntent", "ObservedTextFragment"]
)
def test_execution_fact_times_reject_nonzero_submicrosecond_precision(model):
    from pathlib import Path

    raw = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "lux-java-backend/docs/fixtures/v2/item-delta.json"
        ).read_text()
    )
    if model == "OutboundFactIntent":
        raw.pop("event_seq")
    elif model == "ObservedTextFragment":
        body = raw.pop("body")
        raw.pop("event_seq")
        raw.pop("event_type")
        raw.update(item=body["item"], text=body["text"])
    raw["occurred_at"] = "2026-10-12T01:02:03.1234567Z"
    with pytest.raises(ValidationError):
        getattr(protocol, model).model_validate_json(json.dumps(raw))
    raw["occurred_at"] = "2026-10-12T01:02:03.123456000Z"
    assert (
        getattr(protocol, model)
        .model_validate_json(json.dumps(raw))
        .occurred_at.microsecond
        == 123456
    )


def test_input_original_time_metadata_retains_general_timestamp_contract(tmp_path):
    from lux_agent.store import AgentStore

    raw = json.loads(input_json())
    raw["occurred_at"] = "2026-10-12T01:02:03.1234567Z"
    envelope = json.dumps(raw)
    with AgentStore(tmp_path / "agent.db") as store:
        assert (
            json.loads(store.admit(envelope).envelope)["occurred_at"]
            == raw["occurred_at"]
        )
