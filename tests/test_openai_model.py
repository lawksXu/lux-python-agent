import asyncio
import json
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from test_protocol_v2 import SESSION, USER

from lux_agent.execution import (
    ExecutionUnknown,
    ModelRequest,
    ModelResponseObserved,
    ModelTextFragment,
)
from lux_agent.java_tools import model_tools
from lux_agent.protocol import GetTaskCallContent, RunIdentity, TurnIdentity
from lux_agent.store import AgentStore, ExecutionPreparation

TASK = "00000000-0000-0000-0000-000000000009"


def request():
    run = RunIdentity(id=str(uuid4()), run_index=1)
    return ModelRequest(
        execution_id=str(uuid4()),
        user_id=USER,
        session_id=SESSION,
        turn=TurnIdentity(id=str(uuid4()), turn_index=1, run=run),
        messages=(
            SystemMessage("Keep full context"),
            HumanMessage("old"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_task",
                        "args": {"task_id": TASK},
                        "id": "old-call",
                        "type": "tool_call",
                    }
                ],
                response_metadata={"id": "resp_old"},
            ),
            ToolMessage(content="original receipt", tool_call_id="old-call"),
            HumanMessage("new"),
        ),
        candidates=(),
        tools=model_tools(),
    )


def prepare(store, req):
    store.register_run(USER, SESSION, req.turn.run.id)
    store.prepare_execution(
        ExecutionPreparation(
            execution_id=req.execution_id,
            kind="MODEL",
            user_id=USER,
            session_id=SESSION,
            run_id=req.turn.run.id,
            turn_id=req.turn.id,
            request_snapshot=req.model_dump_json(),
        )
    )
    store.execution_started(req.execution_id)


def response(status="completed", output=None):
    return {
        "id": "resp_original",
        "object": "response",
        "created_at": 1,
        "status": status,
        "error": None,
        "incomplete_details": None,
        "model": "gpt-6-luna",
        "output": output or [],
        "parallel_tool_calls": True,
        "tools": [],
        "tool_choice": "auto",
        "temperature": 1.0,
        "top_p": 1.0,
        "metadata": {},
        "usage": None,
    }


def event(kind, **data):
    return (
        f"data: {json.dumps({'type': kind, 'sequence_number': 1, **data})}\n\n".encode()
    )


def stream_fixture(complete=True):
    reasoning = {"type": "reasoning", "id": "rs_original", "summary": []}
    call = {
        "type": "function_call",
        "id": "fc_original",
        "call_id": "call_original",
        "name": "get_task",
        "arguments": "",
        "status": "in_progress",
    }
    message = {
        "type": "message",
        "id": "msg_original",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Hello world", "annotations": []}],
    }
    chunks = [
        event("response.created", response=response("in_progress")),
        event("response.output_item.added", output_index=0, item=reasoning),
        event(
            "response.reasoning_summary_part.added",
            output_index=0,
            item_id="rs_original",
            summary_index=0,
            part={"type": "summary_text", "text": ""},
        ),
        event(
            "response.reasoning_summary_text.delta",
            output_index=0,
            item_id="rs_original",
            summary_index=0,
            delta="Think first",
        ),
        event(
            "response.output_text.delta",
            output_index=1,
            item_id="msg_original",
            content_index=0,
            delta="Hello ",
        ),
        event(
            "response.output_text.delta",
            output_index=1,
            item_id="msg_original",
            content_index=0,
            delta="world",
        ),
        event(
            "response.output_text.done",
            output_index=1,
            item_id="msg_original",
            content_index=0,
            text="Hello world",
        ),
        event("response.output_item.added", output_index=2, item=call),
        event(
            "response.function_call_arguments.delta",
            output_index=2,
            item_id="fc_original",
            delta='{"task_id":',
        ),
        event(
            "response.function_call_arguments.delta",
            output_index=2,
            item_id="fc_original",
            delta=json.dumps(TASK) + "}",
        ),
    ]
    final = response(
        output=[
            dict(reasoning, summary=[{"type": "summary_text", "text": "Think first"}]),
            message,
            dict(call, arguments=json.dumps({"task_id": TASK}), status="completed"),
        ]
    )
    if complete:
        chunks.append(event("response.completed", response=final))
    return chunks, final


class DisconnectStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        raise httpx.ReadError("offline disconnect")


def test_chatopenai_stream_receipt_before_fragments_and_full_context(tmp_path):
    async def scenario():
        from lux_agent.openai_model import OpenAIModelExecutor

        chunks, _ = stream_fixture()
        received = []

        def handler(req):
            received.append(req)
            return httpx.Response(
                200,
                content=b"".join(chunks),
                headers={"content-type": "text/event-stream"},
            )

        with AgentStore(tmp_path / "agent.db") as store:
            req = request()
            prepare(store, req)
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="test-account",
                    http_client=client,
                )
                emissions = []
                async for emission in executor.stream(req):
                    assert (
                        store.model_provider_receipt(req.execution_id).response_id
                        == "resp_original"
                    )
                    emissions.append(emission)
                observed = emissions[-1]
                assert isinstance(observed, ModelResponseObserved)
                assert [
                    e.text
                    for e in emissions
                    if isinstance(e, ModelTextFragment) and e.slot == "answer"
                ] == ["Hello ", "world"]
                assert [i.kind for i in observed.items] == [
                    "REASONING_SUMMARY",
                    "MESSAGE",
                    "TOOL_CALL",
                ]
                call = observed.items[-1]
                assert call.call_id == "call_original"
                assert isinstance(call.payload, GetTaskCallContent)
                assert call.payload.arguments.task_id == TASK
                assert observed.messages[0].tool_calls[0]["id"] == "call_original"
                assert observed.messages[0].id == "resp_original"
                assert received[0].headers["X-Client-Request-Id"] == req.execution_id
                body = json.loads(received[0].content)
                assert "extra_headers" not in body
                assert body["model"] == "gpt-6-luna"
                assert body["stream"] is True and body["store"] is True
                assert body["truncation"] == "disabled"
                assert "previous_response_id" not in body and "conversation" not in body
                assert len(body["input"]) == 5
                assert body["input"][3]["call_id"] == "old-call"
                assert {t["name"] for t in body["tools"]} == {t.name for t in req.tools}
                assert len(received) == 1
                assert "offline-test-key" not in (
                    tmp_path / "agent.db"
                ).read_bytes().decode(errors="ignore")

    asyncio.run(scenario())


def test_disconnect_reopens_original_receipt_and_only_retrieves(tmp_path):
    async def scenario():
        from lux_agent.openai_model import OpenAIModelExecutor

        chunks, final = stream_fixture(complete=False)
        received = []

        def handler(req):
            received.append((req.method, req.url.path))
            if req.method == "POST":
                return httpx.Response(
                    200,
                    stream=DisconnectStream(chunks),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(200, json=final)

        path = tmp_path / "agent.db"
        req = request()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with AgentStore(path) as store:
                prepare(store, req)
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="test-account",
                    http_client=client,
                )
                with pytest.raises(ExecutionUnknown):
                    async for _ in executor.stream(req):
                        pass
                store.execution_unknown(req.execution_id)
            with AgentStore(path) as store:
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="test-account",
                    http_client=client,
                )
                recovered = await executor.reconcile(req)
                assert recovered.items[-1].call_id == "call_original"
                assert recovered.items[-1].payload.arguments.task_id == TASK
                assert recovered.items[0].payload.blocks[0].text == "Think first"
                assert recovered.messages[0].id == "resp_original"
                assert received == [
                    ("POST", "/v1/responses"),
                    ("GET", "/v1/responses/resp_original"),
                ]

    asyncio.run(scenario())


def test_missing_response_id_and_account_mismatch_never_resubmit(tmp_path):
    async def scenario():
        from lux_agent.openai_model import OpenAIModelExecutor

        calls = []

        def handler(req):
            calls.append(req.method)
            raise httpx.ReadError("lost before response.created")

        with AgentStore(tmp_path / "agent.db") as store:
            req = request()
            prepare(store, req)
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="test-account",
                    http_client=client,
                )
                with pytest.raises(ExecutionUnknown):
                    async for _ in executor.stream(req):
                        pass
                assert await executor.reconcile(req) is None
                with pytest.raises(ExecutionUnknown):
                    async for _ in executor.stream(req):
                        pass
                other = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="other-account",
                    http_client=client,
                )
                with pytest.raises(ValueError, match="binding"):
                    await other.reconcile(req)
                assert calls == ["POST"]

    asyncio.run(scenario())


def test_complete_invalid_arguments_are_confirmed_failure(tmp_path):
    async def scenario():
        from lux_agent.execution import ConfirmedExecutionFailure
        from lux_agent.openai_model import OpenAIModelExecutor

        chunks, _ = stream_fixture()
        # Valid JSON is still invalid against the tool's known business schema.
        chunks = [c.replace(TASK.encode(), b"not-a-task-id") for c in chunks]
        with AgentStore(tmp_path / "agent.db") as store:
            req = request()
            prepare(store, req)
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(
                        200,
                        content=b"".join(chunks),
                        headers={"content-type": "text/event-stream"},
                    )
                )
            ) as client:
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="test-account",
                    http_client=client,
                )
                with pytest.raises(ConfirmedExecutionFailure) as error:
                    async for _ in executor.stream(req):
                        pass
                assert error.value.failure.code == "MODEL_RESPONSE_INVALID"

    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["failed", "cancelled", "incomplete"])
def test_original_terminal_provider_outcome_is_confirmed_failure(tmp_path, status):
    async def scenario():
        from lux_agent.execution import ConfirmedExecutionFailure
        from lux_agent.openai_model import OpenAIModelExecutor

        req = request()
        calls = []

        def handler(http_request):
            calls.append(http_request.method)
            return httpx.Response(200, json=response(status))

        with AgentStore(tmp_path / "agent.db") as store:
            prepare(store, req)
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="test-account",
                    http_client=client,
                )
                binding = executor._binding(req)
                store.prepare_model_provider(binding)
                store.observe_model_provider(binding, "resp_original")
                store.execution_unknown(req.execution_id)
                with pytest.raises(ConfirmedExecutionFailure) as error:
                    await executor.reconcile(req)
                store.execution_confirmed(
                    req.execution_id, error.value.failure.model_dump_json(), failed=True
                )
                assert store.execution(req.execution_id).state == "FAILED"
                assert calls == ["GET"]

    asyncio.run(scenario())


def test_existing_v2_schema_gets_additive_receipts_without_reset(tmp_path):
    import sqlite3

    from lux_agent.store import _SCHEMA

    path = tmp_path / "agent.db"
    with sqlite3.connect(path) as old:
        old.executescript(_SCHEMA)
        old.execute(
            "INSERT INTO run VALUES (?,?,?,?)",
            ("original-run", USER, SESSION, "original-time"),
        )
    with AgentStore(path) as store:
        assert store.session_ids() == (SESSION,)
        assert store.model_provider_receipt(str(uuid4())) is None
    with sqlite3.connect(path) as current:
        assert current.execute("SELECT run_id,created_at FROM run").fetchall() == [
            ("original-run", "original-time")
        ]
        assert current.execute("PRAGMA user_version").fetchone()[0] == 2


def test_native_checkpoint_restart_retrieves_original_and_completes_once(tmp_path):
    async def scenario():
        from test_protocol_v2 import input_json

        from lux_agent.openai_model import OpenAIModelExecutor
        from lux_agent.runtime import AgentRuntime

        final = response(
            output=[
                {
                    "type": "message",
                    "id": "msg_original",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Recovered answer",
                            "annotations": [],
                        }
                    ],
                }
            ]
        )
        received = []

        def handler(req):
            received.append((req.method, req.url.path))
            if req.method == "POST":
                return httpx.Response(
                    200,
                    stream=DisconnectStream(
                        [
                            event("response.created", response=response("in_progress")),
                            event(
                                "response.output_text.delta",
                                output_index=0,
                                item_id="msg_original",
                                content_index=0,
                                delta="Recovered ",
                            ),
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(200, json=final)

        path = tmp_path / "agent.db"
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with AgentStore(path) as store:
                store.admit(input_json())
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="test-account",
                    http_client=client,
                )
                async with AgentRuntime(store, executor=executor) as runtime:
                    await runtime.advance(SESSION)
                    state = await runtime.state(SESSION)
                    original_execution = state.active_turn.model_execution_id
                    assert store.execution(original_execution).state == "UNKNOWN"
                    assert (
                        store.model_provider_receipt(original_execution).response_id
                        == "resp_original"
                    )
            with AgentStore(path) as store:
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="test-account",
                    http_client=client,
                )
                async with AgentRuntime(
                    store, executor=executor, model_reconciler=executor
                ) as runtime:
                    await runtime.advance(SESSION)
                    assert (await runtime.state(SESSION)).runs[-1].status == "COMPLETED"
                    assert store.execution(original_execution).state == "CONFIRMED"
                    assert (
                        sum(
                            f.event_type == "MODEL_RESPONSE_COMPLETED"
                            for f in store.events(SESSION)
                        )
                        == 1
                    )
                    await runtime.advance(SESSION)
                    assert received == [
                        ("POST", "/v1/responses"),
                        ("GET", "/v1/responses/resp_original"),
                    ]

    asyncio.run(scenario())


def test_recovery_rejects_project_scope_change_before_network(tmp_path, monkeypatch):
    async def scenario():
        from lux_agent.execution import ExecutionUnknown
        from lux_agent.openai_model import OpenAIModelExecutor

        calls = []

        def handler(req):
            calls.append(req)
            return httpx.Response(
                200,
                stream=DisconnectStream(
                    [
                        event("response.created", response=response("in_progress")),
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )

        monkeypatch.setenv("OPENAI_PROJECT_ID", "project-original")
        with AgentStore(tmp_path / "agent.db") as store:
            req = request()
            prepare(store, req)
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="test-account",
                    http_client=client,
                )
                with pytest.raises(ExecutionUnknown):
                    async for _ in executor.stream(req):
                        pass
                monkeypatch.setenv("OPENAI_PROJECT_ID", "project-changed")
                other = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="test-account",
                    http_client=client,
                )
                with pytest.raises(ValueError, match="binding"):
                    await other.reconcile(req)
                assert len(calls) == 1
                assert calls[0].headers["OpenAI-Project"] == "project-original"

    asyncio.run(scenario())


def test_stream_incomplete_persists_failed_and_seals_original_partial(tmp_path):
    async def scenario():
        from test_protocol_v2 import input_json

        from lux_agent.openai_model import OpenAIModelExecutor
        from lux_agent.runtime import AgentRuntime

        calls = []

        def handler(req):
            calls.append(req.method)
            return httpx.Response(
                200,
                content=b"".join(
                    [
                        event("response.created", response=response("in_progress")),
                        event(
                            "response.output_text.delta",
                            output_index=0,
                            item_id="msg_original",
                            content_index=0,
                            delta="Partial",
                        ),
                        event("response.incomplete", response=response("incomplete")),
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )

        with AgentStore(tmp_path / "agent.db") as store:
            store.admit(input_json())
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="test-account",
                    http_client=client,
                )
                async with AgentRuntime(
                    store, executor=executor, model_reconciler=executor
                ) as runtime:
                    await runtime.advance(SESSION)
                    source = store.sync(USER, SESSION)
                    assert source.runs[-1].status == "FAILED"
                    assert source.runs[-1].error_code == "MODEL_RESPONSE_INCOMPLETE"
                    assert source.items[-1].status == "INTERRUPTED"
                    await runtime.advance(SESSION)
                    assert calls == ["POST"]

    asyncio.run(scenario())


@pytest.mark.parametrize("unavailable", [False, True])
def test_known_response_running_or_unavailable_stays_unknown(tmp_path, unavailable):
    async def scenario():
        from lux_agent.openai_model import OpenAIModelExecutor

        calls = []

        def handler(req):
            calls.append(req.method)
            if req.method == "POST":
                return httpx.Response(
                    200,
                    stream=DisconnectStream(
                        [
                            event("response.created", response=response("in_progress")),
                        ]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if unavailable:
                return httpx.Response(
                    404,
                    json={
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Unavailable",
                        }
                    },
                )
            return httpx.Response(200, json=response("in_progress"))

        with AgentStore(tmp_path / "agent.db") as store:
            req = request()
            prepare(store, req)
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="test-account",
                    http_client=client,
                )
                with pytest.raises(ExecutionUnknown):
                    async for _ in executor.stream(req):
                        pass
                store.execution_unknown(req.execution_id)
                assert await executor.reconcile(req) is None
                assert store.execution(req.execution_id).state == "UNKNOWN"
                assert calls == ["POST", "GET"]

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [400, 401, 403])
def test_definitive_create_rejection_is_confirmed_failed(tmp_path, status):
    async def scenario():
        from lux_agent.execution import ConfirmedExecutionFailure
        from lux_agent.openai_model import OpenAIModelExecutor

        calls = []

        def handler(http_request):
            calls.append(http_request.method)
            return httpx.Response(
                status,
                json={
                    "error": {
                        "type": "invalid_request_error",
                        "code": "context_length_exceeded",
                        "message": "Sensitive provider details",
                    }
                },
            )

        with AgentStore(tmp_path / "agent.db") as store:
            req = request()
            prepare(store, req)
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="test-account",
                    http_client=client,
                )
                with pytest.raises(ConfirmedExecutionFailure) as error:
                    async for _ in executor.stream(req):
                        pass
                assert error.value.failure.code == "MODEL_PROVIDER_REJECTED"
                assert "Sensitive provider details" not in error.value.failure.message
                assert (
                    store.model_provider_receipt(req.execution_id).response_id is None
                )
                assert calls == ["POST"]

    asyncio.run(scenario())


def test_rejected_original_does_not_block_later_session_input(tmp_path):
    async def scenario():
        from test_protocol_v2 import input_json

        from lux_agent.openai_model import OpenAIModelExecutor
        from lux_agent.runtime import AgentRuntime

        calls = []
        final = response(output=[])

        def handler(req):
            calls.append(req.method)
            if len(calls) == 1:
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "type": "invalid_request_error",
                            "code": "context_length_exceeded",
                            "message": "Rejected before acceptance",
                        }
                    },
                )
            return httpx.Response(
                200,
                content=b"".join(
                    [
                        event("response.created", response=response("in_progress")),
                        event(
                            "response.output_text.delta",
                            output_index=0,
                            item_id="msg_original",
                            content_index=0,
                            delta="New input completed",
                        ),
                        event("response.completed", response=final),
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )

        path = tmp_path / "agent.db"
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with AgentStore(path) as store:
                store.admit(input_json())
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="test-account",
                    http_client=client,
                )
                async with AgentRuntime(
                    store, executor=executor, model_reconciler=executor
                ) as runtime:
                    await runtime.advance(SESSION)
                    assert store.sync(USER, SESSION).runs[-1].status == "FAILED"
                    assert (
                        store.sync(USER, SESSION).runs[-1].error_code
                        == "MODEL_PROVIDER_REJECTED"
                    )
            with AgentStore(path) as store:
                store.admit(input_json(sequence=2, text="Later input"))
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="test-account",
                    http_client=client,
                )
                async with AgentRuntime(
                    store, executor=executor, model_reconciler=executor
                ) as runtime:
                    await runtime.advance(SESSION)
                    assert [run.status for run in store.sync(USER, SESSION).runs] == [
                        "FAILED",
                        "COMPLETED",
                    ]
                    await runtime.advance(SESSION)
                    assert calls == ["POST", "POST"]

    asyncio.run(scenario())


@pytest.mark.parametrize("accepted_id", [False, True])
def test_uncertain_or_already_accepted_error_remains_unknown(
    tmp_path, accepted_id, caplog
):
    async def scenario():
        from openai import APIStatusError

        from lux_agent.openai_model import OpenAIModelExecutor

        calls = []

        class AcceptedErrorStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield event("response.created", response=response("in_progress"))
                raise APIStatusError(
                    "Sensitive stream failure",
                    response=httpx.Response(
                        400,
                        request=calls[0],
                        headers={"X-Private-Header": "Sensitive header value"},
                    ),
                    body={"error": "Sensitive response body"},
                )

        def handler(req):
            calls.append(req)
            if accepted_id:
                return httpx.Response(
                    200,
                    stream=AcceptedErrorStream(),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                503,
                json={
                    "error": {
                        "type": "server_error",
                        "message": "Sensitive response body",
                    }
                },
                headers={"X-Private-Header": "Sensitive header value"},
            )

        with AgentStore(tmp_path / "agent.db") as store:
            req = request()
            prepare(store, req)
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="test-account",
                    http_client=client,
                )
                with pytest.raises(ExecutionUnknown):
                    async for _ in executor.stream(req):
                        pass
                saved = store.model_provider_receipt(req.execution_id)
                assert saved.response_id == ("resp_original" if accepted_id else None)
                assert len(calls) == 1
                diagnostics = [
                    record
                    for record in caplog.records
                    if record.name == "lux_agent.openai_model"
                ]
                assert len(diagnostics) == 1
                diagnostic = diagnostics[0]
                message = diagnostic.getMessage()
                assert req.execution_id in message
                assert (
                    "APIStatusError" if accepted_id else "OpenAIAPIError"
                ) in message
                assert f"status_code={400 if accepted_id else 503}" in message
                assert diagnostic.exc_info is None
                assert diagnostic.stack_info is None
                for sensitive in (
                    "Sensitive stream failure",
                    "Sensitive response body",
                    "Sensitive header value",
                    "offline-test-key",
                ):
                    assert sensitive not in caplog.text

    asyncio.run(scenario())


def test_transport_error_diagnostics_do_not_log_exception_details(tmp_path, caplog):
    async def scenario():
        from lux_agent.openai_model import OpenAIModelExecutor

        calls = []

        def handler(http_request):
            calls.append(http_request.method)
            raise httpx.ReadError("Sensitive transport message")

        with AgentStore(tmp_path / "agent.db") as store:
            req = request()
            prepare(store, req)
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-test-key",
                    account_ref="test-account",
                    http_client=client,
                )
                with pytest.raises(ExecutionUnknown):
                    async for _ in executor.stream(req):
                        pass
                store.execution_unknown(req.execution_id)
                assert await executor.reconcile(req) is None
                assert store.execution(req.execution_id).state == "UNKNOWN"
                assert calls == ["POST"]
                diagnostics = [
                    record
                    for record in caplog.records
                    if record.name == "lux_agent.openai_model"
                ]
                assert len(diagnostics) == 1
                diagnostic = diagnostics[0]
                assert req.execution_id in diagnostic.getMessage()
                assert "OpenAIConnectionError" in diagnostic.getMessage()
                assert "status_code=None" in diagnostic.getMessage()
                assert diagnostic.exc_info is None
                assert diagnostic.stack_info is None
                assert "Sensitive transport message" not in caplog.text
                assert "offline-test-key" not in caplog.text

    asyncio.run(scenario())
