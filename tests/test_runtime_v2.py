import asyncio
from uuid import uuid4

import pytest
from test_protocol_v2 import SESSION, USER, input_json

from lux_agent import protocol


class ReplyExecutor:
    def __init__(self):
        self.requests = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    @property
    def live_slots(self):
        from lux_agent.execution import LiveOutputSlot

        return (LiveOutputSlot(key="answer", kind="MESSAGE"),)

    async def stream(self, request):
        from lux_agent.execution import (
            ModelResponseObserved,
            ModelTextFragment,
            ObservedModelItem,
        )

        self.requests.append(request)
        self.started.set()
        await self.release.wait()
        yield ModelTextFragment(slot="answer", text="Hello ")
        yield ModelTextFragment(slot="answer", text="world")
        yield ModelResponseObserved(
            items=(
                ObservedModelItem(
                    slot="answer",
                    kind="MESSAGE",
                    payload=protocol.OrderedContent(
                        blocks=(protocol.TextBlock(text="Hello world"),)
                    ),
                ),
            ),
            provider_response={"id": "original-response"},
        )


def test_native_checkpoint_preserves_context_and_completed_response_on_restart(
    tmp_path,
):
    async def scenario():
        from lux_agent.runtime import AgentRuntime
        from lux_agent.store import AgentStore

        path = tmp_path / "agent.db"
        executor = ReplyExecutor()
        with AgentStore(path) as store:
            store.admit(input_json())
            async with AgentRuntime(store, executor=executor) as runtime:
                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                assert state.input_progress.incorporated_through == 1
                assert state.active_run is None
                assert state.runs[-1].status == "COMPLETED"
                assert [f.event_type for f in store.events(SESSION)].count(
                    "MODEL_RESPONSE_COMPLETED"
                ) == 1
                assert [
                    f.body.stream_seq
                    for f in store.events(SESSION)
                    if f.event_type == "ITEM_TEXT_DELTA"
                ] == [1, 2]
        with AgentStore(path) as store:
            async with AgentRuntime(store, executor=executor) as runtime:
                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                assert state.runs[-1].run.run_index == 1
                assert len(executor.requests) == 1
                assert (
                    store.command_state(store.commands(SESSION)[0].command.event_id)
                    == "APPLIED"
                )

    asyncio.run(scenario())


def test_steering_only_enters_the_next_model_turn(tmp_path):
    async def scenario():
        from lux_agent.runtime import AgentRuntime
        from lux_agent.store import AgentStore

        executor = ReplyExecutor()
        executor.release.clear()
        with AgentStore(tmp_path / "agent.db") as store:
            store.admit(input_json())
            async with AgentRuntime(store, executor=executor) as runtime:
                active = asyncio.create_task(runtime.advance(SESSION))
                await executor.started.wait()
                store.admit(input_json(2, "steer"))
                assert len(executor.requests[0].messages) == 1
                executor.release.set()
                await active
                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                assert len(executor.requests) == 2
                assert len(state.runs) == 1
                assert len(state.turns) == 2
                assert state.input_progress.incorporated_through == 2

    asyncio.run(scenario())


def test_stop_finishes_current_turn_and_later_input_wakes_a_new_run(tmp_path):
    async def scenario():
        from datetime import UTC, datetime

        from lux_agent.runtime import AgentRuntime
        from lux_agent.store import AgentStore

        executor = ReplyExecutor()
        executor.release.clear()
        with AgentStore(tmp_path / "agent.db") as store:
            store.admit(input_json())
            async with AgentRuntime(store, executor=executor) as runtime:
                active = asyncio.create_task(runtime.advance(SESSION))
                await executor.started.wait()
                stop = protocol.StopSession(
                    schema_version=2,
                    event_id=str(uuid4()),
                    occurred_at=datetime.now(UTC),
                    user_id=USER,
                    session_id=SESSION,
                )
                store.admit(stop.model_dump_json())
                store.admit(input_json(2, "after stop"))
                executor.release.set()
                await active
                assert len(executor.requests) == 1
                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                assert [run.status for run in state.runs] == ["CANCELLED", "COMPLETED"]
                assert [run.run.run_index for run in state.runs] == [1, 2]
                store.admit(stop.model_dump_json())
                await runtime.advance(SESSION)
                assert len(executor.requests) == 2

    asyncio.run(scenario())


def test_unknown_model_keeps_original_partial_item_and_blocks_new_wake(tmp_path):
    async def scenario():
        from lux_agent.execution import ModelTextFragment
        from lux_agent.runtime import AgentRuntime
        from lux_agent.store import AgentStore

        class UnknownExecutor(ReplyExecutor):
            async def stream(self, request):
                self.requests.append(request)
                yield ModelTextFragment(slot="answer", text="observed")
                raise OSError("outcome lost")

        path = tmp_path / "agent.db"
        executor = UnknownExecutor()
        with AgentStore(path) as store:
            store.admit(input_json())
            async with AgentRuntime(store, executor=executor) as runtime:
                await runtime.advance(SESSION)
                original = executor.requests[0]
                state = await runtime.state(SESSION)
                assert state.active_run.snapshot.status == "RUNNING"
                assert state.active_run.phase == "RECOVERY"
                assert state.active_turn.snapshot.status == "INTERRUPTED"
                assert store.sync(USER, SESSION).items[0].status == "PARTIAL"
                store.admit(input_json(2, "next"))
        with AgentStore(path) as store:
            async with AgentRuntime(store, executor=executor) as runtime:
                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                assert len(executor.requests) == 1
                assert state.active_turn.model_execution_id == original.execution_id
                assert state.pending_inputs[0].item.input_sequence_no == 2

    asyncio.run(scenario())


def test_gap_repair_freezes_all_inputs_in_context_order_without_java_run_claim(
    tmp_path,
):
    async def scenario():
        from lux_agent.runtime import AgentRuntime
        from lux_agent.store import AgentStore

        originals = (
            input_json(1, "first"),
            input_json(2, "second"),
            input_json(3, "third"),
        )

        class Repair:
            def __init__(self):
                self.calls = []

            async def read_range(self, owner, session, after, through, limit):
                self.calls.append((after, through))
                return tuple(
                    raw
                    for raw in originals
                    if after
                    < protocol.decode_command(raw).item.input_sequence_no
                    <= through
                )

        repair = Repair()
        executor = ReplyExecutor()
        with AgentStore(tmp_path / "agent.db") as store:
            store.admit(originals[2])
            async with AgentRuntime(store, executor=executor, repair=repair) as runtime:
                await runtime.advance(SESSION)
                assert [
                    message.content for message in executor.requests[0].messages
                ] == ["first", "second", "third"]
                assert (
                    await runtime.state(SESSION)
                ).input_progress.incorporated_through == 3
                assert repair.calls == [(0, 3)]

    asyncio.run(scenario())


def test_stop_at_idle_keeps_unconsumed_inputs_and_does_not_create_empty_run(tmp_path):
    async def scenario():
        from datetime import UTC, datetime

        from lux_agent.runtime import AgentRuntime
        from lux_agent.store import AgentStore

        stop = protocol.StopSession(
            schema_version=2,
            event_id=str(uuid4()),
            occurred_at=datetime.now(UTC),
            user_id=USER,
            session_id=SESSION,
        )
        executor = ReplyExecutor()
        with AgentStore(tmp_path / "agent.db") as store:
            store.admit(input_json())
            store.admit(stop.model_dump_json())
            async with AgentRuntime(store, executor=executor) as runtime:
                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                assert not state.runs and not executor.requests
                assert len(state.pending_inputs) == 1
                store.admit(input_json(2, "post-stop"))
                await runtime.advance(SESSION)
                assert [m.content for m in executor.requests[0].messages] == [
                    " 你好\n ",
                    "post-stop",
                ]

    asyncio.run(scenario())


def test_confirmed_failure_seals_observed_stream_and_unconfigured_model_is_explicit(
    tmp_path,
):
    async def scenario():
        from lux_agent.execution import ConfirmedExecutionFailure, ModelTextFragment
        from lux_agent.runtime import AgentRuntime
        from lux_agent.store import AgentStore

        class FailedExecutor(ReplyExecutor):
            async def stream(self, request):
                self.requests.append(request)
                yield ModelTextFragment(slot="answer", text="observed")
                raise ConfirmedExecutionFailure(
                    "MODEL_REJECTED", "original response failed"
                )

        for index, executor in enumerate((None, FailedExecutor())):
            with AgentStore(tmp_path / f"agent-{index}.db") as store:
                store.admit(input_json())
                async with AgentRuntime(store, executor=executor) as runtime:
                    await runtime.advance(SESSION)
                    state = await runtime.state(SESSION)
                    assert state.runs[-1].status == "FAILED"
                    if executor:
                        assert (
                            store.sync(USER, SESSION).items[0].status == "INTERRUPTED"
                        )
                    else:
                        assert state.runs[-1].error_code == "MODEL_NOT_CONFIGURED"

    asyncio.run(scenario())


class ToolBatchExecutor(ReplyExecutor):
    async def stream(self, request):
        from lux_agent.execution import ModelResponseObserved, ObservedModelItem

        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelResponseObserved(
                items=tuple(
                    ObservedModelItem(
                        kind="TOOL_CALL",
                        payload=protocol.SubmitGenerationCallContent(
                            arguments=protocol.SubmitGenerationArguments(
                                kind="IMAGE",
                                request=protocol.ConnectorRequest(
                                    prompt=f"image-{index}"
                                ),
                            )
                        ),
                        call_id=f"provider-call-{index}",
                        tool_name="submit_generation",
                    )
                    for index in range(2)
                ),
                provider_response={"response": "calls"},
            )
        else:
            yield ModelResponseObserved(
                items=(
                    ObservedModelItem(
                        kind="MESSAGE",
                        payload=protocol.OrderedContent(
                            blocks=(protocol.TextBlock(text="done"),)
                        ),
                    ),
                ),
                provider_response={"response": "final"},
            )


def test_parallel_tools_finish_batch_before_steer_or_stop_and_preserve_call_ids(
    tmp_path,
):
    async def scenario():
        from datetime import UTC, datetime

        from lux_agent.runtime import AgentRuntime, validate_context
        from lux_agent.store import AgentStore

        class Tools:
            def __init__(self):
                self.requests = []
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def execute(self, request):
                self.requests.append(request)
                if len(self.requests) == 2:
                    self.started.set()
                await self.release.wait()
                return protocol.TaskSubmissionReceipt(task_id=str(uuid4()))

        executor, tools = ToolBatchExecutor(), Tools()
        with AgentStore(tmp_path / "agent.db") as store:
            store.admit(input_json())
            async with AgentRuntime(store, executor=executor, tools=tools) as runtime:
                task = asyncio.create_task(runtime.advance(SESSION))
                await tools.started.wait()
                assert [fact.event_type for fact in store.events(SESSION)].count(
                    "MODEL_RESPONSE_COMPLETED"
                ) == 1
                store.admit(
                    protocol.StopSession(
                        schema_version=2,
                        event_id=str(uuid4()),
                        occurred_at=datetime.now(UTC),
                        user_id=USER,
                        session_id=SESSION,
                    ).model_dump_json()
                )
                store.admit(input_json(2, "steer after stop"))
                assert len(executor.requests) == 1
                tools.release.set()
                await task
                state = await runtime.state(SESSION)
                assert state.runs[-1].status == "CANCELLED"
                results = [
                    snapshot
                    for snapshot in state.items
                    if snapshot.kind == "TOOL_RESULT"
                ]
                assert {snapshot.call_id for snapshot in results} == {
                    "provider-call-0",
                    "provider-call-1",
                }
                assert all(
                    request.operation_id == request.call.item.id
                    for request in tools.requests
                )
                validate_context(state.messages)
                await runtime.advance(SESSION)
                assert len(executor.requests) == 2 and len(tools.requests) == 2
                assert (await runtime.state(SESSION)).runs[-1].run.run_index == 2

    asyncio.run(scenario())


def test_unknown_tool_beats_fatal_failure_and_reconciles_without_reexecuting_batch(
    tmp_path,
):
    async def scenario():
        from lux_agent.execution import ConfirmedExecutionFailure
        from lux_agent.runtime import AgentRuntime, validate_context
        from lux_agent.store import AgentStore

        class Tools:
            def __init__(self):
                self.requests = []

            async def execute(self, request):
                self.requests.append(request)
                if request.call.call_id == "provider-call-0":
                    raise ConfirmedExecutionFailure("TOOL_FATAL", "confirmed failure")
                raise OSError("submit outcome unknown")

        class Reconciler:
            def __init__(self):
                self.ready = False

            async def reconcile(self, request):
                return (
                    protocol.TaskSubmissionReceipt(task_id=SESSION)
                    if self.ready
                    else None
                )

        executor, tools, reconciler = ToolBatchExecutor(), Tools(), Reconciler()
        with AgentStore(tmp_path / "agent.db") as store:
            store.admit(input_json())
            async with AgentRuntime(
                store, executor=executor, tools=tools, tool_reconciler=reconciler
            ) as runtime:
                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                assert state.active_run.phase == "RECOVERY"
                assert state.active_run.snapshot.status == "RUNNING"
                assert state.active_turn.fatal_error.code == "TOOL_FATAL"
                store.admit(input_json(2, "next"))
                await runtime.advance(SESSION)
                assert len(executor.requests) == 1 and len(tools.requests) == 2
                reconciler.ready = True
                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                assert state.runs[-1].status == "FAILED"
                validate_context(state.messages)
                await runtime.advance(SESSION)
                assert len(executor.requests) == 2 and len(tools.requests) == 2
                assert (await runtime.state(SESSION)).runs[-1].status == "COMPLETED"

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["timeout", "unavailable"])
def test_get_task_read_failure_settles_and_allows_stop_and_later_input(
    failure, tmp_path
):
    async def scenario():
        from datetime import UTC, datetime

        import httpx

        from lux_agent.backend import BackendClient
        from lux_agent.runtime import AgentRuntime, validate_context
        from lux_agent.store import AgentStore

        class ReadTaskExecutor(ReplyExecutor):
            async def stream(self, request):
                from lux_agent.execution import ModelResponseObserved, ObservedModelItem

                self.requests.append(request)
                if len(self.requests) == 1:
                    yield ModelResponseObserved(
                        items=(
                            ObservedModelItem(
                                kind="TOOL_CALL",
                                payload=protocol.GetTaskCallContent(
                                    arguments=protocol.GetTaskArguments(task_id=SESSION)
                                ),
                                call_id="provider-read-call",
                                tool_name="get_task",
                            ),
                        ),
                        provider_response={"id": "read-call-response"},
                    )
                    return
                yield ModelResponseObserved(
                    items=(
                        ObservedModelItem(
                            kind="MESSAGE",
                            payload=protocol.OrderedContent(
                                blocks=(protocol.TextBlock(text="read complete"),)
                            ),
                        ),
                    ),
                    provider_response={"id": f"response-{len(self.requests)}"},
                )

        read_started = asyncio.Event()
        release_read = asyncio.Event()
        incoming_requests = []

        async def handler(incoming):
            incoming_requests.append(incoming)
            read_started.set()
            await release_read.wait()
            if failure == "timeout":
                raise httpx.ReadTimeout("response timed out", request=incoming)
            return httpx.Response(503, request=incoming)

        executor = ReadTaskExecutor()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://backend"
        ) as client:

            class Tools:
                def __init__(self):
                    self.requests = []

                async def execute(self, request):
                    self.requests.append(request)
                    return await BackendClient(client).execute(request)

            tools = Tools()
            with AgentStore(tmp_path / f"{failure}.db") as store:
                store.admit(input_json())
                async with AgentRuntime(
                    store, executor=executor, tools=tools
                ) as runtime:
                    active = asyncio.create_task(runtime.advance(SESSION))
                    await read_started.wait()
                    stop = protocol.StopSession(
                        schema_version=2,
                        event_id=str(uuid4()),
                        occurred_at=datetime.now(UTC),
                        user_id=USER,
                        session_id=SESSION,
                    )
                    store.admit(stop.model_dump_json())
                    store.admit(input_json(2, "after stopped turn"))
                    release_read.set()
                    await active

                    state = await runtime.state(SESSION)
                    assert [run.status for run in state.runs] == ["CANCELLED"]
                    tool_result = next(
                        item for item in state.items if item.kind == "TOOL_RESULT"
                    )
                    assert tool_result.call_id == "provider-read-call"
                    assert isinstance(tool_result.payload, protocol.ToolError)
                    assert tool_result.payload.code == "TASK_READ_UNAVAILABLE"
                    assert len(incoming_requests) == len(tools.requests) == 1
                    assert (
                        store.execution(tools.requests[0].execution_id).state
                        == "CONFIRMED"
                    )
                    validate_context(state.messages)

                    await runtime.advance(SESSION)
                    state = await runtime.state(SESSION)
                    assert [run.status for run in state.runs] == [
                        "CANCELLED",
                        "COMPLETED",
                    ]
                    assert state.input_progress.incorporated_through == 2
                    assert len(executor.requests) == 2
                    assert len(tools.requests) == 1

    asyncio.run(scenario())


def test_interrupted_get_task_is_settled_after_checkpoint_reopen_without_new_read(
    tmp_path,
):
    async def scenario():
        from datetime import UTC, datetime

        from lux_agent.execution import ModelResponseObserved, ObservedModelItem
        from lux_agent.runtime import AgentRuntime, validate_context
        from lux_agent.store import AgentStore

        class ReadTaskExecutor(ReplyExecutor):
            async def stream(self, request):
                self.requests.append(request)
                if len(self.requests) == 1:
                    yield ModelResponseObserved(
                        items=(
                            ObservedModelItem(
                                kind="TOOL_CALL",
                                payload=protocol.GetTaskCallContent(
                                    arguments=protocol.GetTaskArguments(task_id=SESSION)
                                ),
                                call_id="interrupted-read-call",
                                tool_name="get_task",
                            ),
                        ),
                        provider_response={"id": "original-read-response"},
                    )
                    return
                yield ModelResponseObserved(
                    items=(
                        ObservedModelItem(
                            kind="MESSAGE",
                            payload=protocol.OrderedContent(
                                blocks=(protocol.TextBlock(text="continued"),)
                            ),
                        ),
                    ),
                    provider_response={"id": f"response-{len(self.requests)}"},
                )

        class Tools:
            def __init__(self, interrupted):
                self.interrupted = interrupted
                self.requests = []

            async def execute(self, request):
                self.requests.append(request)
                if self.interrupted:
                    raise OSError("read interrupted before a snapshot arrived")
                return protocol.TaskQuerySnapshot(
                    task_id=SESSION,
                    version=1,
                    status="RUNNING",
                    assets=(),
                    failure=None,
                )

        class Reconciler:
            def __init__(self):
                self.requests = []

            async def reconcile(self, request):
                self.requests.append(request)
                return protocol.TaskQuerySnapshot(
                    task_id=SESSION,
                    version=2,
                    status="SUCCEEDED",
                    assets=(),
                    failure=None,
                )

        path = tmp_path / "interrupted-read.db"
        executor, original_tools = ReadTaskExecutor(), Tools(interrupted=True)
        with AgentStore(path) as store:
            store.admit(input_json())
            async with AgentRuntime(
                store, executor=executor, tools=original_tools
            ) as runtime:
                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                binding = state.active_turn.tools[0]
                assert store.execution(binding.execution_id).state == "UNKNOWN"
                assert binding.result is None
                store.admit(
                    protocol.StopSession(
                        schema_version=2,
                        event_id=str(uuid4()),
                        occurred_at=datetime.now(UTC),
                        user_id=USER,
                        session_id=SESSION,
                    ).model_dump_json()
                )
                store.admit(input_json(2, "after restart"))

        fresh_tools, reconciler = Tools(interrupted=False), Reconciler()
        with AgentStore(path) as store:
            async with AgentRuntime(
                store,
                executor=executor,
                tools=fresh_tools,
                tool_reconciler=reconciler,
            ) as runtime:
                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                assert [run.status for run in state.runs] == ["CANCELLED"]
                result = next(
                    item
                    for item in state.items
                    if item.item.id == binding.result_item.id
                )
                assert result.call_id == "interrupted-read-call"
                assert isinstance(result.payload, protocol.ToolError)
                assert result.payload.code == "TASK_READ_INTERRUPTED"
                assert fresh_tools.requests == []
                assert reconciler.requests == []
                assert store.execution(binding.execution_id).state == "CONFIRMED"
                validate_context(state.messages)

                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                assert [run.status for run in state.runs] == [
                    "CANCELLED",
                    "COMPLETED",
                ]
                assert state.input_progress.incorporated_through == 2
                assert len(executor.requests) == 2

    asyncio.run(scenario())


def test_completed_receipt_survives_checkpoint_failure_without_second_model_call(
    tmp_path,
):
    async def scenario():
        from lux_agent.runtime import AgentRuntime
        from lux_agent.store import AgentStore

        class CrashRuntime(AgentRuntime):
            async def _complete_model(self, graph_state):
                raise OSError("checkpoint boundary crash")

        path, executor = tmp_path / "agent.db", ReplyExecutor()
        with AgentStore(path) as store:
            store.admit(input_json())
            async with CrashRuntime(store, executor=executor) as runtime:
                try:
                    await runtime.advance(SESSION)
                except OSError:
                    pass
                state = await runtime.state(SESSION)
                assert (
                    store.execution(state.active_turn.model_execution_id).state
                    == "CONFIRMED"
                )
                item_id = executor.requests[0].candidates[0].item.id
        with AgentStore(path) as store:
            async with AgentRuntime(store, executor=executor) as runtime:
                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                assert len(executor.requests) == 1
                assert state.items[-1].item.id == item_id
                assert state.runs[-1].status == "COMPLETED"
                assert (
                    len(
                        [
                            fact
                            for fact in store.events(SESSION)
                            if fact.event_type == "MODEL_RESPONSE_COMPLETED"
                        ]
                    )
                    == 1
                )

    asyncio.run(scenario())


def test_native_driver_is_exclusive_even_within_one_process(tmp_path):
    async def scenario():
        import pytest

        from lux_agent.runtime import AgentRuntime
        from lux_agent.store import AgentStore

        with AgentStore(tmp_path / "agent.db") as store:
            async with AgentRuntime(store):
                with pytest.raises(RuntimeError, match="driver"):
                    async with AgentRuntime(store):
                        pass

    asyncio.run(scenario())


def test_dispatcher_wakes_checkpointed_post_stop_input_without_another_command(
    tmp_path,
):
    async def scenario():
        from datetime import UTC, datetime

        from lux_agent.runtime import AgentRuntime
        from lux_agent.store import AgentStore

        executor = ReplyExecutor()
        executor.release.clear()
        with AgentStore(tmp_path / "agent.db") as store:
            store.admit(input_json())
            async with AgentRuntime(store, executor=executor) as runtime:
                await runtime.start()
                await executor.started.wait()
                store.admit(
                    protocol.StopSession(
                        schema_version=2,
                        event_id=str(uuid4()),
                        occurred_at=datetime.now(UTC),
                        user_id=USER,
                        session_id=SESSION,
                    ).model_dump_json()
                )
                store.admit(input_json(2, "post-stop"))
                executor.release.set()
                async with asyncio.timeout(1):
                    while len(executor.requests) < 2:
                        await asyncio.sleep(0.01)
                assert len(executor.requests) == 2

    asyncio.run(scenario())


def test_task_notice_automatically_wakes_new_run_and_uses_developer_projection(
    tmp_path,
):
    async def scenario():
        import json

        from langchain_core.messages import convert_to_openai_messages

        from lux_agent.runtime import AgentRuntime
        from lux_agent.store import AgentStore

        executor = ReplyExecutor()
        with AgentStore(tmp_path / "agent.db") as store:
            store.admit(input_json())
            async with AgentRuntime(store, executor=executor) as runtime:
                await runtime.advance(SESSION)
                raw = json.loads(input_json(2))
                raw["item"].update(
                    source="SYSTEM",
                    kind="TASK_EVENT",
                    content={"task_id": str(uuid4())},
                )
                store.admit(json.dumps(raw))
                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                assert len(state.runs) == 2
                projected = convert_to_openai_messages(executor.requests[-1].messages)
                assert projected[-1]["role"] == "developer"
                assert json.loads(projected[-1]["content"]) == raw["item"]["content"]

    asyncio.run(scenario())


def test_repair_validates_session_and_contiguous_page_before_durable_admission(
    tmp_path,
):
    async def scenario():
        import json

        import pytest

        from lux_agent.runtime import AgentRuntime
        from lux_agent.store import AgentStore

        other = json.loads(input_json(1))
        other["session_id"] = str(uuid4())

        class Repair:
            async def read_range(self, owner, session, after, through, limit):
                return (json.dumps(other),)

        with AgentStore(tmp_path / "agent.db") as store:
            store.admit(input_json(2))
            async with AgentRuntime(
                store, executor=ReplyExecutor(), repair=Repair()
            ) as runtime:
                with pytest.raises(ValueError, match="Session"):
                    await runtime.advance(SESSION)
                assert other["session_id"] not in store.session_ids()

    asyncio.run(scenario())


def test_tool_batch_persistence_failure_drains_other_inflight_calls_before_return(
    tmp_path,
):
    async def scenario():
        import pytest

        from lux_agent.runtime import AgentRuntime
        from lux_agent.store import AgentStore

        class Store(AgentStore):
            def execution_started(self, execution_id):
                receipt = self.execution(execution_id)
                if (
                    receipt.preparation.kind == "TOOL"
                    and "provider-call-0" in receipt.preparation.request_snapshot
                ):
                    raise OSError("receipt disk failure")
                super().execution_started(execution_id)

        class Tools:
            def __init__(self):
                self.finished = False

            async def execute(self, request):
                await asyncio.sleep(0.05)
                self.finished = True
                return protocol.TaskSubmissionReceipt(task_id=str(uuid4()))

        tools = Tools()
        with Store(tmp_path / "agent.db") as store:
            store.admit(input_json())
            async with AgentRuntime(
                store, executor=ToolBatchExecutor(), tools=tools
            ) as runtime:
                with pytest.raises(OSError):
                    await runtime.advance(SESSION)
                assert tools.finished

    asyncio.run(scenario())


def test_unknown_model_reconciles_confirmed_failure_after_restart_and_honors_later_wake(
    tmp_path,
):
    async def scenario():
        from datetime import UTC, datetime

        from lux_agent.execution import ConfirmedExecutionFailure, ModelTextFragment
        from lux_agent.runtime import AgentRuntime
        from lux_agent.store import AgentStore

        class Unknown(ReplyExecutor):
            async def stream(self, request):
                self.requests.append(request)
                yield ModelTextFragment(slot="answer", text="observed original")
                raise OSError("outcome unknown")

        class FailedReconciler:
            async def reconcile(self, request):
                raise ConfirmedExecutionFailure(
                    "ORIGINAL_MODEL_FAILED", "original failure confirmed"
                )

        path, original = tmp_path / "agent.db", Unknown()
        with AgentStore(path) as store:
            store.admit(input_json())
            async with AgentRuntime(store, executor=original) as runtime:
                await runtime.advance(SESSION)
                old = await runtime.state(SESSION)
                model_id = old.active_turn.model_execution_id
                item_id = original.requests[0].candidates[0].item.id
                store.admit(
                    protocol.StopSession(
                        schema_version=2,
                        event_id=str(uuid4()),
                        occurred_at=datetime.now(UTC),
                        user_id=USER,
                        session_id=SESSION,
                    ).model_dump_json()
                )
                store.admit(input_json(2, "later input"))
        fresh = ReplyExecutor()
        with AgentStore(path) as store:
            async with AgentRuntime(
                store, executor=fresh, model_reconciler=FailedReconciler()
            ) as runtime:
                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                assert store.execution(model_id).state == "FAILED"
                assert state.runs[-1].status == "FAILED"
                assert not fresh.requests
                interrupted = next(
                    item
                    for item in store.sync(USER, SESSION).items
                    if item.item.id == item_id
                )
                assert interrupted.status == "INTERRUPTED"
                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                assert [run.status for run in state.runs] == ["FAILED", "COMPLETED"]
                assert fresh.requests[0].execution_id != model_id
                assert len(original.requests) == len(fresh.requests) == 1

    asyncio.run(scenario())


def test_unknown_tool_reconciles_failure_after_restart_and_pairs_original_call(
    tmp_path,
):
    async def scenario():
        from lux_agent.execution import ConfirmedExecutionFailure
        from lux_agent.runtime import AgentRuntime, validate_context
        from lux_agent.store import AgentStore

        class Tools:
            def __init__(self):
                self.requests = []

            async def execute(self, request):
                self.requests.append(request)
                if request.call.call_id == "provider-call-0":
                    return protocol.TaskSubmissionReceipt(task_id=SESSION)
                raise OSError("original outcome unknown")

        class FailedReconciler:
            async def reconcile(self, request):
                raise ConfirmedExecutionFailure(
                    "ORIGINAL_TOOL_FAILED", "original tool failure confirmed"
                )

        path, executor, tools = tmp_path / "agent.db", ToolBatchExecutor(), Tools()
        with AgentStore(path) as store:
            store.admit(input_json())
            async with AgentRuntime(store, executor=executor, tools=tools) as runtime:
                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                unknown = next(
                    binding
                    for binding in state.active_turn.tools
                    if binding.result is None
                )
                store.admit(input_json(2, "later input"))
        with AgentStore(path) as store:
            async with AgentRuntime(
                store,
                executor=executor,
                tools=tools,
                tool_reconciler=FailedReconciler(),
            ) as runtime:
                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                assert store.execution(unknown.execution_id).state == "FAILED"
                assert state.runs[-1].status == "FAILED"
                result = next(
                    item
                    for item in state.items
                    if item.item.id == unknown.result_item.id
                )
                assert result.call_id == unknown.call.call_id
                assert result.payload.code == "ORIGINAL_TOOL_FAILED"
                validate_context(state.messages)
                await runtime.advance(SESSION)
                assert (await runtime.state(SESSION)).runs[-1].status == "COMPLETED"
                assert len(tools.requests) == 2
                assert len(executor.requests) == 2

    asyncio.run(scenario())


def test_oversized_complete_fact_retains_receipt_and_checkpoint_intent_without_tools(
    monkeypatch, tmp_path
):
    async def scenario():
        import pytest

        from lux_agent import store as storage
        from lux_agent.execution import ModelResponseObserved, ObservedModelItem
        from lux_agent.runtime import AgentRuntime
        from lux_agent.store import AgentStore

        class LargeExecutor(ReplyExecutor):
            async def stream(self, request):
                self.requests.append(request)
                yield ModelResponseObserved(
                    items=(
                        ObservedModelItem(
                            kind="MESSAGE",
                            payload=protocol.OrderedContent(
                                blocks=(protocol.TextBlock(text="你" * 2000),)
                            ),
                        ),
                        ObservedModelItem(
                            kind="TOOL_CALL",
                            call_id="original-call",
                            tool_name="submit_generation",
                            payload=protocol.SubmitGenerationCallContent(
                                arguments=protocol.SubmitGenerationArguments(
                                    kind="IMAGE",
                                    request=protocol.ConnectorRequest(prompt="draw"),
                                )
                            ),
                        ),
                    ),
                    provider_response={"original": "你" * 2000},
                )

        class Tools:
            def __init__(self):
                self.requests = []

            async def execute(self, request):
                self.requests.append(request)
                return protocol.TaskSubmissionReceipt(task_id=SESSION)

        monkeypatch.setattr(storage, "MAX_EXECUTION_FACT_BYTES", 1500, raising=False)
        executor, tools = LargeExecutor(), Tools()
        with AgentStore(tmp_path / "agent.db") as store:
            store.admit(input_json())
            async with AgentRuntime(store, executor=executor, tools=tools) as runtime:
                with pytest.raises(ValueError, match="envelope"):
                    await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                assert state.active_turn.model_completed
                assert state.pending_events[-1].event_type == "MODEL_RESPONSE_COMPLETED"
                original = store.execution(executor.requests[0].execution_id)
                assert (
                    original.state == "CONFIRMED" and "你" in original.response_snapshot
                )
                head = store.high_water(SESSION)
                with pytest.raises(ValueError, match="envelope"):
                    await runtime.advance(SESSION)
                assert store.high_water(SESSION) == head
                assert len(executor.requests) == 1 and not tools.requests

    asyncio.run(scenario())
