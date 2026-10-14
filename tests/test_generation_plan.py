"""Regression coverage for strict shared plan wire values and runtime boundaries."""

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from lux_agent.generation_plan import (
    JavaPlanToolInvocation,
    SubmitGenerationPlanArguments,
)


def test_missing_focus_revision_is_rejected():
    with pytest.raises(ValidationError):
        SubmitGenerationPlanArguments.model_validate(
            {
                "plan_id": "00000000-0000-0000-0000-000000000001",
                "expected_plan_revision": 1,
            }
        )


def test_shared_invocations_round_trip():
    examples = fixtures()
    adapter = TypeAdapter(JavaPlanToolInvocation)
    for example in examples:
        value = example["value"]
        if value.get("type") == "generation_plan_tool_invocation":
            decoded = adapter.validate_json(json.dumps(value))
            assert json.loads(decoded.model_dump_json()) == value


def test_literal_integer_version_rejects_boolean():
    from lux_agent.generation_plan import JavaPlanToolOperationReceipt

    value = next(
        example["value"]
        for example in fixtures()
        if example["value"].get("type") == "generation_plan_tool_operation_receipt"
    )
    value["schema_version"] = True
    with pytest.raises(ValidationError):
        JavaPlanToolOperationReceipt.model_validate_json(json.dumps(value))


@pytest.mark.parametrize("media", ["image", "video"])
@pytest.mark.parametrize("invalid_count", [True, 1.0])
def test_result_file_count_requires_integer_literal(media, invalid_count):
    from lux_agent.generation_plan import NormalizedGenerationRequest

    value = next(
        example["value"]
        for example in fixtures()
        if example["value"].get("request_type") == f"{media}_generation_plan_v1"
    )
    value["result_expectations"]["file_count"] = invalid_count
    with pytest.raises(ValidationError):
        TypeAdapter(NormalizedGenerationRequest).validate_json(json.dumps(value))


def fixtures():
    path = (
        Path(__file__).resolve().parents[2]
        / "lux-java-backend"
        / "docs"
        / "fixtures"
        / "generation-plan-v1"
        / "examples.json"
    )
    return json.loads(path.read_text())


def test_all_shared_examples_round_trip():
    from lux_agent import generation_plan as plans

    for example in fixtures():
        value = example["value"]
        if value.get("type") == "generation_plan_tool_invocation":
            adapter = plans.INVOCATION_ADAPTER
        elif value.get("type") == "generation_plan_tool_operation_receipt":
            adapter = TypeAdapter(plans.JavaPlanToolOperationReceipt)
        elif value.get("type") == "generation_plan":
            adapter = TypeAdapter(plans.GenerationPlan)
        elif value.get("type") == "model_catalog_snapshot":
            adapter = plans.RESULT_ADAPTER
        else:
            adapter = TypeAdapter(plans.NormalizedGenerationRequest)
        decoded = adapter.validate_json(json.dumps(value))
        assert json.loads(decoded.model_dump_json()) == value
        if isinstance(decoded, plans.PlanWireModel):
            value["unexpected"] = True
            with pytest.raises(ValidationError):
                adapter.validate_json(json.dumps(value))


def test_model_exposes_arguments_only_and_keeps_get_task():
    from lux_agent.java_tools import model_tools

    tools = model_tools()
    assert len(tools) == 9
    assert "submit_generation" not in {tool.name for tool in tools}
    assert "get_task" in {tool.name for tool in tools}
    for tool in tools:
        assert "operation_id" not in tool.parameters.get("properties", {})
        assert "session_id" not in tool.parameters.get("properties", {})


def plan_call(tool, arguments):
    from datetime import UTC, datetime
    from uuid import uuid4

    from test_protocol_v2 import SESSION, USER

    from lux_agent import protocol
    from lux_agent.execution import ToolExecutionRequest

    run = protocol.RunIdentity(id=str(uuid4()), run_index=1)
    turn = protocol.TurnIdentity(id=str(uuid4()), turn_index=1, run=run)
    call = protocol.AgentItemSnapshot(
        item=protocol.AgentItemIdentity(id=str(uuid4()), item_index=1, turn=turn),
        source_revision=1,
        kind="TOOL_CALL",
        status="COMPLETED",
        payload=protocol.PLAN_CALL_TYPES[tool](arguments=arguments),
        completed_at=datetime.now(UTC),
        call_id="provider-" + str(uuid4()),
        tool_name=tool,
    )
    return ToolExecutionRequest(
        execution_id=str(uuid4()), user_id=USER, session_id=SESSION, call=call
    )


def empty_context():
    from test_protocol_v2 import SESSION

    from lux_agent import generation_plan as plans

    return plans.GenerationPlanContext(
        focus=plans.SessionPlanFocus(
            session_id=SESSION, active_plan_id=None, revision=0
        ),
        plan=None,
        catalog=plans.ModelCatalogSnapshot(
            type="model_catalog_snapshot",
            catalog_revision=1,
            submission_policy=plans.GenerationSubmissionPolicy(
                maximum_tasks_per_submission=32
            ),
            models=(),
        ),
    )


def test_same_response_mutation_submit_is_rejected_before_http(tmp_path):
    import asyncio

    from test_protocol_v2 import SESSION, input_json

    from lux_agent import generation_plan as plans
    from lux_agent import protocol
    from lux_agent.execution import ModelResponseObserved, ObservedModelItem
    from lux_agent.runtime import AgentRuntime
    from lux_agent.store import AgentStore

    class Executor:
        live_slots = ()

        async def stream(self, request):
            yield ModelResponseObserved(
                items=(
                    ObservedModelItem(
                        kind="TOOL_CALL",
                        tool_name="focus_generation_plan",
                        call_id="focus",
                        payload=protocol.FocusGenerationPlanCallContent(
                            arguments=plans.FocusGenerationPlanArguments(
                                plan_id="00000000-0000-0000-0000-000000000001",
                                expected_focus_revision=0,
                            )
                        ),
                    ),
                    ObservedModelItem(
                        kind="TOOL_CALL",
                        tool_name="submit_generation_plan",
                        call_id="submit",
                        payload=protocol.SubmitGenerationPlanCallContent(
                            arguments=plans.SubmitGenerationPlanArguments(
                                plan_id="00000000-0000-0000-0000-000000000001",
                                expected_focus_revision=0,
                                expected_plan_revision=1,
                            )
                        ),
                    ),
                ),
                provider_response={},
            )

    class Boundary:
        calls = 0

        async def read_generation_plan_context(self, owner, session):
            return empty_context()

        async def execute(self, request):
            self.calls += 1
            raise AssertionError("Dependent Java invocation must never occur")

    async def scenario():
        boundary = Boundary()
        with AgentStore(tmp_path / "agent.db") as store:
            store.admit(input_json())
            async with AgentRuntime(
                store, executor=Executor(), tools=boundary, plan_context_reader=boundary
            ) as runtime:
                await runtime.advance(SESSION)
                state = await runtime.state(SESSION)
                results = [item for item in state.items if item.kind == "TOOL_RESULT"]
                assert len(results) == 2
                assert all(
                    item.payload.code == "DEPENDENT_TOOL_BATCH" for item in results
                )
                assert boundary.calls == 0

    asyncio.run(scenario())


def test_plan_http_timeout_reconciles_original_read_receipt():
    import asyncio

    import httpx

    from lux_agent import generation_plan as plans
    from lux_agent.backend import BackendClient

    request = plan_call(
        "get_model_catalog", plans.GetModelCatalogArguments(media_type=None)
    )

    async def scenario():
        received = []

        async def handler(incoming):
            received.append(incoming)
            if incoming.method == "POST":
                value = json.loads(incoming.content)
                assert value["operation_id"] == request.call.item.id
                assert value["session_id"] == request.session_id
                assert value["arguments"] == {"media_type": None}
                raise httpx.ReadTimeout("lost original receipt")
            assert incoming.url.path.endswith(
                "/generation-plan-tool-operations/" + request.call.item.id
            )
            return httpx.Response(
                200,
                json={
                    "type": "generation_plan_tool_operation_receipt",
                    "schema_version": 1,
                    "operation_id": request.call.item.id,
                    "session_id": request.session_id,
                    "tool_name": "get_model_catalog",
                    "request_hash": BackendClient.request_hash(request),
                    "result": empty_context().catalog.model_dump(mode="json"),
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://backend"
        ) as client:
            backend = BackendClient(client)
            with pytest.raises(httpx.ReadTimeout):
                await backend.execute(request)
            receipt = await backend.reconcile(request)
            assert receipt.operation_id == request.call.item.id
            assert [r.method for r in received] == ["POST", "GET"]

    asyncio.run(scenario())


def test_hydration_failure_blocks_model_invocation(tmp_path):
    import asyncio

    from test_protocol_v2 import SESSION, input_json
    from test_runtime_v2 import ReplyExecutor

    from lux_agent.runtime import AgentRuntime
    from lux_agent.store import AgentStore

    class Reader:
        async def read_generation_plan_context(self, owner, session):
            raise OSError("authoritative state unavailable")

    async def scenario():
        executor = ReplyExecutor()
        with AgentStore(tmp_path / "agent.db") as store:
            store.admit(input_json())
            async with AgentRuntime(
                store, executor=executor, plan_context_reader=Reader()
            ) as runtime:
                with pytest.raises(OSError):
                    await runtime.advance(SESSION)
                assert executor.requests == []

    asyncio.run(scenario())


def test_unknown_plan_call_recovers_original_receipt_and_refreshes_context(tmp_path):
    import asyncio

    from test_protocol_v2 import SESSION, input_json

    from lux_agent import generation_plan as plans
    from lux_agent import protocol
    from lux_agent.backend import BackendClient
    from lux_agent.execution import ModelResponseObserved, ObservedModelItem
    from lux_agent.runtime import AgentRuntime
    from lux_agent.store import AgentStore

    class Executor:
        live_slots = ()

        def __init__(self):
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelResponseObserved(
                    items=(
                        ObservedModelItem(
                            kind="TOOL_CALL",
                            tool_name="get_model_catalog",
                            call_id="catalog",
                            payload=protocol.GetModelCatalogCallContent(
                                arguments=plans.GetModelCatalogArguments(
                                    media_type=None
                                )
                            ),
                        ),
                    ),
                    provider_response={"id": "original"},
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
                    provider_response={"id": "final"},
                )

    class Boundary:
        calls = 0
        reads = 0
        recovered = False
        request = None

        async def read_generation_plan_context(self, owner, session):
            self.reads += 1
            context = empty_context()
            return context.model_copy(
                update={
                    "catalog": context.catalog.model_copy(
                        update={"catalog_revision": self.reads}
                    )
                }
            )

        async def execute(self, request):
            self.calls += 1
            self.request = request
            raise OSError("receipt response lost")

        async def reconcile(self, request):
            assert request == self.request
            if not self.recovered:
                return None
            return plans.JavaPlanToolOperationReceipt(
                type="generation_plan_tool_operation_receipt",
                schema_version=1,
                operation_id=request.operation_id,
                session_id=request.session_id,
                tool_name=request.call.tool_name,
                request_hash=BackendClient.request_hash(request),
                result=empty_context().catalog,
            )

    async def scenario():
        path = tmp_path / "agent.db"
        executor, boundary = Executor(), Boundary()
        with AgentStore(path) as store:
            store.admit(input_json())
            async with AgentRuntime(
                store,
                executor=executor,
                tools=boundary,
                tool_reconciler=boundary,
                plan_context_reader=boundary,
            ) as runtime:
                await runtime.advance(SESSION)
                assert len(executor.requests) == 1
                assert store.execution(boundary.request.execution_id).state == "UNKNOWN"
                await runtime.advance(SESSION)
                assert len(executor.requests) == 1
                assert boundary.calls == 1
        boundary.recovered = True
        with AgentStore(path) as store:
            async with AgentRuntime(
                store,
                executor=executor,
                tools=boundary,
                tool_reconciler=boundary,
                plan_context_reader=boundary,
            ) as runtime:
                await runtime.advance(SESSION)
                raw = store.execution(boundary.request.execution_id).response_snapshot
                assert (
                    json.loads(raw)["type"] == "generation_plan_tool_operation_receipt"
                )
                state = await runtime.state(SESSION)
                result = next(
                    item for item in state.items if item.kind == "TOOL_RESULT"
                )
                assert result.call_id == "catalog"
                assert result.payload.type == "model_catalog_snapshot"
                assert result.payload.catalog_revision == 1
                await runtime.advance(SESSION)
                assert len(executor.requests) == 2
                assert (
                    executor.requests[
                        -1
                    ].generation_plan_context.catalog.catalog_revision
                    == 2
                )
                assert boundary.calls == 1
                assert (await runtime.state(SESSION)).runs[-1].status == "COMPLETED"

    asyncio.run(scenario())


def test_read_cursor_hash_uses_java_canonical_instant():
    from lux_agent import generation_plan as plans
    from lux_agent.backend import BackendClient

    request = plan_call(
        "read_generation_plan",
        plans.ReadGenerationPlanArguments(
            query=plans.SessionPlansQuery(
                type="list",
                limit=10,
                cursor=plans.PlansCursor(
                    before_created_at="2026-10-13T00:00:00Z",
                    before_id="00000000-0000-0000-0000-000000000001",
                ),
            )
        ),
    )
    query = request.call.payload.arguments.query
    precise = query.model_copy(
        update={
            "cursor": query.cursor.model_copy(
                update={"before_created_at": "2026-10-13T00:00:00.000000Z"}
            )
        }
    )
    second = request.model_copy(
        update={
            "call": request.call.model_copy(
                update={
                    "payload": request.call.payload.model_copy(
                        update={
                            "arguments": request.call.payload.arguments.model_copy(
                                update={"query": precise}
                            )
                        }
                    )
                }
            )
        }
    )
    assert BackendClient.request_hash(request) == BackendClient.request_hash(second)


def test_receipt_scope_and_hash_mismatch_are_rejected():
    import httpx

    from lux_agent import generation_plan as plans
    from lux_agent.backend import BackendClient

    request = plan_call(
        "get_model_catalog", plans.GetModelCatalogArguments(media_type=None)
    )
    receipt = plans.JavaPlanToolOperationReceipt(
        type="generation_plan_tool_operation_receipt",
        schema_version=1,
        operation_id=request.operation_id,
        session_id=request.session_id,
        tool_name="get_model_catalog",
        request_hash="0" * 64,
        result=empty_context().catalog,
    )
    with pytest.raises(ValueError, match="binding mismatch"):
        BackendClient._plan_receipt(
            request, httpx.Response(200, content=receipt.model_dump_json())
        )


def test_preflight_allows_independent_frozen_submissions_and_rejects_draft_pairs():
    from lux_agent import generation_plan as plans
    from lux_agent.java_tools import dependent_calls

    frozen_value = next(
        example["value"]
        for example in fixtures()
        if example["name"] == "frozen image plan shape only, unregistered model"
    )
    frozen = plans.GenerationPlan.model_validate_json(json.dumps(frozen_value))
    context = empty_context().model_copy(
        update={
            "focus": plans.SessionPlanFocus(
                session_id=frozen.session_id, active_plan_id=frozen.id, revision=2
            ),
            "plan": plans.GenerationPlanSnapshot(
                plan=frozen,
                validation=plans.PlanValidation(
                    can_submit=True, issues=(), validated_catalog_revision=1
                ),
            ),
        }
    )
    arguments = plans.SubmitGenerationPlanArguments(
        plan_id=frozen.id,
        expected_plan_revision=frozen.revision,
        expected_focus_revision=2,
    )
    calls = tuple(plan_call("submit_generation_plan", arguments).call for _ in range(2))
    assert dependent_calls(calls, context) == frozenset()
    draft = frozen.model_copy(
        update={"state": "DRAFT", "frozen_request": None, "frozen_at": None}
    )
    draft_context = context.model_copy(
        update={"plan": context.plan.model_copy(update={"plan": draft})}
    )
    assert dependent_calls(calls, draft_context) == frozenset(
        call.item.id for call in calls
    )


def test_preflight_rejects_same_plan_update_plus_submit():
    from lux_agent import generation_plan as plans
    from lux_agent.java_tools import dependent_calls

    update_value = next(
        example["value"]
        for example in fixtures()
        if example["name"] == "update_generation_plan"
    )
    update = plans.INVOCATION_ADAPTER.validate_json(json.dumps(update_value))
    first = plan_call("update_generation_plan", update.arguments).call
    second = plan_call(
        "submit_generation_plan",
        plans.SubmitGenerationPlanArguments(
            plan_id=update.arguments.plan_id,
            expected_plan_revision=update.arguments.expected_plan_revision,
            expected_focus_revision=update.arguments.expected_focus_revision,
        ),
    ).call
    assert dependent_calls((first, second), empty_context()) == frozenset(
        (first.item.id, second.item.id)
    )


def test_parent_receipt_rejects_duplicate_child_operation():
    import httpx

    from lux_agent import generation_plan as plans
    from lux_agent.backend import BackendClient

    arguments = plans.SubmitGenerationPlanArguments(
        plan_id="00000000-0000-0000-0000-000000000001",
        expected_plan_revision=2,
        expected_focus_revision=1,
    )
    request = plan_call("submit_generation_plan", arguments)
    value = next(
        example["value"]
        for example in fixtures()
        if example["name"] == "three-file batch submission receipt shape only"
    )
    value["operation_id"] = request.operation_id
    value["session_id"] = request.session_id
    value["request_hash"] = BackendClient.request_hash(request)
    value["result"]["tasks"][1]["task_operation_id"] = value["result"]["tasks"][0][
        "task_operation_id"
    ]
    with pytest.raises(ValueError, match="child Task receipt membership"):
        BackendClient._plan_receipt(request, httpx.Response(200, json=value))


@pytest.mark.parametrize("field", ["provider_id", "provider_model_id", "adapter_id"])
@pytest.mark.parametrize("blank", [" \t\n", "\u00a0", "\u2003", "\u3000"])
def test_provider_binding_rejects_whitespace_only(field, blank):
    from lux_agent.generation_plan import ProviderBinding

    value = {
        "provider_id": "provider",
        "provider_model_id": "model",
        "adapter_id": "adapter",
    }
    value[field] = blank
    with pytest.raises(ValidationError):
        ProviderBinding.model_validate_json(json.dumps(value))


def test_provider_binding_preserves_nonblank_values():
    from lux_agent.generation_plan import ProviderBinding

    value = {
        "provider_id": " provider ",
        "provider_model_id": "\u00a0model\u00a0",
        "adapter_id": "\u2003adapter\u3000",
    }
    binding = ProviderBinding.model_validate_json(json.dumps(value))
    assert binding.model_dump(mode="json") == value
