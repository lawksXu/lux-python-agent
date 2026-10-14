import asyncio
import json
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest
from test_protocol_v2 import SESSION, USER, input_json

from lux_agent import protocol
from lux_agent.execution import ToolExecutionRequest


def tool_request(name="submit_generation"):
    run = protocol.RunIdentity(id=str(uuid4()), run_index=1)
    turn = protocol.TurnIdentity(id=str(uuid4()), turn_index=1, run=run)
    payload = (
        protocol.SubmitGenerationCallContent(
            arguments=protocol.SubmitGenerationArguments(
                kind="IMAGE", request=protocol.ConnectorRequest(prompt="draw")
            )
        )
        if name == "submit_generation"
        else protocol.GetTaskCallContent(
            arguments=protocol.GetTaskArguments(task_id=str(uuid4()))
        )
    )
    call = protocol.AgentItemSnapshot(
        item=protocol.AgentItemIdentity(id=str(uuid4()), item_index=1, turn=turn),
        source_revision=1,
        kind="TOOL_CALL",
        status="COMPLETED",
        payload=payload,
        completed_at=datetime.now(UTC),
        call_id="provider-call",
        tool_name=name,
    )
    return ToolExecutionRequest(
        execution_id=str(uuid4()), user_id=USER, session_id=SESSION, call=call
    )


def test_submit_reconciles_original_operation_without_resubmission():
    async def scenario():
        from lux_agent.backend import BackendClient

        task = str(uuid4())
        requests = []
        request = tool_request()

        async def handler(incoming):
            requests.append(incoming)
            if incoming.method == "POST":
                assert (
                    json.loads(incoming.content)["operation_id"] == request.call.item.id
                )
                raise httpx.ReadTimeout("response lost", request=incoming)
            assert incoming.url.path.endswith(
                "/tasks/by-operation/" + request.call.item.id
            )
            return httpx.Response(
                200,
                json={
                    "type": "task_submission_receipt",
                    "task_id": task,
                    "accepted": True,
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://backend"
        ) as client:
            backend = BackendClient(client)
            with pytest.raises(httpx.ReadTimeout):
                await backend.execute(request)
            result = await backend.reconcile(request)
            assert result.task_id == task
            assert [r.method for r in requests] == ["POST", "GET"]

    asyncio.run(scenario())


def test_query_error_is_tool_error_and_repair_preserves_original_values():
    async def scenario():
        from lux_agent.backend import BackendClient

        original = input_json()

        async def handler(request):
            if "/inputs" in request.url.path:
                assert request.url.params["through_input_sequence_no"] == "1"
                return httpx.Response(200, content="[" + original + "]")
            return httpx.Response(404)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://backend"
        ) as client:
            backend = BackendClient(client)
            result = await backend.execute(tool_request("get_task"))
            assert result.code == "TASK_NOT_FOUND"
            repaired = await backend.read_range(USER, SESSION, 0, 1, 100)
            assert json.loads(repaired[0]) == json.loads(original)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["timeout", "unavailable"])
def test_get_task_transport_failure_is_a_settled_unobserved_read(failure):
    async def scenario():
        from lux_agent.backend import BackendClient

        request = tool_request("get_task")
        received = []

        async def handler(incoming):
            received.append(incoming)
            if failure == "timeout":
                raise httpx.ReadTimeout("response timed out", request=incoming)
            return httpx.Response(503, request=incoming)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://backend"
        ) as client:
            result = await BackendClient(client).execute(request)

        assert isinstance(result, protocol.ToolError)
        assert result.code == "TASK_READ_UNAVAILABLE"
        assert "no task snapshot was observed" in result.message.lower()
        assert [incoming.method for incoming in received] == ["GET"]

    asyncio.run(scenario())
