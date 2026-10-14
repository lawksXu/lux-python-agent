"""Named Java business HTTP boundary; uncertain submissions are only queried."""

import hashlib
import json
from datetime import datetime
from typing import Literal

import httpx

from lux_agent import generation_plan as plans
from lux_agent.execution import (
    PLAN_TOOL_NAMES,
    ToolExecutionRequest,
    ToolExecutionResult,
)
from lux_agent.protocol import (
    ConnectorRequest,
    InputAccepted,
    ResourceId,
    TaskQuerySnapshot,
    TaskSubmissionReceipt,
    ToolError,
    WireModel,
    decode_command,
)


class TaskSubmitRequest(WireModel):
    operation_id: ResourceId
    session_id: ResourceId
    kind: Literal["IMAGE", "VIDEO"]
    request: ConnectorRequest


class BackendClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        prefix = f"/internal/v2/users/{request.user_id}"
        if request.call.tool_name in PLAN_TOOL_NAMES:
            invocation = self._invocation(request)
            response = await self._client.post(
                f"{prefix}/sessions/{request.session_id}/generation-plan-tools/{request.call.tool_name}",
                content=invocation.model_dump_json(),
                headers={"content-type": "application/json"},
            )
            response.raise_for_status()
            return self._plan_receipt(request, response)
        if request.call.tool_name == "submit_generation":
            arguments = request.call.payload.arguments
            body = TaskSubmitRequest(
                operation_id=request.call.item.id,
                session_id=request.session_id,
                kind=arguments.kind,
                request=arguments.request,
            )
            response = await self._client.post(
                f"{prefix}/tasks",
                content=body.model_dump_json(),
                headers={"content-type": "application/json"},
            )
            error = self._rejection(response)
            return error or TaskSubmissionReceipt.model_validate_json(response.content)
        arguments = request.call.payload.arguments
        try:
            response = await self._client.get(
                f"{prefix}/sessions/{request.session_id}/tasks/{arguments.task_id}"
            )
        except httpx.RequestError:
            return ToolError(
                code="TASK_READ_UNAVAILABLE",
                message="No task snapshot was observed because the read request failed",
            )
        if response.status_code >= 500 or response.status_code in (408, 429):
            return ToolError(
                code="TASK_READ_UNAVAILABLE",
                message=(
                    "No task snapshot was observed because the read request failed "
                    f"(HTTP {response.status_code})"
                ),
            )
        error = self._rejection(response)
        return error or TaskQuerySnapshot.model_validate_json(response.content)

    @staticmethod
    def _rejection(response: httpx.Response) -> ToolError | None:
        if response.status_code >= 500 or response.status_code in (408, 429):
            response.raise_for_status()
        if 400 <= response.status_code < 500:
            return ToolError(
                code="TASK_NOT_FOUND"
                if response.status_code == 404
                else "TASK_REQUEST_REJECTED",
                message=f"Task request rejected (HTTP {response.status_code})",
            )
        response.raise_for_status()
        return None

    async def reconcile(
        self, request: ToolExecutionRequest
    ) -> ToolExecutionResult | None:
        if request.operation_id is None:
            # A different get_task read cannot recover the original observation.
            return None
        if request.call.tool_name in PLAN_TOOL_NAMES:
            response = await self._client.get(
                f"/internal/v2/users/{request.user_id}/sessions/{request.session_id}"
                f"/generation-plan-tool-operations/{request.operation_id}"
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return self._plan_receipt(request, response)
        response = await self._client.get(
            f"/internal/v2/users/{request.user_id}/tasks/by-operation/{request.operation_id}"
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return TaskSubmissionReceipt.model_validate_json(response.content)

    @staticmethod
    def _invocation(request: ToolExecutionRequest) -> plans.JavaPlanToolInvocation:
        return plans.INVOCATION_ADAPTER.validate_json(
            json.dumps(
                {
                    "type": "generation_plan_tool_invocation",
                    "operation_id": request.call.item.id,
                    "session_id": request.session_id,
                    "tool_name": request.call.tool_name,
                    "arguments": request.call.payload.arguments.model_dump(mode="json"),
                }
            )
        )

    @classmethod
    def request_hash(cls, request: ToolExecutionRequest) -> str:
        invocation = cls._invocation(request)
        wire = invocation.model_dump(mode="json")
        if (
            isinstance(invocation, plans.ReadGenerationPlanInvocation)
            and isinstance(invocation.arguments.query, plans.SessionPlansQuery)
            and invocation.arguments.query.cursor is not None
        ):
            # Java's typed Instant codec fixes six fractional UTC digits for hashing.
            cursor = invocation.arguments.query.cursor
            wire["arguments"]["query"]["cursor"]["before_created_at"] = (
                datetime.fromisoformat(cursor.before_created_at)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z")
            )
        canonical = json.dumps(
            {
                "operation": "generation-plan-tools-v1",
                "input": {"user_id": request.user_id, "invocation": wire},
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def _plan_receipt(
        cls, request: ToolExecutionRequest, response: httpx.Response
    ) -> plans.JavaPlanToolOperationReceipt:
        receipt = plans.JavaPlanToolOperationReceipt.model_validate_json(
            response.content
        )
        if (
            receipt.operation_id != request.operation_id
            or receipt.session_id != request.session_id
            or receipt.tool_name != request.call.tool_name
            or receipt.request_hash != cls.request_hash(request)
        ):
            raise ValueError("Original plan operation receipt binding mismatch")
        if isinstance(receipt.result, plans.PlanTaskSubmissionReceipt):
            tasks = receipt.result.tasks
            if (
                [task.index for task in tasks] != list(range(len(tasks)))
                or len(tasks) != receipt.result.requested_file_count
                or len({task.task_id for task in tasks}) != len(tasks)
                or len({task.task_operation_id for task in tasks}) != len(tasks)
                or any(task.task_operation_id == receipt.operation_id for task in tasks)
            ):
                raise ValueError("Invalid original child Task receipt membership")
        return receipt

    async def read_generation_plan_context(
        self, owner: str, session: str
    ) -> plans.GenerationPlanContext:
        response = await self._client.get(
            f"/internal/v2/users/{owner}/sessions/{session}/generation-plan-context"
        )
        response.raise_for_status()
        context = plans.GenerationPlanContext.model_validate_json(response.content)
        if context.focus.session_id != session:
            raise ValueError("Authoritative context Session mismatch")
        if (
            (context.plan is None) != (context.focus.active_plan_id is None)
            or context.plan is not None
            and (
                context.plan.plan.id != context.focus.active_plan_id
                or context.plan.plan.session_id != session
                or context.plan.plan.user_id != owner
            )
        ):
            raise ValueError("Authoritative context plan/focus/owner mismatch")
        return context

    async def read_range(
        self, owner: str, session: str, after: int, through: int, limit: int
    ) -> tuple[str, ...]:
        response = await self._client.get(
            f"/internal/v2/users/{owner}/sessions/{session}/inputs",
            params={
                "after_input_sequence_no": after,
                "through_input_sequence_no": through,
                "limit": limit,
            },
        )
        response.raise_for_status()
        # This ingress preserves original JSON values/time strings before decoding.
        values = json.loads(response.content)
        if not isinstance(values, list):
            raise ValueError("Input repair requires an array of original envelopes")
        originals = []
        for value in values:
            raw = json.dumps(
                value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
            command = decode_command(raw)
            if (
                not isinstance(command, InputAccepted)
                or command.user_id != owner
                or command.session_id != session
            ):
                raise ValueError("Input repair owner/Session/type mismatch")
            originals.append(raw)
        return tuple(originals)
