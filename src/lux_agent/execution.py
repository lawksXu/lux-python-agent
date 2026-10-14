"""Injected model/tool boundaries; provider-owned response data stays opaque."""

from collections.abc import AsyncIterator
from typing import Literal, Protocol, get_args

from langchain_core.messages import AnyMessage
from pydantic import Field, JsonValue

from lux_agent import generation_plan as plans
from lux_agent.media_types import MediaObservation
from lux_agent.protocol import (
    AgentItemIdentity,
    AgentItemSnapshot,
    ErrorMessage,
    LifecycleErrorCode,
    OrderedContent,
    ResourceId,
    Timestamp,
    ToolCallContent,
    ToolName,
    ToolResultContent,
    TurnIdentity,
    WireModel,
)


class LiveOutputSlot(WireModel):
    key: str
    kind: Literal["MESSAGE", "REASONING_SUMMARY"]


class StreamItemCandidate(WireModel):
    slot: str
    model_execution_id: ResourceId
    item: AgentItemIdentity
    initial_snapshot: AgentItemSnapshot
    start_event_id: ResourceId


class ModelToolDefinition(WireModel):
    name: ToolName
    description: str
    # JSON Schema is an external framework mapping, not an anonymous business record.
    parameters: dict[str, JsonValue]


class ModelRequest(WireModel):
    execution_id: ResourceId
    user_id: ResourceId
    session_id: ResourceId
    turn: TurnIdentity
    messages: tuple[AnyMessage, ...]
    candidates: tuple[StreamItemCandidate, ...]
    generation_plan_context: plans.GenerationPlanContext | None = None
    tools: tuple[ModelToolDefinition, ...] = ()
    media_input_ids: tuple[ResourceId, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    media_observations: tuple[MediaObservation, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )


class MediaObserver(Protocol):
    async def prepare(self, request: ModelRequest) -> ModelRequest: ...
    async def invocation_messages(
        self, request: ModelRequest
    ) -> tuple[AnyMessage, ...]: ...


class ModelTextFragment(WireModel):
    slot: str
    text: str


class ObservedModelItem(WireModel):
    slot: str | None = None
    kind: Literal["MESSAGE", "REASONING_SUMMARY", "TOOL_CALL"]
    payload: OrderedContent | ToolCallContent
    call_id: str | None = None
    tool_name: ToolName | None = None


class ModelResponseObserved(WireModel):
    items: tuple[ObservedModelItem, ...]
    # Only the provider adapter owns/interprets this archived original response.
    provider_response: JsonValue
    messages: tuple[AnyMessage, ...] = ()


class ModelResponseReceipt(WireModel):
    observed: ModelResponseObserved
    items: tuple[AgentItemSnapshot, ...]
    messages: tuple[AnyMessage, ...]
    completion_event_id: ResourceId
    response_at: Timestamp


class ExecutionFailure(WireModel):
    code: LifecycleErrorCode
    message: ErrorMessage


class ToolExecutionRequest(WireModel):
    execution_id: ResourceId
    user_id: ResourceId
    session_id: ResourceId
    call: AgentItemSnapshot

    @property
    def operation_id(self) -> str | None:
        return (
            self.call.item.id
            if self.call.tool_name == "submit_generation"
            or self.call.tool_name in PLAN_TOOL_NAMES
            else None
        )


class ModelExecutor(Protocol):
    @property
    def live_slots(self) -> tuple[LiveOutputSlot, ...]: ...
    def stream(
        self, request: ModelRequest
    ) -> AsyncIterator[ModelTextFragment | ModelResponseObserved]: ...


PLAN_TOOL_NAMES = get_args(plans.GenerationPlanToolName)

ToolExecutionResult = ToolResultContent | plans.JavaPlanToolOperationReceipt


class PlanContextReader(Protocol):
    async def read_generation_plan_context(
        self, owner: str, session: str
    ) -> plans.GenerationPlanContext: ...


class ToolExecutor(Protocol):
    async def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult: ...


class ModelReconciler(Protocol):
    async def reconcile(
        self, request: ModelRequest
    ) -> ModelResponseObserved | None: ...


class ToolReconciler(Protocol):
    async def reconcile(
        self, request: ToolExecutionRequest
    ) -> ToolExecutionResult | None: ...


class ExecutionUnknown(RuntimeError):
    """An external outcome must be reconciled using the original identity."""


class ConfirmedExecutionFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.failure = ExecutionFailure(code=code, message=message)
