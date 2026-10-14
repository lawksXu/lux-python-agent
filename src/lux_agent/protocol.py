"""Strict named v2 Session input and Python execution-fact contracts."""

import re
from datetime import datetime
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    TypeAdapter,
    field_validator,
    model_validator,
)

from lux_agent import generation_plan as plans

MAX_KAFKA_RECORD_BYTES = 32 * 1024 * 1024
MAX_KAFKA_FETCH_BYTES = 64 * 1024 * 1024
MAX_EXECUTION_FACT_BYTES = 31 * 1024 * 1024


def canonical_uuid(value: str) -> str:
    if str(UUID(value)) != value:
        raise ValueError("Expected canonical lowercase UUID")
    return value


def aware_timestamp(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("Timestamp requires a timezone")
    return value


def source_precision(value: str | datetime) -> datetime:
    if isinstance(value, str):
        fraction = re.search(r"T\d{2}:\d{2}:\d{2}\.(\d+)", value)
        if fraction and any(digit != "0" for digit in fraction.group(1)[6:]):
            raise ValueError(
                "Source lifecycle timestamps require microsecond precision"
            )
        return datetime.fromisoformat(value)
    return value


def error_message_capacity(value: str) -> str:
    if len(value.encode("utf-8")) > 65535:
        raise ValueError("Lifecycle error message exceeds UTF-8 TEXT capacity")
    return value


ResourceId = Annotated[str, AfterValidator(canonical_uuid)]
Timestamp = Annotated[datetime, AfterValidator(aware_timestamp)]
SourceTimestamp = Annotated[Timestamp, BeforeValidator(source_precision)]
PositiveSequence = Annotated[StrictInt, Field(ge=1, le=2**63 - 1)]
Watermark = Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]
AssetSourceIndex = Annotated[StrictInt, Field(ge=0, le=2**31 - 1)]
LifecycleErrorCode = Annotated[str, Field(max_length=64)]
ErrorMessage = Annotated[str, AfterValidator(error_message_capacity)]
CallId = Annotated[str, Field(min_length=1, max_length=255)]
RunStatus = Literal["RUNNING", "COMPLETED", "FAILED", "CANCELLED"]
TurnStatus = Literal[
    "PREPARED", "RUNNING", "INTERRUPTED", "COMPLETED", "FAILED", "CANCELLED"
]
ItemKind = Literal["MESSAGE", "REASONING_SUMMARY", "TOOL_CALL", "TOOL_RESULT"]
TaskStatus = Literal[
    "CREATED", "SUBMITTED", "RUNNING", "IMPORTING", "SUCCEEDED", "FAILED"
]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TextBlock(WireModel):
    type: Literal["text"] = "text"
    text: str


class AssetReferenceBlock(WireModel):
    type: Literal["asset"] = "asset"
    asset_id: ResourceId


ContentBlock = Annotated[TextBlock | AssetReferenceBlock, Field(discriminator="type")]


class OrderedContent(WireModel):
    blocks: tuple[ContentBlock, ...]


class ConnectorRequest(WireModel):
    """Provider-owned parameters are opaque here; only asset_ids is interpreted."""

    model_config = ConfigDict(extra="allow", frozen=True, strict=True)
    asset_ids: tuple[ResourceId, ...] = ()
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)


class SubmitGenerationArguments(WireModel):
    kind: Literal["IMAGE", "VIDEO"]
    request: ConnectorRequest


class GetTaskArguments(WireModel):
    task_id: ResourceId


class SubmitGenerationCallContent(WireModel):
    arguments: SubmitGenerationArguments


class GetTaskCallContent(WireModel):
    arguments: GetTaskArguments


class GetModelCatalogCallContent(WireModel):
    arguments: plans.GetModelCatalogArguments


class ReadGenerationPlanCallContent(WireModel):
    arguments: plans.ReadGenerationPlanArguments


class CreateGenerationPlanCallContent(WireModel):
    arguments: plans.CreateGenerationPlanArguments


class UpdateGenerationPlanCallContent(WireModel):
    arguments: plans.UpdateGenerationPlanArguments


class ChangeGenerationTargetCallContent(WireModel):
    arguments: plans.ChangeGenerationTargetArguments


class CloneGenerationPlanCallContent(WireModel):
    arguments: plans.CloneGenerationPlanArguments


class FocusGenerationPlanCallContent(WireModel):
    arguments: plans.FocusGenerationPlanArguments


class SubmitGenerationPlanCallContent(WireModel):
    arguments: plans.SubmitGenerationPlanArguments


PLAN_CALL_TYPES = {
    "get_model_catalog": GetModelCatalogCallContent,
    "read_generation_plan": ReadGenerationPlanCallContent,
    "create_generation_plan": CreateGenerationPlanCallContent,
    "update_generation_plan": UpdateGenerationPlanCallContent,
    "change_generation_target": ChangeGenerationTargetCallContent,
    "clone_generation_plan": CloneGenerationPlanCallContent,
    "focus_generation_plan": FocusGenerationPlanCallContent,
    "submit_generation_plan": SubmitGenerationPlanCallContent,
}
ToolName = Literal["submit_generation", "get_task"] | plans.GenerationPlanToolName

PlanToolCallContent = (
    GetModelCatalogCallContent
    | ReadGenerationPlanCallContent
    | CreateGenerationPlanCallContent
    | UpdateGenerationPlanCallContent
    | ChangeGenerationTargetCallContent
    | CloneGenerationPlanCallContent
    | FocusGenerationPlanCallContent
    | SubmitGenerationPlanCallContent
)

ToolCallContent = SubmitGenerationCallContent | GetTaskCallContent | PlanToolCallContent


class TaskSubmissionReceipt(WireModel):
    type: Literal["task_submission_receipt"] = "task_submission_receipt"
    task_id: ResourceId
    accepted: Literal[True] = True

    @field_validator("accepted", mode="before")
    @classmethod
    def boolean_acceptance(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("Acceptance must be true")
        return value


class TaskAssetReference(WireModel):
    asset_id: ResourceId
    source_index: AssetSourceIndex


class TaskFailure(WireModel):
    stage: Literal["GENERATION", "IMPORT"]
    code: str | None
    message: str | None


class TaskQuerySnapshot(WireModel):
    type: Literal["task_query_snapshot"] = "task_query_snapshot"
    task_id: ResourceId
    version: PositiveSequence
    status: TaskStatus
    assets: tuple[TaskAssetReference, ...]
    failure: TaskFailure | None

    @model_validator(mode="after")
    def failure_and_assets(self) -> Self:
        if (self.status == "FAILED") != (self.failure is not None):
            raise ValueError("Failure detail must be present exactly for FAILED")
        indexes = [asset.source_index for asset in self.assets]
        if indexes != sorted(set(indexes)) or len(
            {a.asset_id for a in self.assets}
        ) != len(self.assets):
            raise ValueError("Asset references require unique ordered source positions")
        return self


class ToolError(WireModel):
    type: Literal["tool_error"] = "tool_error"
    code: str
    message: str

    @field_validator("code")
    @classmethod
    def nonblank_code(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Tool error code must not be blank")
        return value


ToolResultContent = Annotated[
    TaskSubmissionReceipt
    | TaskQuerySnapshot
    | ToolError
    | plans.GenerationPlanToolResult,
    Field(discriminator="type"),
]
_TOOL_RESULT = TypeAdapter(ToolResultContent)


def decode_tool_result(raw: str | bytes) -> ToolResultContent:
    return _TOOL_RESULT.validate_json(raw)


class TaskEventContent(WireModel):
    task_id: ResourceId


class AcceptedInputItem(WireModel):
    id: ResourceId
    sequence_no: PositiveSequence
    input_sequence_no: PositiveSequence
    source: Literal["USER", "SYSTEM"]
    kind: Literal["MESSAGE", "TASK_EVENT"]
    content: OrderedContent | TaskEventContent

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        if self.kind == "TASK_EVENT":
            if self.source != "SYSTEM" or not isinstance(
                self.content, TaskEventContent
            ):
                raise ValueError("Task notifications are SYSTEM inputs")
        elif not isinstance(self.content, OrderedContent) or not self.content.blocks:
            raise ValueError("Messages require ordered content")
        elif not any(
            isinstance(b, AssetReferenceBlock) or b.text.strip()
            for b in self.content.blocks
        ):
            raise ValueError("Input messages must contain content")
        return self


class Envelope(WireModel):
    schema_version: Literal[2]
    event_id: ResourceId
    occurred_at: Timestamp
    user_id: ResourceId
    session_id: ResourceId

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("Protocol version must be an integer")
        return value

    @property
    def partition_key(self) -> str:
        return f"{self.user_id}:{self.session_id}"


class InputAccepted(Envelope):
    event_type: Literal["INPUT_ACCEPTED"] = "INPUT_ACCEPTED"
    item: AcceptedInputItem

    @model_validator(mode="after")
    def input_identity(self) -> Self:
        if self.event_id != self.item.id:
            raise ValueError("Input event ID must equal Item ID")
        return self


class StopSession(Envelope):
    event_type: Literal["STOP_SESSION"] = "STOP_SESSION"


class ContinueSession(Envelope):
    event_type: Literal["CONTINUE_SESSION"] = "CONTINUE_SESSION"


class QuerySessionExecution(Envelope):
    event_type: Literal["QUERY_SESSION_EXECUTION"] = "QUERY_SESSION_EXECUTION"
    limit: Annotated[StrictInt, Field(ge=1, le=1000)]


class ReplaySessionEvents(Envelope):
    event_type: Literal["REPLAY_SESSION_EVENTS"] = "REPLAY_SESSION_EVENTS"
    after_event_seq: Watermark
    through_event_seq: Watermark
    limit: Annotated[StrictInt, Field(ge=1, le=1000)]

    @model_validator(mode="after")
    def replay_range(self) -> Self:
        if self.through_event_seq < self.after_event_seq:
            raise ValueError("Replay bounds are reversed")
        return self


Command = Annotated[
    InputAccepted
    | StopSession
    | ContinueSession
    | QuerySessionExecution
    | ReplaySessionEvents,
    Field(discriminator="event_type"),
]
_COMMAND = TypeAdapter(Command)


def decode_command(raw: str | bytes) -> Command:
    return _COMMAND.validate_json(raw)


class RunIdentity(WireModel):
    id: ResourceId
    run_index: PositiveSequence


class TurnIdentity(WireModel):
    id: ResourceId
    turn_index: PositiveSequence
    run: RunIdentity


class AgentItemIdentity(WireModel):
    id: ResourceId
    item_index: PositiveSequence
    turn: TurnIdentity


class RunSnapshot(WireModel):
    run: RunIdentity
    source_revision: PositiveSequence
    status: RunStatus
    started_at: SourceTimestamp | None
    finished_at: SourceTimestamp | None
    final_item: AgentItemIdentity | None
    error_code: LifecycleErrorCode | None
    error_message: ErrorMessage | None

    @model_validator(mode="after")
    def lifecycle(self) -> Self:
        if (self.status != "RUNNING") != (self.finished_at is not None):
            raise ValueError("Run terminal status and time must agree")
        if self.status == "COMPLETED" and self.final_item is None:
            raise ValueError("Completed Run requires selected final Item")
        if self.final_item is not None and self.final_item.turn.run != self.run:
            raise ValueError("Final Item belongs to another Run")
        return self


class TurnSnapshot(WireModel):
    turn: TurnIdentity
    source_revision: PositiveSequence
    status: TurnStatus
    started_at: SourceTimestamp | None
    finished_at: SourceTimestamp | None
    error_code: LifecycleErrorCode | None
    error_message: ErrorMessage | None

    @model_validator(mode="after")
    def lifecycle(self) -> Self:
        terminal = self.status in ("COMPLETED", "FAILED", "CANCELLED")
        if terminal != (self.finished_at is not None):
            raise ValueError("Turn terminal status and time must agree")
        if self.status == "CANCELLED" and self.started_at is not None:
            raise ValueError("Only uninvoked Turns may be cancelled")
        return self


class AgentItemSnapshot(WireModel):
    item: AgentItemIdentity
    source: Literal["AGENT"] = "AGENT"
    source_revision: PositiveSequence
    kind: ItemKind
    status: Literal["PARTIAL", "COMPLETED", "INTERRUPTED"]
    payload: OrderedContent | ToolCallContent | ToolResultContent
    completed_at: SourceTimestamp | None
    call_id: CallId | None
    tool_name: ToolName | None

    @model_validator(mode="after")
    def item_shape(self) -> Self:
        if (self.status == "COMPLETED") != (self.completed_at is not None):
            raise ValueError("Completion time must match Item status")
        if self.kind in ("MESSAGE", "REASONING_SUMMARY"):
            if (
                not isinstance(self.payload, OrderedContent)
                or self.call_id is not None
                or self.tool_name is not None
            ):
                raise ValueError(
                    "Message Item requires ordered content without call fields"
                )
            if self.status == "PARTIAL" and (
                self.source_revision != 1 or self.payload.blocks
            ):
                raise ValueError("Only initial empty revision-one PARTIAL is valid")
        elif self.kind == "TOOL_CALL":
            expected = (
                SubmitGenerationCallContent
                if self.tool_name == "submit_generation"
                else GetTaskCallContent
                if self.tool_name == "get_task"
                else PLAN_CALL_TYPES.get(self.tool_name)
            )
            if (
                self.tool_name is None
                or self.call_id is None
                or expected is None
                or not isinstance(self.payload, expected)
                or self.status != "COMPLETED"
            ):
                raise ValueError("Tool call binding/arguments are invalid")
        elif (
            self.call_id is None
            or self.tool_name is not None
            or not isinstance(
                self.payload,
                (
                    TaskSubmissionReceipt,
                    TaskQuerySnapshot,
                    ToolError,
                    plans.PlanWireModel,
                ),
            )
            or self.status != "COMPLETED"
        ):
            raise ValueError("Tool result binding/outcome is invalid")
        return self


class ModelResponseCompleted(WireModel):
    model_execution_id: Annotated[str, Field(min_length=1)]
    turn: TurnIdentity
    items: tuple[AgentItemSnapshot, ...]

    @model_validator(mode="after")
    def response_binding(self) -> Self:
        ids = [snapshot.item.id for snapshot in self.items]
        positions = [snapshot.item.item_index for snapshot in self.items]
        if len(set(ids)) != len(ids) or positions != sorted(set(positions)):
            raise ValueError("Response Items require unique ordered bindings")
        if any(
            s.item.turn != self.turn
            or s.status != "COMPLETED"
            or s.kind == "TOOL_RESULT"
            for s in self.items
        ):
            raise ValueError("Response contains invalid Item associations")
        calls = [
            snapshot.call_id for snapshot in self.items if snapshot.kind == "TOOL_CALL"
        ]
        if len(set(calls)) != len(calls):
            raise ValueError("One model response cannot repeat a tool call identity")
        return self


class ItemTextDelta(WireModel):
    item: AgentItemIdentity
    stream_seq: PositiveSequence
    text: str


class ObservedTextFragment(Envelope):
    occurred_at: SourceTimestamp
    item: AgentItemIdentity
    text: str


FactType = Literal[
    "RUN_SNAPSHOT",
    "TURN_SNAPSHOT",
    "AGENT_ITEM_SNAPSHOT",
    "MODEL_RESPONSE_COMPLETED",
    "ITEM_TEXT_DELTA",
]
FactBody = (
    RunSnapshot
    | TurnSnapshot
    | AgentItemSnapshot
    | ModelResponseCompleted
    | ItemTextDelta
)


class OutboundFactIntent(Envelope):
    occurred_at: SourceTimestamp
    event_type: FactType
    body: FactBody

    @model_validator(mode="after")
    def body_type(self) -> Self:
        expected = {
            "RUN_SNAPSHOT": RunSnapshot,
            "TURN_SNAPSHOT": TurnSnapshot,
            "AGENT_ITEM_SNAPSHOT": AgentItemSnapshot,
            "MODEL_RESPONSE_COMPLETED": ModelResponseCompleted,
            "ITEM_TEXT_DELTA": ItemTextDelta,
        }[self.event_type]
        if not isinstance(self.body, expected):
            raise ValueError("Fact type and body disagree")
        return self


class ExecutionFact(OutboundFactIntent):
    event_seq: PositiveSequence


def decode_fact(raw: str | bytes) -> ExecutionFact:
    return ExecutionFact.model_validate_json(raw)


class SessionExecutionSync(WireModel):
    user_id: ResourceId
    session_id: ResourceId
    event_high_watermark: Watermark
    events: tuple[ExecutionFact, ...]
    runs: tuple[RunSnapshot, ...]
    turns: tuple[TurnSnapshot, ...]
    items: tuple[AgentItemSnapshot, ...]
    next_after_event_seq: Watermark | None
