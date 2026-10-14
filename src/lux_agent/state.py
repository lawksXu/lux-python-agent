"""Named checkpoint state; no Java lifecycle projection controls execution."""

from typing import Literal, TypedDict

from langchain_core.messages import AnyMessage

from lux_agent.execution import ExecutionFailure, StreamItemCandidate
from lux_agent.media_types import MediaObservation
from lux_agent.protocol import (
    AcceptedInputItem,
    AgentItemIdentity,
    AgentItemSnapshot,
    OutboundFactIntent,
    ResourceId,
    RunSnapshot,
    ToolResultContent,
    TurnSnapshot,
    WireModel,
)


class PendingInput(WireModel):
    item: AcceptedInputItem
    ordinal: int


class CheckpointInputProgress(WireModel):
    retained_through: int = 0
    incorporated_through: int = 0
    admission_cut: int = 0
    through_input_sequence_no: int = 0


class WakeRequest(WireModel):
    event_id: ResourceId
    ordinal: int


class ControlEffect(WireModel):
    event_id: ResourceId
    ordinal: int
    event_type: Literal["STOP_SESSION", "CONTINUE_SESSION"]
    resolved_run_id: ResourceId | None = None
    completed: bool = False


class SessionControlState(WireModel):
    effects: tuple[ControlEffect, ...] = ()
    latest_stop_ordinal: int = 0
    pending_wake: WakeRequest | None = None
    stop_requested: bool = False


class ToolBinding(WireModel):
    execution_id: ResourceId
    call: AgentItemSnapshot
    result_item: AgentItemIdentity
    result: ToolResultContent | None = None
    fatal_error: ExecutionFailure | None = None


class RunExecutionState(WireModel):
    snapshot: RunSnapshot
    last_turn_index: int = 0
    phase: Literal["READY", "EXECUTING", "RECOVERY"] = "READY"


class TurnExecutionState(WireModel):
    snapshot: TurnSnapshot
    model_execution_id: ResourceId
    frozen_input_ids: tuple[ResourceId, ...]
    candidates: tuple[StreamItemCandidate, ...]
    last_item_index: int
    model_completed: bool = False
    model_items: tuple[AgentItemSnapshot, ...] = ()
    tools: tuple[ToolBinding, ...] = ()
    fatal_error: ExecutionFailure | None = None
    unknown: bool = False


class SessionAgentState(WireModel):
    state_version: Literal[2] = 2
    user_id: ResourceId
    session_id: ResourceId
    messages: tuple[AnyMessage, ...] = ()
    media_observations: tuple[MediaObservation, ...] = ()
    pending_inputs: tuple[PendingInput, ...] = ()
    input_progress: CheckpointInputProgress = CheckpointInputProgress()
    last_command_ordinal: int = 0
    incorporated_ids: tuple[ResourceId, ...] = ()
    last_run_index: int = 0
    active_run: RunExecutionState | None = None
    active_turn: TurnExecutionState | None = None
    control: SessionControlState = SessionControlState()
    runs: tuple[RunSnapshot, ...] = ()
    turns: tuple[TurnSnapshot, ...] = ()
    items: tuple[AgentItemSnapshot, ...] = ()
    pending_events: tuple[OutboundFactIntent, ...] = ()


class GraphState(TypedDict):
    session: SessionAgentState
