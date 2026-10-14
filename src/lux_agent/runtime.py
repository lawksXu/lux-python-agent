"""One checkpointed Session driver owns logical Turns and external-call barriers."""

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, Self
from uuid import uuid4

import aiosqlite
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from lux_agent import execution, generation_plan, media_types, protocol
from lux_agent import state as checkpoint_state
from lux_agent.checkpoint import SessionStateSerializer
from lux_agent.execution import (
    ConfirmedExecutionFailure,
    ExecutionFailure,
    MediaObserver,
    ModelExecutor,
    ModelReconciler,
    ModelRequest,
    ModelResponseObserved,
    ModelResponseReceipt,
    ModelTextFragment,
    PlanContextReader,
    StreamItemCandidate,
    ToolExecutionRequest,
    ToolExecutor,
    ToolReconciler,
)
from lux_agent.java_tools import dependent_calls, dependent_rejection, model_tools
from lux_agent.protocol import (
    AgentItemIdentity,
    AgentItemSnapshot,
    ContinueSession,
    FactBody,
    InputAccepted,
    ModelResponseCompleted,
    ObservedTextFragment,
    OrderedContent,
    OutboundFactIntent,
    QuerySessionExecution,
    ReplaySessionEvents,
    RunIdentity,
    RunSnapshot,
    StopSession,
    TextBlock,
    ToolError,
    TurnIdentity,
    TurnSnapshot,
    decode_command,
    decode_tool_result,
)
from lux_agent.state import (
    ControlEffect,
    GraphState,
    PendingInput,
    RunExecutionState,
    SessionAgentState,
    ToolBinding,
    TurnExecutionState,
    WakeRequest,
)
from lux_agent.store import AgentStore, ExecutionPreparation

_LOGGER = logging.getLogger(__name__)


class InputRepair(Protocol):
    async def read_range(
        self, owner: str, session: str, after: int, through: int, limit: int
    ) -> tuple[str, ...]: ...


def identity() -> str:
    return str(uuid4())


def _replace_snapshot[T: RunSnapshot | TurnSnapshot | AgentItemSnapshot](
    old: tuple[T, ...], snapshot: T
) -> tuple[T, ...]:
    key = (
        snapshot.run.id
        if isinstance(snapshot, RunSnapshot)
        else snapshot.turn.id
        if isinstance(snapshot, TurnSnapshot)
        else snapshot.item.id
    )

    def key_of(value: T) -> str:
        return (
            value.run.id
            if isinstance(value, RunSnapshot)
            else value.turn.id
            if isinstance(value, TurnSnapshot)
            else value.item.id
        )

    return (
        tuple(snapshot if key_of(value) == key else value for value in old)
        if any(key_of(value) == key for value in old)
        else (*old, snapshot)
    )


def _append_messages(
    messages: tuple[AnyMessage, ...], additions: tuple[AnyMessage, ...]
) -> tuple[AnyMessage, ...]:
    existing = {message.id: message for message in messages if message.id is not None}
    result = list(messages)
    for message in additions:
        if message.id is not None and message.id in existing:
            if existing[message.id] != message:
                raise ValueError("Conflicting model message identity")
        else:
            result.append(message)
            if message.id is not None:
                existing[message.id] = message
    return tuple(result)


def validate_context(messages: tuple[AnyMessage, ...]) -> None:
    """Complete every real assistant call batch before another model request."""
    pending: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            if message.tool_call_id not in pending:
                raise ValueError("Unpaired or duplicate tool result in model context")
            pending.remove(message.tool_call_id)
        else:
            if pending:
                raise ValueError("Incomplete assistant tool-call batch")
            if isinstance(message, AIMessage):
                calls = [call["id"] for call in message.tool_calls]
                if len(set(calls)) != len(calls):
                    raise ValueError("Duplicate call identity in one model response")
                pending.update(calls)
    if pending:
        raise ValueError("Incomplete assistant tool-call batch")


class AgentRuntime:
    def __init__(
        self,
        store: AgentStore,
        executor: ModelExecutor | None = None,
        tools: ToolExecutor | None = None,
        repair: InputRepair | None = None,
        model_reconciler: ModelReconciler | None = None,
        tool_reconciler: ToolReconciler | None = None,
        plan_context_reader: PlanContextReader | None = None,
        media_observer: MediaObserver | None = None,
    ) -> None:
        self.store = store
        self.executor = executor
        self.tools = tools
        self.repair = repair
        self.model_reconciler = model_reconciler
        self.tool_reconciler = tool_reconciler
        self.plan_context_reader = plan_context_reader
        self.media_observer = media_observer
        self.checkpoint_path = Path(f"{store.path}.checkpoints")
        self._stack: AsyncExitStack | None = None
        self._graph = None
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._dispatcher: asyncio.Task[None] | None = None
        self._recovery_due: dict[str, float] = {}

    async def __aenter__(self) -> Self:
        if self._stack is not None:
            raise RuntimeError("Agent runtime is already open")
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            stack.enter_context(self.store.driver_lifecycle())
            descriptor = os.open(self.checkpoint_path, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(descriptor)
            connection = await stack.enter_async_context(
                aiosqlite.connect(self.checkpoint_path)
            )
            await connection.execute("PRAGMA synchronous=FULL")
            # Native serializer permits only the explicitly owned model symbols.
            models = tuple(
                value
                for module in (
                    protocol,
                    execution,
                    checkpoint_state,
                    generation_plan,
                    media_types,
                )
                for value in vars(module).values()
                if isinstance(value, type)
                and issubclass(
                    value, (protocol.WireModel, generation_plan.PlanWireModel)
                )
            )
            serializer = SessionStateSerializer(allowed_msgpack_modules=models)
            saver = AsyncSqliteSaver(connection, serde=serializer)
            await saver.setup()
            graph = StateGraph(GraphState)
            graph.add_node("prepare", self._prepare)
            graph.add_node("start_turn", self._start_turn)
            graph.add_node("model", self._model)
            graph.add_node("complete_model", self._complete_model)
            graph.add_node("tools", self._tools)
            graph.add_node("finish_turn", self._finish_turn)
            graph.add_node("decide", self._decide)
            graph.add_edge(START, "prepare")
            graph.add_conditional_edges(
                "prepare",
                lambda value: "start_turn" if value["session"].active_turn else END,
            )
            graph.add_conditional_edges(
                "start_turn",
                lambda value: (
                    "decide"
                    if value["session"].active_turn.snapshot.status == "CANCELLED"
                    else "model"
                ),
            )
            graph.add_edge("model", "complete_model")
            graph.add_edge("complete_model", "tools")
            graph.add_edge("tools", "finish_turn")
            graph.add_edge("finish_turn", "decide")
            graph.add_edge("decide", END)
            self._graph = graph.compile(checkpointer=saver)
            self._stack = stack
            return self
        except BaseException:
            await stack.aclose()
            raise

    async def __aexit__(self, *_: object) -> None:
        await self.stop()
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._graph = None

    def _config(self, session: str):
        # LangGraph owns this map-shaped adapter contract.
        return {"configurable": {"thread_id": session}}

    async def state(self, session: str) -> SessionAgentState | None:
        snapshot = await self._graph.aget_state(self._config(session))
        value = snapshot.values.get("session")
        return self._typed(value) if value is not None else None

    @staticmethod
    def _typed(value: SessionAgentState) -> SessionAgentState:
        # JsonPlus restores Pydantic instances with model_construct; validate nested
        # fields at the framework boundary instead of trusting reconstructed dicts.
        return SessionAgentState.model_validate(vars(value), strict=False)

    async def _checkpoint_id(self, session: str) -> str:
        snapshot = await self._graph.aget_state(self._config(session))
        return str(snapshot.config["configurable"].get("checkpoint_id", ""))

    async def _repair_markers(self, state: SessionAgentState) -> None:
        applied = (
            *state.incorporated_ids,
            *(effect.event_id for effect in state.control.effects),
        )
        self.store.mark_applied(applied, await self._checkpoint_id(state.session_id))

    def _materialize(self, state: SessionAgentState) -> SessionAgentState:
        for event in state.pending_events:
            self.store.append_fact(event)
        # Removal is persisted only by this node's next sync checkpoint.
        return state.model_copy(update={"pending_events": ()})

    def _emit(self, state: SessionAgentState, body: FactBody) -> SessionAgentState:
        event_type = (
            "RUN_SNAPSHOT"
            if isinstance(body, RunSnapshot)
            else "TURN_SNAPSHOT"
            if isinstance(body, TurnSnapshot)
            else "MODEL_RESPONSE_COMPLETED"
            if isinstance(body, ModelResponseCompleted)
            else "AGENT_ITEM_SNAPSHOT"
        )
        intent = OutboundFactIntent(
            schema_version=2,
            event_id=identity(),
            occurred_at=datetime.now(UTC),
            user_id=state.user_id,
            session_id=state.session_id,
            event_type=event_type,
            body=body,
        )
        return state.model_copy(
            update={"pending_events": (*state.pending_events, intent)}
        )

    def _run_snapshot(
        self, state: SessionAgentState, snapshot: RunSnapshot
    ) -> SessionAgentState:
        run = state.active_run.model_copy(update={"snapshot": snapshot})
        return self._emit(
            state.model_copy(
                update={
                    "active_run": run,
                    "runs": _replace_snapshot(state.runs, snapshot),
                }
            ),
            snapshot,
        )

    def _turn_snapshot(
        self, state: SessionAgentState, snapshot: TurnSnapshot
    ) -> SessionAgentState:
        turn = state.active_turn.model_copy(update={"snapshot": snapshot})
        return self._emit(
            state.model_copy(
                update={
                    "active_turn": turn,
                    "turns": _replace_snapshot(state.turns, snapshot),
                }
            ),
            snapshot,
        )

    def _admissions(self, state: SessionAgentState) -> SessionAgentState:
        pending = list(state.pending_inputs)
        known = {value.item.id for value in pending} | set(state.incorporated_ids)
        control = state.control
        last = state.last_command_ordinal
        for record in self.store.commands(state.session_id, last):
            command = record.command
            if command.user_id != state.user_id:
                raise ValueError("Checkpoint Session owner mismatch")
            last = record.ordinal
            if isinstance(command, InputAccepted):
                if command.item.id not in known:
                    pending.append(
                        PendingInput(item=command.item, ordinal=record.ordinal)
                    )
                    known.add(command.item.id)
                    control = control.model_copy(
                        update={
                            "pending_wake": WakeRequest(
                                event_id=command.event_id, ordinal=record.ordinal
                            )
                        }
                    )
            elif isinstance(command, StopSession):
                effect = ControlEffect(
                    event_id=command.event_id,
                    ordinal=record.ordinal,
                    event_type="STOP_SESSION",
                    resolved_run_id=state.active_run.snapshot.run.id
                    if state.active_run
                    else None,
                    completed=state.active_run is None,
                )
                control = control.model_copy(
                    update={
                        "effects": (*control.effects, effect),
                        "latest_stop_ordinal": record.ordinal,
                        "pending_wake": None,
                        "stop_requested": state.active_run is not None,
                    }
                )
            elif isinstance(command, ContinueSession):
                effect = ControlEffect(
                    event_id=command.event_id,
                    ordinal=record.ordinal,
                    event_type="CONTINUE_SESSION",
                    completed=True,
                )
                control = control.model_copy(
                    update={
                        "effects": (*control.effects, effect),
                        "pending_wake": WakeRequest(
                            event_id=command.event_id, ordinal=record.ordinal
                        ),
                    }
                )
            elif isinstance(command, ReplaySessionEvents):
                self.store.replay(command)
            elif isinstance(command, QuerySessionExecution):
                self.store.observe(command)
        sequences = {value.item.input_sequence_no for value in pending}
        retained = state.input_progress.incorporated_through
        while retained + 1 in sequences:
            retained += 1
        progress = state.input_progress.model_copy(
            update={"retained_through": retained, "admission_cut": last}
        )
        return state.model_copy(
            update={
                "pending_inputs": tuple(pending),
                "control": control,
                "last_command_ordinal": last,
                "input_progress": progress,
            }
        )

    async def _prepare(self, graph_state: GraphState) -> GraphState:
        state = self._admissions(self._typed(graph_state["session"]))
        if state.active_turn is not None:
            return {"session": state}
        if state.active_run is not None and state.control.stop_requested:
            return {"session": self._end_run(state, "CANCELLED")}
        if state.active_run is None and state.control.pending_wake is None:
            return {"session": state}
        through = max(
            (value.item.input_sequence_no for value in state.pending_inputs),
            default=state.input_progress.incorporated_through,
        )
        progress = state.input_progress.model_copy(
            update={"through_input_sequence_no": through}
        )
        state = state.model_copy(update={"input_progress": progress})
        if through > progress.retained_through:
            if self.repair is None:
                return {"session": state}
            after = progress.incorporated_through
            while after < through:
                originals = await self.repair.read_range(
                    state.user_id, state.session_id, after, through, 100
                )
                if not originals:
                    return {"session": state}
                last = after
                for raw in originals:
                    command = decode_command(raw)
                    if (
                        not isinstance(command, InputAccepted)
                        or command.user_id != state.user_id
                        or command.session_id != state.session_id
                    ):
                        raise ValueError("Repair returned another owner/Session")
                    sequence = command.item.input_sequence_no
                    if sequence != last + 1 or sequence > through:
                        raise ValueError(
                            "Repair returned noncontiguous or out-of-range input"
                        )
                    last = sequence
                for raw in originals:
                    self.store.admit(raw)
                after = last
            state = self._admissions(state)
            if state.control.stop_requested:
                return {
                    "session": self._end_run(state, "CANCELLED")
                    if state.active_run
                    else state
                }
        if state.active_run is None:
            if state.control.pending_wake is None:
                return {"session": state}
            validate_context(state.messages)
            run = RunSnapshot(
                run=RunIdentity(id=identity(), run_index=state.last_run_index + 1),
                source_revision=1,
                status="RUNNING",
                started_at=None,
                finished_at=None,
                final_item=None,
                error_code=None,
                error_message=None,
            )
            state = state.model_copy(
                update={
                    "last_run_index": run.run.run_index,
                    "active_run": RunExecutionState(snapshot=run),
                }
            )
            state = self._run_snapshot(state, run)
        run = state.active_run
        frozen = tuple(
            sorted(
                (
                    value
                    for value in state.pending_inputs
                    if value.item.input_sequence_no <= through
                ),
                key=lambda value: value.item.input_sequence_no,
            )
        )
        if [value.item.input_sequence_no for value in frozen] != list(
            range(state.input_progress.incorporated_through + 1, through + 1)
        ):
            return {"session": state}
        messages = state.messages
        for value in frozen:
            item = value.item
            if item.kind == "TASK_EVENT":
                message = SystemMessage(
                    content=item.content.model_dump_json(),
                    id=item.id,
                    additional_kwargs={
                        "lux_task_notice": True,
                        "__openai_role__": "developer",
                    },
                )
            else:
                text = "".join(
                    block.text
                    if isinstance(block, TextBlock)
                    else f"[asset:{block.asset_id}]"
                    for block in item.content.blocks
                )
                message_type = HumanMessage if item.source == "USER" else SystemMessage
                message = message_type(
                    content=text,
                    id=item.id,
                    additional_kwargs={"lux_input": item.model_dump(mode="json")},
                )
            messages = _append_messages(messages, (message,))
        turn_id = TurnIdentity(
            id=identity(), turn_index=run.last_turn_index + 1, run=run.snapshot.run
        )
        model_id = identity()
        candidates = []
        slots = self.executor.live_slots if self.executor is not None else ()
        if len({slot.key for slot in slots}) != len(slots):
            raise ValueError("Adapter live slots must be unique")
        for index, slot in enumerate(slots, 1):
            item_id = AgentItemIdentity(id=identity(), item_index=index, turn=turn_id)
            initial = AgentItemSnapshot(
                item=item_id,
                source_revision=1,
                kind=slot.kind,
                status="PARTIAL",
                payload=OrderedContent(blocks=()),
                completed_at=None,
                call_id=None,
                tool_name=None,
            )
            candidates.append(
                StreamItemCandidate(
                    slot=slot.key,
                    model_execution_id=model_id,
                    item=item_id,
                    initial_snapshot=initial,
                    start_event_id=identity(),
                )
            )
        turn = TurnSnapshot(
            turn=turn_id,
            source_revision=1,
            status="PREPARED",
            started_at=None,
            finished_at=None,
            error_code=None,
            error_message=None,
        )
        progress = state.input_progress.model_copy(
            update={"incorporated_through": through}
        )
        state = state.model_copy(
            update={
                "messages": messages,
                "pending_inputs": tuple(
                    value
                    for value in state.pending_inputs
                    if value.item.input_sequence_no > through
                ),
                "input_progress": progress,
                "incorporated_ids": (
                    *state.incorporated_ids,
                    *(value.item.id for value in frozen),
                ),
                "control": state.control.model_copy(update={"pending_wake": None}),
                "active_run": run.model_copy(
                    update={"last_turn_index": turn_id.turn_index, "phase": "EXECUTING"}
                ),
                "active_turn": TurnExecutionState(
                    snapshot=turn,
                    model_execution_id=model_id,
                    frozen_input_ids=tuple(value.item.id for value in frozen),
                    candidates=tuple(candidates),
                    last_item_index=len(candidates),
                ),
            }
        )
        return {"session": self._turn_snapshot(state, turn)}

    async def _start_turn(self, graph_state: GraphState) -> GraphState:
        state = self._admissions(self._typed(graph_state["session"]))
        turn = state.active_turn
        if turn.snapshot.status != "PREPARED":
            return {"session": state}
        if state.control.stop_requested:
            snapshot = turn.snapshot.model_copy(
                update={
                    "status": "CANCELLED",
                    "source_revision": 2,
                    "finished_at": datetime.now(UTC),
                }
            )
            return {"session": self._turn_snapshot(state, snapshot)}
        timestamp = datetime.now(UTC)
        if state.active_run.snapshot.started_at is None:
            run = state.active_run.snapshot.model_copy(
                update={
                    "started_at": timestamp,
                    "source_revision": state.active_run.snapshot.source_revision + 1,
                }
            )
            state = self._run_snapshot(state, run)
        snapshot = turn.snapshot.model_copy(
            update={"status": "RUNNING", "source_revision": 2, "started_at": timestamp}
        )
        return {"session": self._turn_snapshot(state, snapshot)}

    def _request(self, state: SessionAgentState) -> ModelRequest:
        turn = state.active_turn
        return ModelRequest(
            execution_id=turn.model_execution_id,
            user_id=state.user_id,
            session_id=state.session_id,
            turn=turn.snapshot.turn,
            messages=state.messages,
            candidates=turn.candidates,
            tools=model_tools(),
            media_input_ids=turn.frozen_input_ids if self.media_observer else (),
            media_observations=state.media_observations if self.media_observer else (),
        )

    def _response_receipt(
        self, state: SessionAgentState, observed: ModelResponseObserved
    ) -> ModelResponseReceipt:
        turn = state.active_turn
        candidates = {candidate.slot: candidate for candidate in turn.candidates}
        index = turn.last_item_index
        snapshots = []
        seen_slots: set[str] = set()
        timestamp = datetime.now(UTC)
        for output in observed.items:
            candidate = candidates.get(output.slot)
            if candidate is not None:
                if (
                    output.slot in seen_slots
                    or candidate.initial_snapshot.kind != output.kind
                ):
                    raise ValueError("Conflicting observed stream binding")
                seen_slots.add(output.slot)
                binding, revision = candidate.item, 2
            else:
                index += 1
                binding = AgentItemIdentity(
                    id=identity(), item_index=index, turn=turn.snapshot.turn
                )
                revision = 1
            snapshots.append(
                AgentItemSnapshot(
                    item=binding,
                    source_revision=revision,
                    kind=output.kind,
                    status="COMPLETED",
                    payload=output.payload,
                    completed_at=timestamp,
                    call_id=output.call_id,
                    tool_name=output.tool_name,
                )
            )
        snapshots.sort(key=lambda value: value.item.item_index)
        actual_starts = {
            fact.body.item.id
            for fact in self.store.events(state.session_id, limit=2**31 - 1)
            if fact.event_type == "AGENT_ITEM_SNAPSHOT"
            and isinstance(fact.body, AgentItemSnapshot)
            and fact.body.item.turn == turn.snapshot.turn
        }
        if actual_starts - {value.item.id for value in snapshots}:
            raise ValueError("Original response omitted an observed Item")
        messages = observed.messages
        if not messages:
            text = "".join(
                block.text
                for snapshot in snapshots
                if snapshot.kind in ("MESSAGE", "REASONING_SUMMARY")
                for block in snapshot.payload.blocks
                if isinstance(block, TextBlock)
            )
            calls = [
                {
                    "name": snapshot.tool_name,
                    "id": snapshot.call_id,
                    "args": snapshot.payload.arguments.model_dump(mode="json"),
                    "type": "tool_call",
                }
                for snapshot in snapshots
                if snapshot.kind == "TOOL_CALL"
            ]
            messages = (
                AIMessage(content=text, tool_calls=calls, id=turn.model_execution_id),
            )
        expected = {
            snapshot.call_id: snapshot
            for snapshot in snapshots
            if snapshot.kind == "TOOL_CALL"
        }
        actual = [
            call
            for message in messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        ]
        if (
            any(not isinstance(message, AIMessage) for message in messages)
            or len(expected) != len(actual)
            or set(expected) != {call["id"] for call in actual}
            or any(
                call["name"] != expected[call["id"]].tool_name
                or call["args"]
                != expected[call["id"]].payload.arguments.model_dump(mode="json")
                for call in actual
            )
        ):
            raise ValueError("Provider response and normalized call bindings disagree")
        messages = tuple(
            message
            if message.id
            else message.model_copy(update={"id": f"{turn.model_execution_id}:{index}"})
            for index, message in enumerate(messages)
        )
        return ModelResponseReceipt(
            observed=observed,
            items=tuple(snapshots),
            messages=messages,
            completion_event_id=identity(),
            response_at=timestamp,
        )

    async def _model(self, graph_state: GraphState) -> GraphState:
        state = self._typed(graph_state["session"])
        await self._repair_markers(state)
        self.store.register_run(
            state.user_id, state.session_id, state.active_run.snapshot.run.id
        )
        state = self._materialize(state)
        turn = state.active_turn
        request = self._request(state)
        if turn.model_completed:
            return {"session": state}
        existing_receipt = self.store.execution(request.execution_id)
        if existing_receipt is not None:
            request = ModelRequest.model_validate_json(
                existing_receipt.preparation.request_snapshot
            )
        elif self.plan_context_reader is not None:
            # Hydrate after history assembly/compaction and before invocation.
            # A failure exits before preparation or an external model call.
            context = await self.plan_context_reader.read_generation_plan_context(
                state.user_id, state.session_id
            )
            request = request.model_copy(
                update={
                    "generation_plan_context": context,
                    "messages": (
                        *request.messages,
                        SystemMessage(
                            content="Authoritative generation plan context: "
                            + context.model_dump_json(),
                            id="generation-plan-context-" + request.execution_id,
                        ),
                    ),
                }
            )
        media_failure: ExecutionFailure | None = None
        if existing_receipt is None and self.media_observer is not None:
            try:
                request = await self.media_observer.prepare(request)
            except ValueError:
                media_failure = ExecutionFailure(
                    code="MEDIA_OBSERVATION_FAILED",
                    message="Media could not be verified before model submission",
                )
        prepared = ExecutionPreparation(
            execution_id=request.execution_id,
            kind="MODEL",
            user_id=state.user_id,
            session_id=state.session_id,
            run_id=request.turn.run.id,
            turn_id=request.turn.id,
            request_snapshot=request.model_dump_json(),
        )
        receipt = self.store.execution(request.execution_id)
        if receipt is None:
            receipt = self.store.prepare_execution(prepared)
        if media_failure is not None:
            self.store.execution_confirmed(
                request.execution_id, media_failure.model_dump_json(), failed=True
            )
            return {"session": state}
        if receipt.state in ("CONFIRMED", "FAILED"):
            return {"session": state}
        if receipt.state in ("IN_FLIGHT", "UNKNOWN"):
            self.store.execution_unknown(request.execution_id)
            try:
                response = (
                    await self.model_reconciler.reconcile(request)
                    if self.model_reconciler
                    else None
                )
            except ConfirmedExecutionFailure as error:
                self.store.execution_confirmed(
                    request.execution_id, error.failure.model_dump_json(), failed=True
                )
                return {"session": state}
            if response is not None:
                recovered = self._response_receipt(state, response)
                self.store.execution_confirmed(
                    request.execution_id, recovered.model_dump_json()
                )
            else:
                state = state.model_copy(
                    update={"active_turn": turn.model_copy(update={"unknown": True})}
                )
            return {"session": state}
        if self.executor is None:
            failure = ExecutionFailure(
                code="MODEL_NOT_CONFIGURED", message="Model executor is not configured"
            )
            self.store.execution_confirmed(
                request.execution_id, failure.model_dump_json(), failed=True
            )
            return {"session": state}
        validate_context(request.messages)
        self.store.execution_started(request.execution_id)
        observed: ModelResponseObserved | None = None
        started: set[str] = set()
        candidates = {candidate.slot: candidate for candidate in turn.candidates}
        try:
            async for emission in self.executor.stream(request):
                if observed is not None:
                    raise ValueError("Model emitted data after its complete response")
                if isinstance(emission, ModelResponseObserved):
                    observed = emission
                elif isinstance(emission, ModelTextFragment):
                    candidate = candidates.get(emission.slot)
                    if candidate is None:
                        # Unplanned forms are bound by the complete response.
                        continue
                    if emission.slot not in started:
                        self.store.append_fact(
                            OutboundFactIntent(
                                schema_version=2,
                                event_id=candidate.start_event_id,
                                occurred_at=datetime.now(UTC),
                                user_id=state.user_id,
                                session_id=state.session_id,
                                event_type="AGENT_ITEM_SNAPSHOT",
                                body=candidate.initial_snapshot,
                            )
                        )
                        started.add(emission.slot)
                    self.store.append_fragment(
                        ObservedTextFragment(
                            schema_version=2,
                            event_id=identity(),
                            occurred_at=datetime.now(UTC),
                            user_id=state.user_id,
                            session_id=state.session_id,
                            item=candidate.item,
                            text=emission.text,
                        )
                    )
                else:
                    raise ValueError("Invalid model emission")
            if observed is None:
                raise RuntimeError(
                    "Model stream ended without its original complete response"
                )
            saved = self._response_receipt(state, observed)
            self.store.execution_confirmed(
                request.execution_id, saved.model_dump_json()
            )
        except ConfirmedExecutionFailure as error:
            self.store.execution_confirmed(
                request.execution_id, error.failure.model_dump_json(), failed=True
            )
        except BaseException:
            self.store.execution_unknown(request.execution_id)
            if (
                isinstance(asyncio.current_task().cancelling(), int)
                and asyncio.current_task().cancelling()
            ):
                raise
            state = state.model_copy(
                update={"active_turn": turn.model_copy(update={"unknown": True})}
            )
        return {"session": state}

    async def _complete_model(self, graph_state: GraphState) -> GraphState:
        state = self._typed(graph_state["session"])
        turn = state.active_turn
        if turn.model_completed:
            return {"session": state}
        receipt = self.store.execution(turn.model_execution_id)
        if receipt.state == "FAILED":
            failure = ExecutionFailure.model_validate_json(receipt.response_snapshot)
            # Only a confirmed failure seals observed partials. UNKNOWN keeps them
            # open so reconciliation can complete these original Item identities.
            observed = self.store.sync(state.user_id, state.session_id).items
            for snapshot in observed:
                if (
                    snapshot.item.turn != turn.snapshot.turn
                    or snapshot.status != "PARTIAL"
                ):
                    continue
                fragments = tuple(
                    TextBlock(text=fact.body.text)
                    for fact in self.store.events(state.session_id, limit=2**31 - 1)
                    if fact.event_type == "ITEM_TEXT_DELTA"
                    and fact.body.item == snapshot.item
                )
                interrupted = snapshot.model_copy(
                    update={
                        "source_revision": snapshot.source_revision + 1,
                        "status": "INTERRUPTED",
                        "payload": OrderedContent(blocks=fragments),
                    }
                )
                state = self._emit(
                    state.model_copy(
                        update={"items": _replace_snapshot(state.items, interrupted)}
                    ),
                    interrupted,
                )
            return {
                "session": state.model_copy(
                    update={
                        "active_turn": turn.model_copy(
                            update={"fatal_error": failure, "unknown": False}
                        )
                    }
                )
            }
        if receipt.state != "CONFIRMED":
            return {
                "session": state.model_copy(
                    update={"active_turn": turn.model_copy(update={"unknown": True})}
                )
            }
        saved = ModelResponseReceipt.model_validate_json(receipt.response_snapshot)
        frozen_request = ModelRequest.model_validate_json(
            receipt.preparation.request_snapshot
        )
        state = state.model_copy(
            update={"media_observations": frozen_request.media_observations}
        )
        indexes = [snapshot.item.item_index for snapshot in saved.items]
        index = max((turn.last_item_index, *indexes))
        bindings = []
        for call in saved.items:
            if call.kind == "TOOL_CALL":
                index += 1
                result_item = AgentItemIdentity(
                    id=identity(), item_index=index, turn=turn.snapshot.turn
                )
                bindings.append(
                    ToolBinding(
                        execution_id=identity(), call=call, result_item=result_item
                    )
                )
        items = state.items
        for snapshot in saved.items:
            items = _replace_snapshot(items, snapshot)
        response = ModelResponseCompleted(
            model_execution_id=turn.model_execution_id,
            turn=turn.snapshot.turn,
            items=saved.items,
        )
        event = OutboundFactIntent(
            schema_version=2,
            event_id=saved.completion_event_id,
            occurred_at=saved.response_at,
            user_id=state.user_id,
            session_id=state.session_id,
            event_type="MODEL_RESPONSE_COMPLETED",
            body=response,
        )
        return {
            "session": state.model_copy(
                update={
                    "messages": _append_messages(state.messages, saved.messages),
                    "items": items,
                    "active_turn": turn.model_copy(
                        update={
                            "model_completed": True,
                            "model_items": saved.items,
                            "last_item_index": index,
                            "tools": tuple(bindings),
                            "unknown": False,
                        }
                    ),
                    "pending_events": (*state.pending_events, event),
                }
            )
        }

    @staticmethod
    def _external_tool_result(
        request: ToolExecutionRequest, result: execution.ToolExecutionResult
    ) -> execution.ToolExecutionResult:
        if request.call.tool_name in protocol.PLAN_CALL_TYPES:
            if not isinstance(result, generation_plan.JavaPlanToolOperationReceipt):
                raise ValueError(
                    "Plan tools must retain their complete original receipt"
                )
            if (
                result.operation_id != request.operation_id
                or result.session_id != request.session_id
                or result.tool_name != request.call.tool_name
            ):
                raise ValueError("Original plan operation receipt binding mismatch")
        return result

    @staticmethod
    def _tool_result(raw: str) -> protocol.ToolResultContent:
        # Preserve complete receipts; conversation Items expose their result.
        import json

        value = json.loads(raw)
        if value.get("type") == "generation_plan_tool_operation_receipt":
            return generation_plan.JavaPlanToolOperationReceipt.model_validate_json(
                raw
            ).result
        return decode_tool_result(raw)

    async def _tool(
        self, state: SessionAgentState, binding: ToolBinding
    ) -> ToolBinding:
        request = ToolExecutionRequest(
            execution_id=binding.execution_id,
            user_id=state.user_id,
            session_id=state.session_id,
            call=binding.call,
        )
        prepared = ExecutionPreparation(
            execution_id=request.execution_id,
            kind="TOOL",
            user_id=state.user_id,
            session_id=state.session_id,
            run_id=request.call.item.turn.run.id,
            turn_id=request.call.item.turn.id,
            tool_call_item_id=request.call.item.id,
            operation_id=request.operation_id,
            request_snapshot=request.model_dump_json(),
        )
        receipt = self.store.execution(request.execution_id)
        if receipt is None:
            receipt = self.store.prepare_execution(prepared)
        if receipt.state in ("CONFIRMED", "FAILED"):
            result = self._tool_result(receipt.response_snapshot)
            fatal = (
                ExecutionFailure(code=result.code, message=result.message)
                if receipt.state == "FAILED"
                else binding.fatal_error
            )
            return binding.model_copy(update={"result": result, "fatal_error": fatal})
        if receipt.state in ("UNKNOWN", "IN_FLIGHT"):
            self.store.execution_unknown(request.execution_id)
            if request.call.tool_name == "get_task":
                # Reads cannot recover their original observation after restart;
                # querying again would attach a newer snapshot to the old call.
                result = ToolError(
                    code="TASK_READ_INTERRUPTED",
                    message=(
                        "The original task read was interrupted before a snapshot "
                        "was observed"
                    ),
                )
                self.store.execution_confirmed(
                    request.execution_id, result.model_dump_json()
                )
                return binding.model_copy(update={"result": result})
            try:
                result = (
                    await self.tool_reconciler.reconcile(request)
                    if self.tool_reconciler
                    else None
                )
            except ConfirmedExecutionFailure as error:
                result = ToolError(
                    code=error.failure.code, message=error.failure.message
                )
                self.store.execution_confirmed(
                    request.execution_id, result.model_dump_json(), failed=True
                )
                return binding.model_copy(
                    update={"result": result, "fatal_error": error.failure}
                )
            if result is None:
                return binding
            result = self._external_tool_result(request, result)
            self.store.execution_confirmed(
                request.execution_id, result.model_dump_json()
            )
            return binding.model_copy(
                update={"result": self._tool_result(result.model_dump_json())}
            )
        if self.tools is None:
            result = ToolError(
                code="TOOL_NOT_EXECUTED",
                message="No tool executor is configured; this call was not invoked",
            )
            self.store.execution_confirmed(
                request.execution_id, result.model_dump_json()
            )
            return binding.model_copy(
                update={"result": self._tool_result(result.model_dump_json())}
            )
        self.store.execution_started(request.execution_id)
        try:
            result = self._external_tool_result(
                request, await self.tools.execute(request)
            )
            self.store.execution_confirmed(
                request.execution_id, result.model_dump_json()
            )
            return binding.model_copy(
                update={"result": self._tool_result(result.model_dump_json())}
            )
        except ConfirmedExecutionFailure as error:
            result = ToolError(code=error.failure.code, message=error.failure.message)
            self.store.execution_confirmed(
                request.execution_id, result.model_dump_json(), failed=True
            )
            return binding.model_copy(
                update={"result": result, "fatal_error": error.failure}
            )
        except BaseException:
            self.store.execution_unknown(request.execution_id)
            if asyncio.current_task().cancelling():
                raise
            return binding

    async def _tools(self, graph_state: GraphState) -> GraphState:
        state = self._materialize(self._typed(graph_state["session"]))
        turn = state.active_turn
        if not turn.model_completed:
            return {"session": state}
        model_receipt = self.store.execution(turn.model_execution_id)
        model_request = ModelRequest.model_validate_json(
            model_receipt.preparation.request_snapshot
        )
        rejected = dependent_calls(
            tuple(binding.call for binding in turn.tools),
            model_request.generation_plan_context,
        )
        # New plan calls require Java authority; historical calls remain recoverable.
        unavailable = frozenset(
            binding.call.item.id
            for binding in turn.tools
            if binding.call.tool_name in protocol.PLAN_CALL_TYPES
            and model_request.generation_plan_context is None
            and self.store.execution(binding.execution_id) is None
        )

        async def settle(binding: ToolBinding) -> ToolBinding:
            if binding.call.item.id in rejected or binding.call.item.id in unavailable:
                result = (
                    dependent_rejection()
                    if binding.call.item.id in rejected
                    else ToolError(
                        code="PLAN_CONTEXT_UNAVAILABLE",
                        message="Authoritative generation plan context is required.",
                    )
                )
                request = ToolExecutionRequest(
                    execution_id=binding.execution_id,
                    user_id=state.user_id,
                    session_id=state.session_id,
                    call=binding.call,
                )
                receipt = self.store.execution(binding.execution_id)
                if receipt is not None:
                    return await self._tool(state, binding)
                self.store.prepare_execution(
                    ExecutionPreparation(
                        execution_id=binding.execution_id,
                        kind="TOOL",
                        user_id=state.user_id,
                        session_id=state.session_id,
                        run_id=binding.call.item.turn.run.id,
                        turn_id=binding.call.item.turn.id,
                        tool_call_item_id=binding.call.item.id,
                        operation_id=request.operation_id,
                        request_snapshot=request.model_dump_json(),
                    )
                )
                self.store.execution_confirmed(
                    binding.execution_id, result.model_dump_json()
                )
                return binding.model_copy(update={"result": result})
            return await self._tool(state, binding)

        # Reconcile uncertain originals before starting new business calls.
        for binding in turn.tools:
            receipt = self.store.execution(binding.execution_id)
            if receipt is not None and receipt.state in ("UNKNOWN", "IN_FLIGHT"):
                recovered = await self._tool(state, binding)
                if recovered.result is None:
                    return {
                        "session": state.model_copy(
                            update={
                                "active_turn": turn.model_copy(update={"unknown": True})
                            }
                        )
                    }
        settled = await asyncio.gather(
            *(settle(binding) for binding in turn.tools),
            return_exceptions=True,
        )
        # Persistence failures still drain sibling external calls before leaving
        # the graph node or closing its receipt store.
        for outcome in settled:
            if isinstance(outcome, BaseException):
                raise outcome
        bindings = tuple(
            outcome for outcome in settled if isinstance(outcome, ToolBinding)
        )
        fatal = next(
            (
                binding.fatal_error
                for binding in bindings
                if binding.fatal_error is not None
            ),
            turn.fatal_error,
        )
        unknown = any(binding.result is None for binding in bindings)
        return {
            "session": state.model_copy(
                update={
                    "active_turn": turn.model_copy(
                        update={
                            "tools": bindings,
                            "fatal_error": fatal,
                            "unknown": unknown,
                        }
                    )
                }
            )
        }

    async def _finish_turn(self, graph_state: GraphState) -> GraphState:
        state = self._typed(graph_state["session"])
        turn = state.active_turn
        for binding in turn.tools:
            if binding.result is None:
                continue
            existing = next(
                (
                    item
                    for item in state.items
                    if item.item.id == binding.result_item.id
                ),
                None,
            )
            if existing is None:
                receipt = self.store.execution(binding.execution_id)
                timestamp = datetime.fromisoformat(receipt.confirmed_at)
                result = AgentItemSnapshot(
                    item=binding.result_item,
                    source_revision=1,
                    kind="TOOL_RESULT",
                    status="COMPLETED",
                    payload=binding.result,
                    completed_at=timestamp,
                    call_id=binding.call.call_id,
                    tool_name=None,
                )
                message = ToolMessage(
                    content=binding.result.model_dump_json(),
                    tool_call_id=binding.call.call_id,
                    id=binding.result_item.id,
                )
                state = state.model_copy(
                    update={
                        "items": _replace_snapshot(state.items, result),
                        "messages": _append_messages(state.messages, (message,)),
                    }
                )
                state = self._emit(state, result)
        turn = state.active_turn
        if turn.unknown:
            snapshot = turn.snapshot
            if snapshot.status != "INTERRUPTED":
                snapshot = snapshot.model_copy(
                    update={
                        "status": "INTERRUPTED",
                        "source_revision": snapshot.source_revision + 1,
                        "error_code": "OUTCOME_UNKNOWN",
                        "error_message": "External outcome requires reconciliation",
                    }
                )
                state = self._turn_snapshot(state, snapshot)
            return {
                "session": state.model_copy(
                    update={
                        "active_run": state.active_run.model_copy(
                            update={"phase": "RECOVERY"}
                        )
                    }
                )
            }
        snapshot = turn.snapshot.model_copy(
            update={
                "status": "FAILED" if turn.fatal_error else "COMPLETED",
                "finished_at": datetime.now(UTC),
                "source_revision": turn.snapshot.source_revision + 1,
                "error_code": turn.fatal_error.code if turn.fatal_error else None,
                "error_message": turn.fatal_error.message if turn.fatal_error else None,
            }
        )
        return {"session": self._turn_snapshot(state, snapshot)}

    def _end_run(
        self,
        state: SessionAgentState,
        status: str,
        final: AgentItemIdentity | None = None,
        failure: ExecutionFailure | None = None,
    ) -> SessionAgentState:
        run = state.active_run.snapshot
        snapshot = run.model_copy(
            update={
                "status": status,
                "source_revision": run.source_revision + 1,
                "finished_at": datetime.now(UTC),
                "final_item": final,
                "error_code": failure.code if failure else None,
                "error_message": failure.message if failure else None,
            }
        )
        state = self._run_snapshot(state, snapshot)
        effects = tuple(
            effect.model_copy(update={"completed": True})
            if effect.resolved_run_id == run.run.id
            else effect
            for effect in state.control.effects
        )
        return state.model_copy(
            update={
                "active_run": None,
                "active_turn": None,
                "control": state.control.model_copy(
                    update={"effects": effects, "stop_requested": False}
                ),
            }
        )

    async def _decide(self, graph_state: GraphState) -> GraphState:
        state = self._admissions(self._typed(graph_state["session"]))
        turn = state.active_turn
        if turn.unknown:
            return {"session": state}
        if turn.fatal_error is not None:
            validate_context(state.messages)
            return {"session": self._end_run(state, "FAILED", failure=turn.fatal_error)}
        if state.control.stop_requested or turn.snapshot.status == "CANCELLED":
            return {"session": self._end_run(state, "CANCELLED")}
        if turn.tools or state.pending_inputs or state.control.pending_wake:
            return {
                "session": state.model_copy(
                    update={
                        "active_turn": None,
                        "active_run": state.active_run.model_copy(
                            update={"phase": "READY"}
                        ),
                    }
                )
            }
        final = next(
            (
                item.item
                for item in reversed(turn.model_items)
                if item.kind == "MESSAGE"
            ),
            None,
        )
        if final is None:
            return {
                "session": self._end_run(
                    state,
                    "FAILED",
                    failure=ExecutionFailure(
                        code="NO_FINAL_MESSAGE",
                        message="Model response supplied no final assistant message",
                    ),
                )
            }
        return {"session": self._end_run(state, "COMPLETED", final=final)}

    async def advance(self, session: str) -> None:
        lock = self._session_locks.setdefault(session, asyncio.Lock())
        async with lock:
            state = await self.state(session)
            commands = self.store.commands(session)
            if state is None:
                if not commands:
                    return
                state = SessionAgentState(
                    user_id=commands[0].command.user_id, session_id=session
                )
            await self._repair_markers(state)
            state = self._materialize(state)
            snapshot = await self._graph.aget_state(self._config(session))
            # Resume native pending steps using their original execution receipts.
            if snapshot.next:
                await self._graph.ainvoke(
                    None, self._config(session), durability="sync"
                )
            else:
                await self._graph.ainvoke(
                    {"session": state}, self._config(session), durability="sync"
                )
            state = await self.state(session)
            await self._repair_markers(state)
            # Transfer terminal facts before any successor can be prepared.
            for event in state.pending_events:
                self.store.append_fact(event)

    async def start(self) -> None:
        if self._dispatcher is not None:
            raise RuntimeError("Dispatcher is already started")
        self._dispatcher = asyncio.create_task(self._dispatch())

    async def _dispatch(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception as error:
                _LOGGER.error("Session dispatch deferred (%s)", type(error).__name__)
            await asyncio.sleep(0.05)

    async def tick(self) -> None:
        for session, task in tuple(self._tasks.items()):
            if task.done():
                try:
                    task.result()
                except Exception as error:
                    _LOGGER.error(
                        "Session advancement deferred (%s)", type(error).__name__
                    )
                del self._tasks[session]
        for session in self.store.session_ids():
            if session in self._tasks:
                continue
            state = await self.state(session)
            commands = self.store.commands(
                session, state.last_command_ordinal if state else 0
            )
            active = state is not None and state.active_run is not None
            queued_wake = state is not None and state.control.pending_wake is not None
            pending_facts = state is not None and bool(state.pending_events)
            recovery = active and state.active_run.phase == "RECOVERY"
            due = asyncio.get_running_loop().time()
            if recovery and not commands and due < self._recovery_due.get(session, 0):
                continue
            if commands or active or queued_wake or pending_facts:
                self._tasks[session] = asyncio.create_task(self.advance(session))
                if recovery:
                    self._recovery_due[session] = due + 5

    async def stop(self) -> None:
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            await asyncio.gather(self._dispatcher, return_exceptions=True)
            self._dispatcher = None
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
