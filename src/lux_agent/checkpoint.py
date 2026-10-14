"""Restore Lux state at the native checkpointer's opaque serialization boundary."""

from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from lux_agent.state import SessionAgentState


class SessionStateSerializer(JsonPlusSerializer):
    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        # JsonPlus uses model_construct for Pydantic models. Validate our root
        # state before LangGraph reserializes it or passes it to graph nodes.
        return self._restore(super().loads_typed(data))

    def _restore(self, value: Any) -> Any:
        # These containers belong to LangGraph's checkpoint/writes protocol.
        if isinstance(value, SessionAgentState):
            return SessionAgentState.model_validate(vars(value), strict=False)
        if isinstance(value, dict):
            return {key: self._restore(member) for key, member in value.items()}
        if isinstance(value, list):
            return [self._restore(member) for member in value]
        if isinstance(value, tuple):
            return tuple(self._restore(member) for member in value)
        return value
