"""Agent-owned ChatOpenAI execution and retrieval of the original response only."""

import json
from collections.abc import AsyncIterator
from typing import Annotated, Literal

import httpx
from langchain_core.messages import AIMessage, AIMessageChunk, message_chunk_to_message
from langchain_openai import ChatOpenAI
from openai import APIError, APIStatusError, AsyncOpenAI
from openai.types.responses import Response
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from lux_agent.execution import (
    ConfirmedExecutionFailure,
    ExecutionUnknown,
    LiveOutputSlot,
    MediaObserver,
    ModelRequest,
    ModelResponseObserved,
    ModelTextFragment,
    ObservedModelItem,
)
from lux_agent.protocol import (
    PLAN_CALL_TYPES,
    GetTaskCallContent,
    OrderedContent,
    SubmitGenerationCallContent,
    TextBlock,
    ToolName,
)
from lux_agent.store import AgentStore, ModelProviderBinding, content_hash

MODEL = "gpt-6-luna"
ENDPOINT = "https://api.openai.com/v1"


class ProviderBlock(BaseModel):
    # LangChain may attach annotations, indexes and opaque provider extensions.
    model_config = ConfigDict(extra="ignore", frozen=True)


class ProviderText(ProviderBlock):
    type: Literal["text"]
    text: str = ""


class ProviderSummary(ProviderBlock):
    type: Literal["summary_text"]
    text: str


class ProviderReasoning(ProviderBlock):
    type: Literal["reasoning"]
    summary: tuple[ProviderSummary, ...] = ()


class ProviderFunctionCall(ProviderBlock):
    type: Literal["function_call"]
    name: ToolName
    call_id: str = Field(min_length=1)
    arguments: str


class ProviderRefusal(ProviderBlock):
    type: Literal["refusal"]
    refusal: str


ProviderContent = Annotated[
    ProviderText | ProviderReasoning | ProviderFunctionCall | ProviderRefusal,
    Field(discriminator="type"),
]
_CONTENT = TypeAdapter(tuple[ProviderContent, ...])
_CALL_TYPES = {
    **PLAN_CALL_TYPES,
    "get_task": GetTaskCallContent,
    "submit_generation": SubmitGenerationCallContent,
}


def observed_message(message: AIMessage) -> ModelResponseObserved:
    """Decode known provider fields and keep its original LangChain message intact."""
    raw_content = message.content
    blocks = (
        _CONTENT.validate_python(raw_content)
        if isinstance(raw_content, list)
        else (ProviderText(type="text", text=raw_content),)
    )
    if message.invalid_tool_calls:
        raise ConfirmedExecutionFailure(
            "MODEL_RESPONSE_INVALID", "Invalid model tool arguments"
        )
    items: list[ObservedModelItem] = []
    texts: list[TextBlock] = []
    summaries: list[TextBlock] = []
    text_index: int | None = None
    reasoning_index: int | None = None
    calls: list[ProviderFunctionCall] = []
    for block in blocks:
        if isinstance(block, (ProviderText, ProviderRefusal)):
            text = block.text if isinstance(block, ProviderText) else block.refusal
            if text:
                if text_index is None:
                    text_index = len(items)
                    items.append(
                        ObservedModelItem(
                            slot="answer",
                            kind="MESSAGE",
                            payload=OrderedContent(blocks=()),
                        )
                    )
                texts.append(TextBlock(text=text))
        elif isinstance(block, ProviderReasoning):
            summary = tuple(
                TextBlock(text=part.text) for part in block.summary if part.text
            )
            if summary:
                if reasoning_index is None:
                    reasoning_index = len(items)
                    items.append(
                        ObservedModelItem(
                            slot="reasoning",
                            kind="REASONING_SUMMARY",
                            payload=OrderedContent(blocks=()),
                        )
                    )
                summaries.extend(summary)
        elif isinstance(block, ProviderFunctionCall):
            calls.append(block)
            content_type = _CALL_TYPES[block.name]
            payload = content_type.model_validate_json(
                '{"arguments":' + block.arguments + "}"
            )
            items.append(
                ObservedModelItem(
                    kind="TOOL_CALL",
                    call_id=block.call_id,
                    tool_name=block.name,
                    payload=payload,
                )
            )
    if text_index is not None:
        items[text_index] = items[text_index].model_copy(
            update={"payload": OrderedContent(blocks=tuple(texts))}
        )
    if reasoning_index is not None:
        items[reasoning_index] = items[reasoning_index].model_copy(
            update={"payload": OrderedContent(blocks=tuple(summaries))}
        )
    if len(calls) != len(message.tool_calls) or any(
        call.call_id != actual["id"]
        or call.name != actual["name"]
        or json.loads(call.arguments) != actual["args"]
        for call, actual in zip(calls, message.tool_calls, strict=True)
    ):
        raise ConfirmedExecutionFailure(
            "MODEL_RESPONSE_INVALID", "Model tool bindings disagree"
        )
    return ModelResponseObserved(
        items=tuple(items),
        provider_response=message.model_dump(mode="json"),
        messages=(message,),
    )


def retrieved_message(response: Response) -> AIMessage:
    """Use public typed OpenAI output models, without private LangChain converters."""
    content: list[dict] = []  # Required external LangChain content-block boundary.
    calls: list[dict] = []  # Required external LangChain tool-call boundary.
    for output in response.output:
        if output.type == "message":
            for block in output.content:
                if block.type == "output_text":
                    content.append(
                        {
                            "type": "text",
                            "text": block.text,
                            "id": output.id,
                            "annotations": [
                                a.model_dump(mode="json") for a in block.annotations
                            ],
                        }
                    )
                elif block.type == "refusal":
                    content.append(
                        {"type": "refusal", "refusal": block.refusal, "id": output.id}
                    )
        elif output.type == "reasoning":
            content.append(output.model_dump(mode="json", exclude_none=True))
        elif output.type == "function_call":
            content.append(output.model_dump(mode="json", exclude_none=True))
            calls.append(
                {
                    "name": output.name,
                    "id": output.call_id,
                    "args": json.loads(output.arguments),
                    "type": "tool_call",
                }
            )
        else:
            raise ConfirmedExecutionFailure(
                "MODEL_RESPONSE_INVALID", "Unsupported model output"
            )
    return AIMessage(
        content=content,
        tool_calls=calls,
        id=response.id,
        response_metadata={
            "id": response.id,
            "status": response.status,
            "model": response.model,
            "model_provider": "openai",
        },
    )


class OpenAIModelExecutor:
    """One submission, durable earliest exposed ID, and original-account retrieval."""

    def __init__(
        self,
        store: AgentStore,
        *,
        api_key: str,
        account_ref: str,
        http_client: httpx.AsyncClient,
        sync_http_client: httpx.Client | None = None,
        media_observer: MediaObserver | None = None,
    ) -> None:
        if not api_key or not account_ref:
            raise ValueError("OpenAI key and account reference are required")
        self.media_observer = media_observer
        self.store = store
        self.account_ref = account_ref
        self.provider = AsyncOpenAI(
            api_key=api_key, base_url=ENDPOINT, max_retries=0, http_client=http_client
        )
        self.model = ChatOpenAI(
            model=MODEL,
            api_key=api_key,
            base_url=ENDPOINT,
            use_responses_api=True,
            output_version="responses/v1",
            max_retries=0,
            use_previous_response_id=False,
            truncation="disabled",
            store=True,
            reasoning={"summary": "auto"},
            http_async_client=http_client,
            http_client=sync_http_client,
            organization=self.provider.organization,
            default_headers={"OpenAI-Project": self.provider.project}
            if self.provider.project
            else None,
        )

    @property
    def live_slots(self) -> tuple[LiveOutputSlot, ...]:
        return (
            LiveOutputSlot(key="answer", kind="MESSAGE"),
            LiveOutputSlot(key="reasoning", kind="REASONING_SUMMARY"),
        )

    def _binding(self, request: ModelRequest) -> ModelProviderBinding:
        return ModelProviderBinding(
            execution_id=request.execution_id,
            user_id=request.user_id,
            session_id=request.session_id,
            request_hash=content_hash(request.model_dump_json()),
            account_ref=self.account_ref,
            model=MODEL,
            endpoint=ENDPOINT,
            organization_id=self.provider.organization,
            project_id=self.provider.project,
        )

    async def stream(
        self, request: ModelRequest
    ) -> AsyncIterator[ModelTextFragment | ModelResponseObserved]:
        binding = self._binding(request)
        if self.store.model_provider_receipt(request.execution_id) is not None:
            raise ExecutionUnknown("Original model submission requires reconciliation")
        # Establish SDK ownership before any cancellable hydration; recovery can
        # conclusively distinguish this adapter's undispatched work from acceptance.
        self.store.prepare_model_provider(binding, dispatch_started=False)
        try:
            if request.media_observations and self.media_observer is None:
                raise ValueError("Media observer is required for this request")
            messages = (
                await self.media_observer.invocation_messages(request)
                if self.media_observer and request.media_observations
                else request.messages
            )
        except ValueError:
            raise ConfirmedExecutionFailure(
                "MEDIA_OBSERVATION_FAILED",
                "Media could not be verified before model submission",
            ) from None
        # The tool list replaces the current model-visible set; history is untouched.
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in request.tools
        ]
        model = self.model.bind_tools(tools) if tools else self.model
        aggregate: AIMessageChunk | None = None
        completed = False
        known_id: str | None = None
        self.store.start_model_provider(binding)
        try:
            async for chunk in model.astream(
                list(messages),
                extra_headers={"X-Client-Request-Id": request.execution_id},
            ):
                response_id = chunk.response_metadata.get("id")
                if isinstance(response_id, str):
                    self.store.observe_model_provider(binding, response_id)
                    known_id = response_id
                if known_id is None:
                    raise ExecutionUnknown(
                        "Provider output preceded original response identity"
                    )
                aggregate = chunk if aggregate is None else aggregate + chunk
                if chunk.chunk_position == "last":
                    status = chunk.response_metadata.get("status")
                    if status != "completed":
                        raise ConfirmedExecutionFailure(
                            "MODEL_RESPONSE_INCOMPLETE",
                            "Original model response did not complete",
                        )
                    completed = True
                # Partial function arguments stay buffered until completion.
                if isinstance(chunk.content, list):
                    for raw in chunk.content:
                        if isinstance(raw, dict) and raw.get("type") == "text":
                            text = ProviderText.model_validate(raw).text
                            if text:
                                yield ModelTextFragment(slot="answer", text=text)
                        elif isinstance(raw, dict) and raw.get("type") == "reasoning":
                            for part in ProviderReasoning.model_validate(raw).summary:
                                if part.text:
                                    yield ModelTextFragment(
                                        slot="reasoning", text=part.text
                                    )
        except APIStatusError as error:
            if (
                known_id is None
                and error.status_code in (400, 401, 403)
                and error.request.method == "POST"
                and error.request.url.path == "/v1/responses"
            ):
                raise ConfirmedExecutionFailure(
                    "MODEL_PROVIDER_REJECTED",
                    "Provider rejected the original model request before acceptance",
                ) from None
            raise ExecutionUnknown(
                "Original model request requires reconciliation"
            ) from None
        except APIError as error:
            raise ExecutionUnknown(
                "Original model request requires reconciliation"
            ) from error
        if not completed or aggregate is None:
            raise ExecutionUnknown("Original model stream ended before completion")
        try:
            observed = observed_message(message_chunk_to_message(aggregate))
        except ValueError:
            raise ConfirmedExecutionFailure(
                "MODEL_RESPONSE_INVALID", "Original model output failed validation"
            ) from None
        yield observed

    async def reconcile(self, request: ModelRequest) -> ModelResponseObserved | None:
        saved = self.store.model_provider_receipt(request.execution_id)
        if saved is None:
            return None
        if saved.binding != self._binding(request):
            raise ValueError("Conflicting model provider recovery binding")
        if not saved.dispatch_started:
            raise ConfirmedExecutionFailure(
                "MODEL_NOT_SUBMITTED",
                "Original model execution was interrupted before provider submission",
            )
        if saved.response_id is None:
            return None
        try:
            response = await self.provider.responses.retrieve(saved.response_id)
        except APIError:
            return None
        if response.id != saved.response_id:
            raise ValueError("Conflicting retrieved provider response identity")
        if response.status in ("failed", "cancelled", "incomplete"):
            raise ConfirmedExecutionFailure(
                "MODEL_PROVIDER_FAILED",
                "Original model response failed or was incomplete",
            )
        if response.status != "completed":
            return None
        try:
            return observed_message(retrieved_message(response))
        except ValueError:
            raise ConfirmedExecutionFailure(
                "MODEL_RESPONSE_INVALID", "Original model output failed validation"
            ) from None
