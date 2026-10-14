"""Optional local durable worker lifespan and a lightweight liveness endpoint."""

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from lux_agent.backend import BackendClient
from lux_agent.delivery import CommandTransport, DeliveryWorker, KafkaTransport
from lux_agent.execution import (
    ModelExecutor,
    ModelReconciler,
    PlanContextReader,
    ToolExecutor,
)
from lux_agent.media_observation import JavaMediaObserver
from lux_agent.openai_model import OpenAIModelExecutor
from lux_agent.protocol import ResourceId, SessionExecutionSync
from lux_agent.reconciliation import (
    ReconciliationSettings,
    create_reconciliation_router,
)
from lux_agent.runtime import AgentRuntime
from lux_agent.store import AgentStore


@dataclass(frozen=True)
class ServiceSettings:
    store_path: Path | None = None
    backend_url: str | None = None
    kafka_bootstrap_servers: str | None = None
    kafka_group_id: str = "lux-agent-v2"
    openai_enabled: bool = False
    openai_account_ref: str = "openai-primary"
    media_observation_enabled: bool = True
    reconciliation: ReconciliationSettings = ReconciliationSettings()

    def __post_init__(self) -> None:
        if self.reconciliation.enabled and self.store_path is None:
            raise ValueError("Reconciliation requires Agent persistence")

    @classmethod
    def from_environment(cls) -> ServiceSettings:
        path = os.environ.get("LUX_AGENT_STORE")
        return cls(
            store_path=Path(path) if path else None,
            backend_url=os.environ.get("LUX_BACKEND_URL"),
            kafka_bootstrap_servers=os.environ.get("LUX_KAFKA_BOOTSTRAP_SERVERS"),
            kafka_group_id=os.environ.get("LUX_KAFKA_GROUP_ID", "lux-agent-v2"),
            openai_enabled=os.environ.get("LUX_OPENAI_ENABLED", "false").lower()
            == "true",
            openai_account_ref=os.environ.get(
                "LUX_OPENAI_ACCOUNT_REF", "openai-primary"
            ),
            media_observation_enabled=os.environ.get(
                "LUX_AGENT_MEDIA_OBSERVATION_ENABLED", "true"
            ).lower()
            == "true",
            reconciliation=ReconciliationSettings.from_environment(),
        )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


def create_app(
    settings: ServiceSettings | None = None,
    *,
    executor: ModelExecutor | None = None,
    model_reconciler: ModelReconciler | None = None,
    tools: ToolExecutor | None = None,
    plan_context_reader: PlanContextReader | None = None,
    transport: CommandTransport | None = None,
) -> FastAPI:
    configuration = settings or ServiceSettings.from_environment()
    runtime: AgentRuntime | None = None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        nonlocal runtime
        if configuration.store_path is None:
            if (
                configuration.kafka_bootstrap_servers is not None
                or transport is not None
            ):
                raise RuntimeError("Kafka workers require LUX_AGENT_STORE")
            yield
            return
        async with AsyncExitStack() as stack:
            store = stack.enter_context(AgentStore(configuration.store_path))
            active_executor = executor
            active_reconciler = model_reconciler
            api_key = os.environ.get("OPENAI_API_KEY")
            backend = None
            observer = None
            if configuration.backend_url is not None:
                client = await stack.enter_async_context(
                    httpx.AsyncClient(base_url=configuration.backend_url, timeout=10)
                )
                backend = BackendClient(client)
                if configuration.media_observation_enabled and (
                    active_executor is not None
                    or configuration.openai_enabled
                    and api_key
                ):
                    observer = JavaMediaObserver(client)
            if active_executor is None and configuration.openai_enabled and api_key:
                model_client = await stack.enter_async_context(
                    httpx.AsyncClient(timeout=60)
                )
                sync_model_client = stack.enter_context(httpx.Client(timeout=60))
                adapter = OpenAIModelExecutor(
                    store,
                    api_key=api_key,
                    account_ref=configuration.openai_account_ref,
                    http_client=model_client,
                    sync_http_client=sync_model_client,
                    media_observer=observer,
                )
                active_executor = adapter
                active_reconciler = adapter
            runtime = await stack.enter_async_context(
                AgentRuntime(
                    store,
                    executor=active_executor,
                    model_reconciler=active_reconciler,
                    tools=tools or backend,
                    repair=backend,
                    tool_reconciler=backend,
                    plan_context_reader=plan_context_reader or backend,
                    media_observer=observer,
                )
            )
            application.state.runtime = runtime
            application.state.media_observer = observer
            delivery_task = None
            wire = transport
            if wire is None and configuration.kafka_bootstrap_servers is not None:
                wire = KafkaTransport(
                    configuration.kafka_bootstrap_servers, configuration.kafka_group_id
                )
                await wire.start()
            if wire is not None:
                delivery_task = asyncio.create_task(DeliveryWorker(store, wire).run())
            await runtime.start()
            try:
                yield
            finally:
                await runtime.stop()
                if delivery_task is not None:
                    delivery_task.cancel()
                    await asyncio.gather(delivery_task, return_exceptions=True)
                application.state.runtime = None
                application.state.media_observer = None
                runtime = None

    application = FastAPI(title="Lux Python Agent", version="0.2.0", lifespan=lifespan)
    application.state.runtime = None
    application.state.media_observer = None
    application.include_router(
        create_reconciliation_router(
            configuration.reconciliation,
            lambda: runtime.store if runtime is not None else None,
        )
    )

    @application.get("/health", tags=["health"])
    async def health() -> HealthResponse:
        return HealthResponse()

    @application.get(
        "/internal/v2/users/{owner}/sessions/{session}/execution",
        response_model=SessionExecutionSync,
    )
    async def execution_sync(
        owner: ResourceId,
        session: ResourceId,
        after_event_seq: Annotated[int, Query(ge=0)] = 0,
        through_event_seq: Annotated[int | None, Query(ge=0)] = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> SessionExecutionSync:
        if runtime is None:
            raise HTTPException(503, "Durable execution runtime is not enabled")
        try:
            return runtime.store.sync(
                owner, session, after_event_seq, through_event_seq, limit
            )
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    return application


app = create_app()
