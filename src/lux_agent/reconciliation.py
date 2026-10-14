"""Authenticated operator attestation of an original, previously lost response ID."""

import asyncio
import os
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import Field, StrictStr

from lux_agent.protocol import ResourceId, WireModel
from lux_agent.store import AgentStore, ModelProviderBinding


@dataclass(frozen=True)
class ReconciliationSettings:
    enabled: bool = False
    bearer_token: str = field(default="", repr=False)
    operator_id: str = ""
    allowed_accounts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.enabled and (
            not self.bearer_token.strip()
            or not self.operator_id.strip()
            or not self.allowed_accounts
            or any(not account.strip() for account in self.allowed_accounts)
        ):
            raise ValueError(
                "Reconciliation requires token, trusted operator and account scope"
            )

    @classmethod
    def from_environment(cls) -> ReconciliationSettings:
        enabled = os.environ.get("LUX_AGENT_RECONCILIATION_ENABLED", "false").lower()
        if enabled not in ("true", "false"):
            raise ValueError("Reconciliation enabled flag must be true or false")
        return cls(
            enabled=enabled == "true",
            bearer_token=os.environ.get("LUX_AGENT_RECONCILIATION_TOKEN", ""),
            operator_id=os.environ.get("LUX_AGENT_RECONCILIATION_OPERATOR", ""),
            allowed_accounts=tuple(
                account.strip()
                for account in os.environ.get(
                    "LUX_AGENT_RECONCILIATION_ACCOUNTS", ""
                ).split(",")
                if account.strip()
            ),
        )


class OriginalModelEvidence(WireModel):
    binding: ModelProviderBinding
    run_id: ResourceId
    turn_id: ResourceId
    response_id: Annotated[StrictStr, Field(pattern=r"^resp_[A-Za-z0-9_-]{1,1019}$")]
    evidence_reference: Annotated[StrictStr, Field(min_length=1, max_length=2048)]
    reason: Annotated[StrictStr, Field(min_length=1, max_length=2048)]


class ModelEvidenceView(WireModel):
    binding: ModelProviderBinding
    run_id: ResourceId
    turn_id: ResourceId
    state: str
    response_id: str | None


class ModelEvidencePage(WireModel):
    executions: tuple[ModelEvidenceView, ...]


def create_reconciliation_router(
    settings: ReconciliationSettings,
    store_provider: Callable[[], AgentStore | None],
    resume: Callable[[str], Awaitable[None]] | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/internal/model-reconciliation", include_in_schema=False)

    def authenticated(
        authorization: Annotated[str | None, Header()] = None,
    ) -> AgentStore:
        if not settings.enabled:
            raise HTTPException(404, "Not found")
        supplied = (
            authorization.removeprefix("Bearer ")
            if authorization and authorization.startswith("Bearer ")
            else ""
        )
        if not secrets.compare_digest(
            supplied.encode(), settings.bearer_token.encode()
        ):
            raise HTTPException(401, "Unauthorized")
        store = store_provider()
        if store is None:
            raise HTTPException(503, "Agent persistence unavailable")
        return store

    @router.get("")
    async def unresolved(
        store: AgentStore = Depends(authenticated),
    ) -> ModelEvidencePage:
        views = await asyncio.to_thread(
            store.unresolved_model_evidence, settings.allowed_accounts
        )
        return ModelEvidencePage(executions=views)

    @router.get("/{execution_id}")
    async def detail(
        execution_id: str, store: AgentStore = Depends(authenticated)
    ) -> ModelEvidenceView:
        view = await asyncio.to_thread(
            store.model_evidence_view, execution_id, settings.allowed_accounts
        )
        if view is None:
            raise HTTPException(404, "Execution not found")
        return view

    @router.post("/{execution_id}/original-receipt")
    async def attach(
        execution_id: str,
        evidence: OriginalModelEvidence,
        store: AgentStore = Depends(authenticated),
    ) -> ModelEvidenceView:
        if evidence.binding.execution_id != execution_id:
            raise HTTPException(409, "Original execution binding differs")
        try:
            view = await asyncio.to_thread(
                store.attach_model_evidence,
                evidence,
                settings.operator_id,
                settings.allowed_accounts,
            )
        except ValueError:
            raise HTTPException(
                409, "Original evidence conflicts with saved execution"
            ) from None
        if resume is not None:
            await resume(view.binding.session_id)
        return view

    return router
