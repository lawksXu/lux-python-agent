from concurrent.futures import ThreadPoolExecutor

import pytest
from test_openai_model import prepare, request

from lux_agent.store import AgentStore, ModelProviderBinding, content_hash


def missing(store):
    req = request()
    prepare(store, req)
    binding = ModelProviderBinding(
        execution_id=req.execution_id,
        user_id=req.user_id,
        session_id=req.session_id,
        request_hash=content_hash(req.model_dump_json()),
        account_ref="test-account",
        model="gpt-6-luna",
        endpoint="https://api.openai.com/v1",
    )
    store.prepare_model_provider(binding)
    store.execution_unknown(req.execution_id)
    return req, binding


def evidence(req, binding):
    from lux_agent.reconciliation import OriginalModelEvidence

    return OriginalModelEvidence(
        binding=binding,
        run_id=req.turn.run.id,
        turn_id=req.turn.id,
        response_id="resp_original",
        evidence_reference="ticket:original-receipt",
        reason="Original receipt recovered from trusted operator evidence",
    )


def test_atomic_idempotent_attachment_and_restart(tmp_path):
    with AgentStore(tmp_path / "agent.db") as store:
        req, binding = missing(store)
        assert hasattr(store, "attach_model_evidence"), "Audited attachment is missing"
        attached = evidence(req, binding)
        with ThreadPoolExecutor(max_workers=4) as workers:
            views = list(
                workers.map(
                    lambda _: store.attach_model_evidence(
                        attached, "operator", ("test-account",)
                    ),
                    range(4),
                )
            )
        assert all(view.response_id == "resp_original" for view in views)
        assert (
            store._connection.execute(
                "SELECT COUNT(*) FROM model_reconciliation_audit"
            ).fetchone()[0]
            == 1
        )
        assert store.execution(req.execution_id).state == "UNKNOWN"
    with AgentStore(tmp_path / "agent.db") as store:
        assert (
            store.model_provider_receipt(req.execution_id).response_id
            == "resp_original"
        )
        assert (
            store.attach_model_evidence(
                attached, "operator", ("test-account",)
            ).response_id
            == "resp_original"
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("account_ref", "other"),
        ("organization_id", "other"),
        ("project_id", "other"),
        ("session_id", "00000000-0000-0000-0000-000000000099"),
        ("request_hash", "0" * 64),
        ("model", "other"),
        ("endpoint", "https://other"),
        ("user_id", "00000000-0000-0000-0000-000000000099"),
    ],
)
def test_binding_mismatch_rejected(tmp_path, field, value):
    with AgentStore(tmp_path / "agent.db") as store:
        req, binding = missing(store)
        with pytest.raises(ValueError):
            store.attach_model_evidence(
                evidence(req, binding.model_copy(update={field: value})),
                "operator",
                ("test-account",),
            )
        assert store.model_provider_receipt(req.execution_id).response_id is None


def test_immutable_terminal_and_audit_failure(tmp_path):
    with AgentStore(tmp_path / "agent.db") as store:
        req, binding = missing(store)
        attached = evidence(req, binding)
        store._connection.execute(
            "CREATE TRIGGER reject_audit BEFORE INSERT ON "
            "model_reconciliation_audit BEGIN SELECT "
            "RAISE(ABORT, 'audit unavailable'); END"
        )
        with pytest.raises(Exception, match="audit unavailable"):
            store.attach_model_evidence(attached, "operator", ("test-account",))
        assert store.model_provider_receipt(req.execution_id).response_id is None
        store._connection.execute("DROP TRIGGER reject_audit")
        store.attach_model_evidence(attached, "operator", ("test-account",))
        with pytest.raises(ValueError):
            store.attach_model_evidence(
                attached.model_copy(update={"response_id": "resp_other"}),
                "operator",
                ("test-account",),
            )
        store.execution_confirmed(req.execution_id, '{"result":"done"}')
        assert (
            store.attach_model_evidence(
                attached, "operator", ("test-account",)
            ).response_id
            == "resp_original"
        )
        with pytest.raises(ValueError):
            store.attach_model_evidence(
                attached.model_copy(update={"evidence_reference": "ticket:other"}),
                "operator",
                ("test-account",),
            )


def test_operator_routes_authenticate_before_payload_and_scope(tmp_path):
    import asyncio

    import httpx
    from fastapi import FastAPI

    from lux_agent.reconciliation import (
        ReconciliationSettings,
        create_reconciliation_router,
    )

    async def scenario():
        with AgentStore(tmp_path / "agent.db") as store:
            req, binding = missing(store)
            disabled = FastAPI()
            disabled.include_router(
                create_reconciliation_router(ReconciliationSettings(), lambda: store)
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=disabled), base_url="http://test"
            ) as client:
                assert (
                    await client.post(
                        f"/internal/model-reconciliation/{req.execution_id}/original-receipt",
                        json={},
                    )
                ).status_code == 404
            active = FastAPI()
            settings = ReconciliationSettings(
                True, "secret", "trusted-operator", ("test-account",)
            )
            active.include_router(create_reconciliation_router(settings, lambda: store))
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=active), base_url="http://test"
            ) as client:
                assert (
                    await client.get("/internal/model-reconciliation")
                ).status_code == 401
                assert (
                    await client.post(
                        f"/internal/model-reconciliation/{req.execution_id}/original-receipt",
                        json={},
                    )
                ).status_code == 401
                headers = {"Authorization": "Bearer secret"}
                response = await client.get(
                    "/internal/model-reconciliation", headers=headers
                )
                assert response.status_code == 200
                assert "request_snapshot" not in response.text
                body = evidence(req, binding).model_dump(mode="json")
                body["operator_id"] = "untrusted"
                assert (
                    await client.post(
                        f"/internal/model-reconciliation/{req.execution_id}/original-receipt",
                        headers=headers,
                        json=body,
                    )
                ).status_code == 422
                body.pop("operator_id")
                assert (
                    await client.post(
                        f"/internal/model-reconciliation/{req.execution_id}/original-receipt",
                        headers=headers,
                        json=body,
                    )
                ).status_code == 200
                assert (
                    store._connection.execute(
                        "SELECT operator_id FROM model_reconciliation_audit"
                    ).fetchone()[0]
                    == "trusted-operator"
                )

    asyncio.run(scenario())


def test_attached_original_response_is_queried_after_restart_without_post(tmp_path):
    import asyncio

    import httpx
    from test_openai_model import response

    from lux_agent.openai_model import ENDPOINT, OpenAIModelExecutor

    async def scenario():
        with AgentStore(tmp_path / "agent.db") as store:
            req = request()
            prepare(store, req)
            binding = ModelProviderBinding(
                execution_id=req.execution_id,
                user_id=req.user_id,
                session_id=req.session_id,
                request_hash=content_hash(req.model_dump_json()),
                account_ref="test-account",
                model="gpt-6-luna",
                endpoint=ENDPOINT,
            )
            store.prepare_model_provider(binding)
            store.execution_unknown(req.execution_id)
            store.attach_model_evidence(
                evidence(req, binding), "operator", ("test-account",)
            )
        calls = []

        def handler(request):
            calls.append((request.method, request.url.path))
            return httpx.Response(200, json=response())

        with AgentStore(tmp_path / "agent.db") as store:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline",
                    account_ref="test-account",
                    http_client=client,
                )
                observed = await executor.reconcile(req)
                assert observed is not None
                assert calls == [("GET", "/v1/responses/resp_original")]
                assert store.execution(req.execution_id).state == "UNKNOWN"

    asyncio.run(scenario())
