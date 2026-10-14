"""Durable pre-dispatch facts distinguish local hydration from paid uncertainty."""

import sqlite3

import pytest
from test_openai_model import prepare, request

from lux_agent.reconciliation import OriginalModelEvidence
from lux_agent.store import AgentStore, ModelProviderBinding, content_hash


def binding_for(req):
    return ModelProviderBinding(
        execution_id=req.execution_id,
        user_id=req.user_id,
        session_id=req.session_id,
        request_hash=content_hash(req.model_dump_json()),
        account_ref="openai-primary",
        model="gpt-6-luna",
        endpoint="https://api.openai.com/v1",
    )


def test_undispatched_phase_reopens_and_cannot_attach_or_observe_provider_evidence(
    tmp_path,
):
    path = tmp_path / "agent.db"
    req = request()
    binding = binding_for(req)
    with AgentStore(path) as store:
        prepare(store, req)
        store.prepare_model_provider(binding, dispatch_started=False)
        assert not store.model_provider_receipt(req.execution_id).dispatch_started
        store.execution_unknown(req.execution_id)
    with AgentStore(path) as store:
        assert not store.model_provider_receipt(req.execution_id).dispatch_started
        assert store.unresolved_model_evidence(("openai-primary",)) == ()
        with pytest.raises(ValueError):
            store.observe_model_provider(binding, "resp_impossible")
        evidence = OriginalModelEvidence(
            binding=binding,
            run_id=req.turn.run.id,
            turn_id=req.turn.id,
            response_id="resp_impossible",
            evidence_reference="fixture",
            reason="fixture",
        )
        with pytest.raises(ValueError):
            store.attach_model_evidence(evidence, "operator", ("openai-primary",))


def test_dispatch_marker_requires_original_inflight_execution_and_persists(tmp_path):
    req = request()
    binding = binding_for(req)
    with AgentStore(tmp_path / "agent.db") as store:
        prepare(store, req)
        store.prepare_model_provider(binding, dispatch_started=False)
        store.start_model_provider(binding)
        assert store.model_provider_receipt(req.execution_id).dispatch_started
        with pytest.raises(ValueError):
            store.start_model_provider(binding)
        store.observe_model_provider(binding, "resp_original")
        assert (
            store.model_provider_receipt(req.execution_id).response_id
            == "resp_original"
        )


def test_old_provider_receipts_upgrade_as_potentially_dispatched_without_reset(
    tmp_path,
):
    path = tmp_path / "agent.db"
    req = request()
    binding = binding_for(req)
    with AgentStore(path) as store:
        prepare(store, req)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE model_provider_receipt")
        connection.execute(
            "CREATE TABLE model_provider_receipt (execution_id TEXT PRIMARY KEY, "
            "binding TEXT NOT NULL, response_id TEXT, created_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO model_provider_receipt VALUES (?,?,NULL,'2026-10-13')",
            (req.execution_id, binding.model_dump_json()),
        )
    with AgentStore(path) as store:
        assert store.model_provider_receipt(req.execution_id).dispatch_started
        assert store.execution(req.execution_id).state == "IN_FLIGHT"
