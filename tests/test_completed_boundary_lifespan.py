"""Production lifespan connects compact media observation and audited recovery."""

import asyncio
import json

import httpx
from test_media_observation import ASSET, image_bytes, java_handler
from test_openai_model import response
from test_protocol_v2 import SESSION, USER, input_json
from test_runtime_v2 import ReplyExecutor


def test_lifespan_prepares_authorized_media_for_the_active_executor(
    tmp_path, monkeypatch
):
    async def scenario():
        import lux_agent.main as main
        from lux_agent.store import AgentStore

        body = image_bytes()
        handle_asset = java_handler(body)

        def handler(incoming):
            if incoming.url.path.endswith("/generation-plan-context"):
                return httpx.Response(
                    200,
                    json={
                        "focus": {
                            "session_id": SESSION,
                            "active_plan_id": None,
                            "revision": 0,
                        },
                        "plan": None,
                        "catalog": {
                            "type": "model_catalog_snapshot",
                            "catalog_revision": 1,
                            "submission_policy": {"maximum_tasks_per_submission": 32},
                            "models": [],
                        },
                    },
                )
            return handle_asset(incoming)

        original_client = httpx.AsyncClient

        class OfflineClient(original_client):
            def __init__(self, **kwargs):
                kwargs.setdefault("transport", httpx.MockTransport(handler))
                super().__init__(**kwargs)

        monkeypatch.setattr(main.httpx, "AsyncClient", OfflineClient)
        raw = json.loads(input_json())
        raw["item"]["content"]["blocks"].append({"type": "asset", "asset_id": ASSET})
        path = tmp_path / "agent.db"
        with AgentStore(path) as store:
            store.admit(json.dumps(raw))
        executor = ReplyExecutor()
        app = main.create_app(
            main.ServiceSettings(store_path=path, backend_url="http://java"),
            executor=executor,
        )
        async with app.router.lifespan_context(app):
            async with asyncio.timeout(3):
                while (
                    not app.state.runtime.store.sync(USER, SESSION).runs
                    or app.state.runtime.store.sync(USER, SESSION).runs[-1].status
                    != "COMPLETED"
                ):
                    await asyncio.sleep(0.01)
            assert executor.requests[0].media_observations
            assert app.state.runtime.media_observer is app.state.media_observer
            assert executor.requests[0].media_observations[0].asset.id == ASSET
        assert app.state.media_observer is None

    asyncio.run(scenario())


def test_lifespan_attachment_resumes_only_the_original_response(tmp_path, monkeypatch):
    async def scenario():
        import lux_agent.main as main
        from lux_agent.reconciliation import ReconciliationSettings
        from lux_agent.store import AgentStore

        requests = []
        message = {
            "type": "message",
            "id": "msg_original",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "Recovered", "annotations": []}
            ],
        }

        def handler(incoming):
            requests.append(incoming)
            if incoming.method == "POST":
                return httpx.Response(
                    503,
                    json={
                        "error": {
                            "message": "fixture unavailable",
                            "type": "server_error",
                        }
                    },
                )
            assert incoming.method == "GET" and incoming.url.path.endswith(
                "/resp_original"
            )
            return httpx.Response(200, json=response(output=[message]))

        original_client = httpx.AsyncClient

        class OfflineClient(original_client):
            def __init__(self, **kwargs):
                kwargs.setdefault("transport", httpx.MockTransport(handler))
                super().__init__(**kwargs)

        monkeypatch.setattr(main.httpx, "AsyncClient", OfflineClient)
        monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture")
        path = tmp_path / "agent.db"
        with AgentStore(path) as store:
            store.admit(input_json())
        app = main.create_app(
            main.ServiceSettings(
                store_path=path,
                openai_enabled=True,
                reconciliation=ReconciliationSettings(
                    True, "fixture-token", "trusted-operator", ("openai-primary",)
                ),
            )
        )
        async with app.router.lifespan_context(app):
            async with original_client(
                transport=httpx.ASGITransport(app=app), base_url="http://agent"
            ) as client:
                assert (
                    await client.get("/internal/model-reconciliation")
                ).status_code == 401
                headers = {"authorization": "Bearer fixture-token"}
                async with asyncio.timeout(3):
                    while True:
                        page = await client.get(
                            "/internal/model-reconciliation", headers=headers
                        )
                        assert page.status_code == 200
                        if (
                            page.json()["executions"]
                            and page.json()["executions"][0]["state"] == "UNKNOWN"
                        ):
                            break
                        await asyncio.sleep(0.01)
                saved = page.json()["executions"][0]
                assert saved["response_id"] is None and saved["state"] == "UNKNOWN"
                evidence = {
                    "binding": saved["binding"],
                    "run_id": saved["run_id"],
                    "turn_id": saved["turn_id"],
                    "response_id": "resp_original",
                    "evidence_reference": "fixture-original-response",
                    "reason": "Verified original receipt in fixture",
                }
                attached = await client.post(
                    f"/internal/model-reconciliation/{saved['binding']['execution_id']}/original-receipt",
                    headers=headers,
                    json=evidence,
                )
                assert attached.status_code == 200
                async with asyncio.timeout(7):
                    while (
                        app.state.runtime.store.sync(USER, SESSION).runs[-1].status
                        != "COMPLETED"
                    ):
                        await asyncio.sleep(0.01)
                assert [request.method for request in requests] == ["POST", "GET"]

    asyncio.run(scenario())
