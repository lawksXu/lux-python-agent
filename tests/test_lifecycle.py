import asyncio

import httpx
from test_protocol_v2 import SESSION, USER, input_json
from test_runtime_v2 import ReplyExecutor


def test_enabled_lifespan_runs_native_driver_and_source_sync(tmp_path):
    async def scenario():
        from lux_agent.main import ServiceSettings, create_app
        from lux_agent.store import AgentStore

        path = tmp_path / "agent.db"
        with AgentStore(path) as store:
            store.admit(input_json())
        app = create_app(ServiceSettings(store_path=path), executor=ReplyExecutor())
        async with app.router.lifespan_context(app):
            async with asyncio.timeout(3):
                while True:
                    source = app.state.runtime.store.sync(USER, SESSION)
                    if source.runs and source.runs[-1].status == "COMPLETED":
                        break
                    await asyncio.sleep(0.01)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get(
                    f"/internal/v2/users/{USER}/sessions/{SESSION}/execution",
                    params={"limit": 1},
                )
                assert response.status_code == 200
                body = response.json()
                assert body["runs"][-1]["status"] == "COMPLETED"
                assert len(body["events"]) == 1
                assert len(body["items"]) == 1
                assert "messages" not in body
                assert (await client.get("/health")).json() == {"status": "ok"}

    asyncio.run(scenario())


def test_disabled_lifespan_keeps_health_lightweight():
    async def scenario():
        from lux_agent.main import ServiceSettings, create_app

        app = create_app(ServiceSettings())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                assert (await client.get("/health")).status_code == 200
                assert (
                    await client.get(
                        f"/internal/v2/users/{USER}/sessions/{SESSION}/execution"
                    )
                ).status_code == 503

    asyncio.run(scenario())


def test_openai_lifespan_wires_executor_and_reconciler_together(tmp_path, monkeypatch):
    async def scenario():
        import json

        from test_openai_model import event, response

        import lux_agent.main as main
        from lux_agent.openai_model import OpenAIModelExecutor
        from lux_agent.store import AgentStore

        requests = []

        def handler(req):
            requests.append(req)
            message = {
                "type": "message",
                "id": "msg_original",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "Ready", "annotations": []}
                ],
            }
            stream = b"".join(
                [
                    event("response.created", response=response("in_progress")),
                    event(
                        "response.output_text.delta",
                        output_index=0,
                        item_id="msg_original",
                        content_index=0,
                        delta="Ready",
                    ),
                    event("response.completed", response=response(output=[message])),
                ]
            )
            return httpx.Response(
                200, content=stream, headers={"content-type": "text/event-stream"}
            )

        original_client = httpx.AsyncClient

        class OfflineClient(original_client):
            def __init__(self, **kwargs):
                super().__init__(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(main.httpx, "AsyncClient", OfflineClient)
        monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
        path = tmp_path / "agent.db"
        with AgentStore(path) as store:
            store.admit(input_json())
        app = main.create_app(
            main.ServiceSettings(store_path=path, openai_enabled=True)
        )
        async with app.router.lifespan_context(app):
            runtime = app.state.runtime
            assert isinstance(runtime.executor, OpenAIModelExecutor)
            assert runtime.model_reconciler is runtime.executor
            async with asyncio.timeout(3):
                while (
                    not runtime.store.sync(USER, SESSION).runs
                    or runtime.store.sync(USER, SESSION).runs[-1].status != "COMPLETED"
                ):
                    await asyncio.sleep(0.01)
            assert len(requests) == 1
            assert json.loads(requests[0].content)["model"] == "gpt-6-luna"
        assert app.state.runtime is None
        with AgentStore(path) as store:
            assert store.sync(USER, SESSION).runs[-1].status == "COMPLETED"

    asyncio.run(scenario())


def test_openai_default_disabled_and_missing_key_make_no_provider_request(
    tmp_path, monkeypatch
):
    async def scenario():
        import lux_agent.main as main
        from lux_agent.main import ServiceSettings, create_app

        def unexpected(*args, **kwargs):
            raise AssertionError("Provider adapter must not be constructed")

        monkeypatch.setattr(main, "OpenAIModelExecutor", unexpected)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        for index, enabled in enumerate((False, True)):
            app = create_app(
                ServiceSettings(
                    store_path=tmp_path / f"agent{index}.db", openai_enabled=enabled
                )
            )
            async with app.router.lifespan_context(app):
                assert app.state.runtime.executor is None
                assert app.state.runtime.model_reconciler is None

    asyncio.run(scenario())
