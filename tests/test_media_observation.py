"""Offline Java authorization and real ChatOpenAI vision transport regressions."""

import asyncio
import base64
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from test_openai_model import prepare, request, stream_fixture
from test_protocol_v2 import SESSION, USER

from lux_agent.store import AgentStore, content_hash

ASSET = "00000000-0000-0000-0000-000000000011"
TASK = "00000000-0000-0000-0000-000000000009"


def image_bytes():
    return subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:s=1200x600",
            "-frames:v",
            "1",
            "-threads",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout


def media_request(body=None):
    from test_protocol_v2 import input_json

    raw = json.loads(input_json())["item"]
    raw["content"]["blocks"].append({"type": "asset", "asset_id": ASSET})
    return request().model_copy(
        update={
            "messages": (
                HumanMessage(
                    content="Inspect uploaded media",
                    id=raw["id"],
                    additional_kwargs={"lux_input": raw},
                ),
            )
        }
    )


def metadata(body, mime="image/png", owner=USER, task=None):
    return {
        "id": ASSET,
        "user_id": owner,
        "task_id": task,
        "filename": "asset",
        "media_type": mime,
        "size_bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "created_at": datetime(2026, 10, 13, tzinfo=UTC).isoformat(),
    }


def java_handler(body, *, meta=None, status=200):
    def handle(req):
        assert req.url.path.startswith(f"/internal/v1/users/{USER}/assets/{ASSET}")
        if req.url.path.endswith("/content"):
            return httpx.Response(
                status,
                content=body,
                headers={"content-type": (meta or metadata(body))["media_type"]},
            )
        return httpx.Response(status, json=meta or metadata(body))

    return handle


def test_empty_observations_preserve_historical_serialization_and_hash():
    from lux_agent.execution import ModelRequest

    req = request()
    old = req.model_dump_json()
    added = ModelRequest.model_validate_json(old[:-1] + ',"media_observations":[]}')
    assert added.model_dump_json() == old
    assert content_hash(added.model_dump_json()) == content_hash(old)


def test_real_chatopenai_transports_verified_compact_image_and_keeps_history(tmp_path):
    async def scenario():
        from lux_agent.media_observation import JavaMediaObserver
        from lux_agent.openai_model import OpenAIModelExecutor

        body = image_bytes()
        payloads = []

        def provider(req):
            payloads.append(json.loads(req.content))
            return httpx.Response(
                200,
                content=b"".join(stream_fixture()[0]),
                headers={"content-type": "text/event-stream"},
            )

        async with httpx.AsyncClient(
            base_url="http://java", transport=httpx.MockTransport(java_handler(body))
        ) as java:
            observer = JavaMediaObserver(java)
            req = media_request()
            original = req.messages[0].model_dump_json()
            frozen = await observer.prepare(req)
            assert frozen.messages[0].model_dump_json() == original
            assert len(frozen.media_observations) == 1
            assert "base64" not in frozen.model_dump_json()
            assert frozen.media_observations[0].source.item_id == req.messages[0].id
            with AgentStore(tmp_path / "agent.db") as store:
                prepare(store, frozen)
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(provider)
                ) as client:
                    executor = OpenAIModelExecutor(
                        store,
                        api_key="offline-key",
                        account_ref="fixture",
                        http_client=client,
                        media_observer=observer,
                    )
                    async for _ in executor.stream(frozen):
                        pass
            blocks = [
                block
                for item in payloads[0]["input"]
                if isinstance(item.get("content"), list)
                for block in item["content"]
            ]
            images = [b for b in blocks if b["type"] == "input_image"]
            assert len(images) == 1
            presented = base64.b64decode(images[0]["image_url"].split(",")[1])
            assert presented.startswith(b"\xff\xd8\xff")
            assert len(presented) <= 1024 * 1024
            assert req.messages[0].model_dump_json() == original

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure",
    ["owner", "checksum", "mime", "audio", "unavailable", "redirect", "declared_limit"],
)
def test_media_boundary_rejects_invalid_or_unavailable_assets(failure):
    async def scenario():
        from lux_agent.media_observation import JavaMediaObserver

        body = image_bytes()
        meta = metadata(body)
        status = 200
        if failure == "owner":
            meta["user_id"] = str(uuid4())
        if failure == "checksum":
            meta["sha256"] = "0" * 64
        if failure == "mime":
            meta["media_type"] = "image/jpeg"
        if failure == "audio":
            meta["media_type"] = "audio/mpeg"
        if failure == "unavailable":
            status = 404
        if failure == "redirect":
            status = 302
        if failure == "declared_limit":
            meta["size_bytes"] = 10 * 1024 * 1024 + 1
        async with httpx.AsyncClient(
            base_url="http://java",
            transport=httpx.MockTransport(java_handler(body, meta=meta, status=status)),
        ) as client:
            with pytest.raises(ValueError):
                await JavaMediaObserver(client).prepare(media_request())

    asyncio.run(scenario())


def test_fresh_task_snapshot_observation_keeps_original_tool_result():
    async def scenario():
        from lux_agent.media_observation import JavaMediaObserver
        from lux_agent.protocol import TaskQuerySnapshot

        body = image_bytes()
        tool = ToolMessage(
            content=TaskQuerySnapshot(
                task_id=TASK,
                version=2,
                status="SUCCEEDED",
                failure=None,
                assets=({"asset_id": ASSET, "source_index": 0},),
            ).model_dump_json(),
            tool_call_id="old-call",
            id=str(uuid4()),
            name="get_task",
        )
        req = request().model_copy(update={"messages": (*request().messages[:3], tool)})
        async with httpx.AsyncClient(
            base_url="http://java",
            transport=httpx.MockTransport(
                java_handler(body, meta=metadata(body, task=TASK))
            ),
        ) as client:
            frozen = await JavaMediaObserver(client).prepare(req)
            observation = frozen.media_observations[0]
            assert observation.source.task_id == TASK
            assert observation.source.task_version == 2
            assert observation.source.item_id == tool.id
            assert frozen.messages[-1] == tool
            visible = await JavaMediaObserver(client).invocation_messages(frozen)
            assert visible[: len(req.messages)] == req.messages

    asyncio.run(scenario())


def video_bytes(tmp_path, mov=False):
    path = tmp_path / ("source.mov" if mov else "source.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=160x90:r=8:d=2",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            str(path),
        ],
        check=True,
    )
    return path.read_bytes()


@pytest.mark.parametrize("mov", [False, True])
def test_real_chatopenai_timestamped_video_frames_with_uploaded_image(tmp_path, mov):
    async def scenario():
        from lux_agent.media_observation import JavaMediaObserver
        from lux_agent.openai_model import OpenAIModelExecutor

        video = video_bytes(tmp_path, mov)
        image = image_bytes()
        other = str(uuid4())
        req = media_request()
        raw = req.messages[0].additional_kwargs["lux_input"]
        raw["content"]["blocks"].append({"type": "asset", "asset_id": other})
        req = req.model_copy(
            update={
                "messages": (
                    HumanMessage(
                        content="Observe image and video",
                        id=raw["id"],
                        additional_kwargs={"lux_input": raw},
                    ),
                )
            }
        )

        def java_handler(req):
            body = video if other in req.url.path else image
            mime = (
                ("video/quicktime" if mov else "video/mp4")
                if other in req.url.path
                else "image/png"
            )
            if req.url.path.endswith("/content"):
                return httpx.Response(200, content=body, headers={"content-type": mime})
            meta = metadata(body, mime=mime)
            if other in req.url.path:
                meta["id"] = other
            return httpx.Response(200, json=meta)

        payloads = []

        def provider(req):
            payloads.append(json.loads(req.content))
            return httpx.Response(
                200,
                content=b"".join(stream_fixture()[0]),
                headers={"content-type": "text/event-stream"},
            )

        async with httpx.AsyncClient(
            base_url="http://java", transport=httpx.MockTransport(java_handler)
        ) as client:
            observer = JavaMediaObserver(client)
            frozen = await observer.prepare(req)
            assert len(frozen.media_observations) == 8
            assert frozen.media_observations[1].timestamp_seconds == 0
            assert all(o.timestamp_seconds < 2 for o in frozen.media_observations[1:])
            assert "base64" not in frozen.model_dump_json()
            with AgentStore(tmp_path / "agent.db") as store:
                prepare(store, frozen)
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(provider)
                ) as provider_client:
                    executor = OpenAIModelExecutor(
                        store,
                        api_key="offline-key",
                        account_ref="fixture",
                        http_client=provider_client,
                        media_observer=observer,
                    )
                    async for _ in executor.stream(frozen):
                        pass
            blocks = [
                block
                for item in payloads[0]["input"]
                if isinstance(item.get("content"), list)
                for block in item["content"]
            ]
            assert sum(b["type"] == "input_image" for b in blocks) == 8
            labels = [
                b["text"]
                for b in blocks
                if b["type"] == "input_text" and "frame timestamp" in b["text"]
            ]
            assert len(labels) == 7

    asyncio.run(scenario())


def test_unnamed_runtime_tool_result_uses_original_call_binding():
    async def scenario():
        from lux_agent.media_observation import JavaMediaObserver
        from lux_agent.protocol import TaskAssetReference, TaskQuerySnapshot

        body = image_bytes()
        req = request()
        tool = ToolMessage(
            content=TaskQuerySnapshot(
                task_id=TASK,
                version=2,
                status="SUCCEEDED",
                failure=None,
                assets=(TaskAssetReference(asset_id=ASSET, source_index=0),),
            ).model_dump_json(),
            tool_call_id="old-call",
            id=str(uuid4()),
        )
        req = req.model_copy(update={"messages": (*req.messages[:3], tool)})
        async with httpx.AsyncClient(
            base_url="http://java",
            transport=httpx.MockTransport(
                java_handler(body, meta=metadata(body, task=TASK))
            ),
        ) as client:
            frozen = await JavaMediaObserver(client).prepare(req)
            assert len(frozen.media_observations) == 1
            assert frozen.media_observations[0].source.item_id == tool.id

    asyncio.run(scenario())


def test_seekable_video_without_faststart_is_observable(tmp_path):
    from lux_agent.media_observation import inspect_media, presentation

    path = tmp_path / "nonfaststart.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=1600x900:r=30:d=4",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-threads",
            "1",
            str(path),
        ],
        check=True,
    )
    body = path.read_bytes()
    assert len(body) > 1024 * 1024
    inspected = inspect_media(body, "video/mp4")
    assert inspected.duration_seconds == 4
    assert presentation(body, 2.0).startswith(b"\xff\xd8\xff")


def test_bad_media_fails_once_and_later_input_survives_native_restart(tmp_path):
    async def scenario():
        from langchain_core.messages import AIMessage
        from test_protocol_v2 import input_json

        from lux_agent.execution import ModelResponseObserved, ObservedModelItem
        from lux_agent.media_observation import JavaMediaObserver
        from lux_agent.protocol import OrderedContent, TextBlock
        from lux_agent.runtime import AgentRuntime

        raw = json.loads(input_json())
        raw["item"]["content"]["blocks"].append({"type": "asset", "asset_id": ASSET})
        seen = []

        class Executor:
            live_slots = ()

            async def stream(self, req):
                seen.append(req)
                yield ModelResponseObserved(
                    items=(
                        ObservedModelItem(
                            kind="MESSAGE",
                            payload=OrderedContent(blocks=(TextBlock(text="done"),)),
                        ),
                    ),
                    provider_response={},
                    messages=(AIMessage(content="done"),),
                )

        path = tmp_path / "agent.db"
        async with httpx.AsyncClient(
            base_url="http://java",
            transport=httpx.MockTransport(lambda req: httpx.Response(404)),
        ) as client:
            with AgentStore(path) as store:
                store.admit(json.dumps(raw))
                async with AgentRuntime(
                    store, executor=Executor(), media_observer=JavaMediaObserver(client)
                ) as runtime:
                    await runtime.advance(SESSION)
                    assert store.sync(USER, SESSION).runs[-1].status == "FAILED"
                    assert (
                        store.sync(USER, SESSION).runs[-1].error_code
                        == "MEDIA_OBSERVATION_FAILED"
                    )
                    await runtime.advance(SESSION)
                    assert seen == []
            with AgentStore(path) as store:
                store.admit(input_json(sequence=2, text="Later text input"))
                async with AgentRuntime(
                    store, executor=Executor(), media_observer=JavaMediaObserver(client)
                ) as runtime:
                    await runtime.advance(SESSION)
                    assert [run.status for run in store.sync(USER, SESSION).runs] == [
                        "FAILED",
                        "COMPLETED",
                    ]
                    assert len(seen) == 1
                    assert len(seen[0].messages) == 2

    asyncio.run(scenario())


def test_successful_image_context_survives_tool_turn_and_native_checkpoint(tmp_path):
    async def scenario():
        from langchain_core.messages import AIMessage
        from test_protocol_v2 import input_json

        from lux_agent.execution import ModelResponseObserved, ObservedModelItem
        from lux_agent.media_observation import JavaMediaObserver
        from lux_agent.protocol import (
            GetTaskCallContent,
            OrderedContent,
            TaskQuerySnapshot,
            TextBlock,
        )
        from lux_agent.runtime import AgentRuntime

        body = image_bytes()
        raw = json.loads(input_json())
        raw["item"]["content"]["blocks"].append({"type": "asset", "asset_id": ASSET})
        observed = []

        class Executor:
            live_slots = ()

            async def stream(self, req):
                observed.append(req)
                if len(observed) == 1:
                    yield ModelResponseObserved(
                        items=(
                            ObservedModelItem(
                                kind="TOOL_CALL",
                                call_id="task-call",
                                tool_name="get_task",
                                payload=GetTaskCallContent(arguments={"task_id": TASK}),
                            ),
                        ),
                        provider_response={},
                        messages=(
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "get_task",
                                        "id": "task-call",
                                        "args": {"task_id": TASK},
                                    }
                                ],
                            ),
                        ),
                    )
                else:
                    assert len(req.media_observations) == 1
                    yield ModelResponseObserved(
                        items=(
                            ObservedModelItem(
                                kind="MESSAGE",
                                payload=OrderedContent(
                                    blocks=(TextBlock(text="done"),)
                                ),
                            ),
                        ),
                        provider_response={},
                        messages=(AIMessage(content="done"),),
                    )

        class Tools:
            async def execute(self, req):
                return TaskQuerySnapshot(
                    task_id=TASK, version=1, status="RUNNING", assets=(), failure=None
                )

        path = tmp_path / "agent.db"
        async with httpx.AsyncClient(
            base_url="http://java", transport=httpx.MockTransport(java_handler(body))
        ) as client:
            with AgentStore(path) as store:
                store.admit(json.dumps(raw))
                async with AgentRuntime(
                    store,
                    executor=Executor(),
                    tools=Tools(),
                    media_observer=JavaMediaObserver(client),
                ) as runtime:
                    await runtime.advance(SESSION)
                    await runtime.advance(SESSION)
                    assert store.sync(USER, SESSION).runs[-1].status == "COMPLETED", (
                        len(observed),
                        [len(r.media_observations) for r in observed],
                        (await runtime.state(SESSION)).active_turn,
                    )
                    assert len(observed) == 2
                    assert (
                        observed[0].media_observations == observed[1].media_observations
                    )
                    assert (
                        "base64" not in (await runtime.state(SESSION)).model_dump_json()
                    )
            with AgentStore(path) as store:
                async with AgentRuntime(
                    store, media_observer=JavaMediaObserver(client)
                ) as runtime:
                    state = await runtime.state(SESSION)
                    assert len(state.media_observations) == 1
                    assert state.media_observations == observed[0].media_observations
                    assert "base64" not in state.model_dump_json()

    asyncio.run(scenario())


def test_media_integrity_change_after_freeze_prevents_provider_post(tmp_path):
    async def scenario():
        from lux_agent.execution import ConfirmedExecutionFailure
        from lux_agent.media_observation import JavaMediaObserver
        from lux_agent.openai_model import OpenAIModelExecutor

        body = image_bytes()
        changed = False
        calls = []

        def java(req):
            if req.url.path.endswith("/content"):
                return httpx.Response(
                    200,
                    content=body + (b"x" if changed else b""),
                    headers={"content-type": "image/png"},
                )
            return httpx.Response(200, json=metadata(body))

        async with httpx.AsyncClient(
            base_url="http://java", transport=httpx.MockTransport(java)
        ) as client:
            observer = JavaMediaObserver(client)
            frozen = await observer.prepare(media_request())
            changed = True
            with AgentStore(tmp_path / "agent.db") as store:
                prepare(store, frozen)
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(lambda req: calls.append(req))
                ) as provider:
                    executor = OpenAIModelExecutor(
                        store,
                        api_key="offline-key",
                        account_ref="fixture",
                        http_client=provider,
                        media_observer=observer,
                    )
                    with pytest.raises(ConfirmedExecutionFailure):
                        async for _ in executor.stream(frozen):
                            pass
                assert calls == []
                assert (
                    store.model_provider_receipt(frozen.execution_id).dispatch_started
                    is False
                )

    asyncio.run(scenario())


def test_native_media_recovery_retrieves_original_without_refetch_or_post(tmp_path):
    async def scenario():
        from test_openai_model import DisconnectStream, event, response
        from test_protocol_v2 import input_json

        from lux_agent.media_observation import JavaMediaObserver
        from lux_agent.openai_model import OpenAIModelExecutor
        from lux_agent.runtime import AgentRuntime

        body = image_bytes()
        raw = json.loads(input_json())
        raw["item"]["content"]["blocks"].append({"type": "asset", "asset_id": ASSET})
        calls = []
        media_calls = []
        media_unavailable = False

        def java(req):
            media_calls.append(req.url.path)
            if media_unavailable:
                raise AssertionError("Recovery must not fetch media")
            return java_handler(body)(req)

        def provider(req):
            calls.append(req.method)
            if req.method == "POST":
                return httpx.Response(
                    200,
                    stream=DisconnectStream(
                        [event("response.created", response=response("in_progress"))]
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                200,
                json=response(
                    output=[
                        {
                            "type": "message",
                            "id": "msg_original",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "Recovered",
                                    "annotations": [],
                                }
                            ],
                        }
                    ]
                ),
            )

        path = tmp_path / "agent.db"
        async with (
            httpx.AsyncClient(
                base_url="http://java", transport=httpx.MockTransport(java)
            ) as java_client,
            httpx.AsyncClient(
                transport=httpx.MockTransport(provider)
            ) as provider_client,
        ):
            observer = JavaMediaObserver(java_client)
            with AgentStore(path) as store:
                store.admit(json.dumps(raw))
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-key",
                    account_ref="fixture",
                    http_client=provider_client,
                    media_observer=observer,
                )
                async with AgentRuntime(
                    store, executor=executor, media_observer=observer
                ) as runtime:
                    await runtime.advance(SESSION)
                    original = (
                        await runtime.state(SESSION)
                    ).active_turn.model_execution_id
                    saved = store.execution(original)
                    assert saved.state == "UNKNOWN"
                    assert "media_observations" in saved.preparation.request_snapshot
                    assert "base64" not in saved.preparation.request_snapshot
            read_count = len(media_calls)
            media_unavailable = True
            with AgentStore(path) as store:
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-key",
                    account_ref="fixture",
                    http_client=provider_client,
                    media_observer=observer,
                )
                async with AgentRuntime(
                    store,
                    executor=executor,
                    media_observer=observer,
                    model_reconciler=executor,
                ) as runtime:
                    await runtime.advance(SESSION)
                    assert store.sync(USER, SESSION).runs[-1].status == "COMPLETED"
                    assert len((await runtime.state(SESSION)).media_observations) == 1
                    assert len(media_calls) == read_count
                    assert calls == ["POST", "GET"]
                    assert store.execution(original).state == "CONFIRMED"

    asyncio.run(scenario())


def test_count_limit_rejects_new_assets_without_rewriting_history():
    async def scenario():
        from lux_agent.media_observation import JavaMediaObserver

        req = media_request()
        raw = req.messages[0].additional_kwargs["lux_input"]
        raw["content"]["blocks"] = [
            {"type": "asset", "asset_id": str(uuid4())} for _ in range(9)
        ]
        original = req.model_dump_json()
        async with httpx.AsyncClient(
            base_url="http://java",
            transport=httpx.MockTransport(
                lambda req: (_ for _ in ()).throw(
                    AssertionError("Count checked before download")
                )
            ),
        ) as client:
            with pytest.raises(ValueError, match="count"):
                await JavaMediaObserver(client).prepare(req)
        assert req.model_dump_json() == original

    asyncio.run(scenario())


def test_real_chatopenai_keeps_prior_call_and_tool_output_with_new_vision(tmp_path):
    async def scenario():
        from lux_agent.media_observation import JavaMediaObserver
        from lux_agent.openai_model import OpenAIModelExecutor

        body = image_bytes()
        uploaded = media_request()
        old = request().messages[:4]
        req = uploaded.model_copy(update={"messages": (*old, *uploaded.messages)})
        payloads = []

        def provider(req):
            payloads.append(json.loads(req.content))
            return httpx.Response(
                200,
                content=b"".join(stream_fixture()[0]),
                headers={"content-type": "text/event-stream"},
            )

        async with httpx.AsyncClient(
            base_url="http://java", transport=httpx.MockTransport(java_handler(body))
        ) as client:
            observer = JavaMediaObserver(client)
            frozen = await observer.prepare(req)
            with AgentStore(tmp_path / "agent.db") as store:
                prepare(store, frozen)
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(provider)
                ) as provider_client:
                    executor = OpenAIModelExecutor(
                        store,
                        api_key="offline-key",
                        account_ref="fixture",
                        http_client=provider_client,
                        media_observer=observer,
                    )
                    async for _ in executor.stream(frozen):
                        pass
        payload = payloads[0]["input"]
        assert (
            next(item for item in payload if item.get("type") == "function_call")[
                "call_id"
            ]
            == "old-call"
        )
        assert (
            next(
                item for item in payload if item.get("type") == "function_call_output"
            )["output"]
            == "original receipt"
        )
        assert frozen.messages[:4] == old

    asyncio.run(scenario())


def test_decoded_pixel_and_video_duration_limits_are_enforced(tmp_path):
    import struct
    import zlib

    from lux_agent.media_observation import inspect_media

    png = bytearray(image_bytes())
    png[16:24] = struct.pack(">II", 9000, 5000)
    png[29:33] = struct.pack(">I", zlib.crc32(png[12:29]))
    with pytest.raises(ValueError):
        inspect_media(bytes(png), "image/png")
    video = tmp_path / "too-long.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:s=16x16:r=1:d=121",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            str(video),
        ],
        check=True,
    )
    with pytest.raises(ValueError, match="duration"):
        inspect_media(video.read_bytes(), "video/mp4")


def test_fresh_task_snapshot_cannot_rebind_another_original_task():
    async def scenario():
        from lux_agent.media_observation import JavaMediaObserver
        from lux_agent.protocol import TaskAssetReference, TaskQuerySnapshot

        body = image_bytes()
        other_task = str(uuid4())
        req = request()
        tool = ToolMessage(
            content=TaskQuerySnapshot(
                task_id=other_task,
                version=2,
                status="SUCCEEDED",
                failure=None,
                assets=(TaskAssetReference(asset_id=ASSET, source_index=0),),
            ).model_dump_json(),
            tool_call_id="old-call",
            id=str(uuid4()),
        )
        req = req.model_copy(update={"messages": (*req.messages[:3], tool)})
        async with httpx.AsyncClient(
            base_url="http://java",
            transport=httpx.MockTransport(
                java_handler(body, meta=metadata(body, task=other_task))
            ),
        ) as client:
            with pytest.raises(ValueError, match="Task"):
                await JavaMediaObserver(client).prepare(req)

    asyncio.run(scenario())


def test_hydration_cancellation_recovers_undispatched_run_then_later_input(tmp_path):
    async def scenario():
        from test_openai_model import event, response
        from test_protocol_v2 import input_json

        from lux_agent.media_observation import JavaMediaObserver
        from lux_agent.openai_model import OpenAIModelExecutor
        from lux_agent.runtime import AgentRuntime

        body = image_bytes()
        raw = json.loads(input_json())
        raw["item"]["content"]["blocks"].append({"type": "asset", "asset_id": ASSET})
        hydration = asyncio.Event()
        release = asyncio.Event()
        reads = 0
        requests = []

        async def java(req):
            nonlocal reads
            reads += 1
            if reads == 3:
                hydration.set()
                await release.wait()
            return java_handler(body)(req)

        def provider(req):
            requests.append(req.method)
            return httpx.Response(
                200,
                content=b"".join(
                    [
                        event("response.created", response=response("in_progress")),
                        event(
                            "response.output_text.delta",
                            output_index=0,
                            item_id="msg_original",
                            content_index=0,
                            delta="Fresh input completed",
                        ),
                        event("response.completed", response=response()),
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )

        path = tmp_path / "agent.db"
        async with (
            httpx.AsyncClient(
                base_url="http://java", transport=httpx.MockTransport(java)
            ) as java_client,
            httpx.AsyncClient(
                transport=httpx.MockTransport(provider)
            ) as provider_client,
        ):
            observer = JavaMediaObserver(java_client)
            with AgentStore(path) as store:
                store.admit(json.dumps(raw))
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-key",
                    account_ref="fixture",
                    http_client=provider_client,
                    media_observer=observer,
                )
                async with AgentRuntime(
                    store,
                    executor=executor,
                    media_observer=observer,
                    model_reconciler=executor,
                ) as runtime:
                    active = asyncio.create_task(runtime.advance(SESSION))
                    await asyncio.wait_for(hydration.wait(), 5)
                    active.cancel()
                    await asyncio.gather(active, return_exceptions=True)
                    original = (
                        await runtime.state(SESSION)
                    ).active_turn.model_execution_id
                    assert store.execution(original).state == "UNKNOWN"
                    marker = store.model_provider_receipt(original)
                    assert marker is not None
                    assert marker.dispatch_started is False
                    assert marker.response_id is None
                    assert requests == []
                    frozen_snapshot = store.execution(
                        original
                    ).preparation.request_snapshot
            release.set()
            with AgentStore(path) as store:
                store.admit(input_json(sequence=2, text="Later fresh text"))
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-key",
                    account_ref="fixture",
                    http_client=provider_client,
                    media_observer=observer,
                )
                async with AgentRuntime(
                    store,
                    executor=executor,
                    media_observer=observer,
                    model_reconciler=executor,
                ) as runtime:
                    for _ in range(3):
                        await runtime.advance(SESSION)
                    assert store.execution(original).state == "FAILED"
                    assert (
                        json.loads(store.execution(original).response_snapshot)["code"]
                        == "MODEL_NOT_SUBMITTED"
                    )
                    assert (
                        store.execution(original).preparation.request_snapshot
                        == frozen_snapshot
                    )
                    assert [run.status for run in store.sync(USER, SESSION).runs] == [
                        "FAILED",
                        "COMPLETED",
                    ]
                    assert requests == ["POST"]
                    assert reads == 3

    asyncio.run(scenario())


def test_reopened_pre_dispatch_receipt_recovers_without_hydration_or_network(tmp_path):
    async def scenario():
        from lux_agent.execution import ConfirmedExecutionFailure
        from lux_agent.openai_model import OpenAIModelExecutor

        req = request()
        path = tmp_path / "agent.db"

        def never_network(req):
            raise AssertionError(
                "Undispatched recovery cannot call any provider endpoint"
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(never_network)
        ) as client:
            with AgentStore(path) as store:
                prepare(store, req)
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-key",
                    account_ref="fixture",
                    http_client=client,
                )
                store.prepare_model_provider(
                    executor._binding(req), dispatch_started=False
                )
            with AgentStore(path) as store:
                executor = OpenAIModelExecutor(
                    store,
                    api_key="offline-key",
                    account_ref="fixture",
                    http_client=client,
                )
                with pytest.raises(ConfirmedExecutionFailure) as error:
                    await executor.reconcile(req)
                assert error.value.failure.code == "MODEL_NOT_SUBMITTED"
                assert (
                    store.model_provider_receipt(req.execution_id).dispatch_started
                    is False
                )

    asyncio.run(scenario())
