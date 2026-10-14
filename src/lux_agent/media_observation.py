"""Observe Java-authorized Assets and resolve frozen references at invocation only."""

import asyncio
import base64
import hashlib
import math
import shutil
import subprocess
from dataclasses import dataclass
from tempfile import TemporaryFile
from typing import TYPE_CHECKING

import httpx
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field

from lux_agent.media_types import (
    MAX_DECODED_PIXELS,
    MAX_IMAGE_BYTES,
    MAX_OBSERVATIONS,
    MAX_PRESENTATION_BYTES,
    MAX_VIDEO_SECONDS,
    JavaAssetMetadata,
    MediaObservation,
    ObservationSource,
)
from lux_agent.protocol import (
    AcceptedInputItem,
    AssetReferenceBlock,
    GetTaskArguments,
    TaskQuerySnapshot,
    decode_tool_result,
)

if TYPE_CHECKING:
    from lux_agent.execution import ModelRequest


class ProbeStream(BaseModel):
    model_config = ConfigDict(extra="ignore")
    width: int
    height: int
    duration: str | None = None


class ProbeFormat(BaseModel):
    model_config = ConfigDict(extra="ignore")
    duration: str | None = None


class ProbeResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    streams: tuple[ProbeStream, ...]
    format: ProbeFormat = Field(default_factory=ProbeFormat)


@dataclass(frozen=True)
class MediaInspection:
    width: int
    height: int
    duration_seconds: float | None


@dataclass(frozen=True)
class MediaSelection:
    asset_id: str
    source: ObservationSource


def _run(command: list[str], body: bytes) -> bytes:
    try:
        # An inherited, seekable descriptor supports ordinary MP4/MOV moov placement.
        # The decoder can open only fd/pipe protocols, never a pathname or network URL.
        with TemporaryFile() as source:
            source.write(body)
            source.seek(0)
            completed = subprocess.run(
                command,
                stdin=source,
                capture_output=True,
                timeout=20,
                check=False,
            )
    except subprocess.SubprocessError, OSError:
        raise ValueError("Media decoder failed or exceeded its time budget") from None
    if completed.returncode or completed.stderr.strip():
        raise ValueError("Media bytes cannot be decoded")
    return completed.stdout


def inspect_media(body: bytes, mime: str) -> MediaInspection:
    if mime == "image/png":
        signature_valid = body.startswith(b"\x89PNG\r\n\x1a\n")
    elif mime == "image/jpeg":
        signature_valid = body.startswith(b"\xff\xd8\xff")
    else:
        signature_valid = body[4:8] == b"ftyp" and (
            (body[8:12] == b"qt  ") == (mime == "video/quicktime")
        )
    if not signature_valid:
        raise ValueError("Asset bytes differ from declared media type")
    # Pipe-only protocols cannot open file/network references inside a container.
    output = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-max_alloc",
            "100000000",
            "-protocol_whitelist",
            "fd,pipe",
            "-fd",
            "0",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,duration:format=duration",
            "-of",
            "json",
            "fd:",
        ],
        body,
    )
    if len(output) > 65536:
        raise ValueError("Media inspection exceeds its read budget")
    probe = ProbeResult.model_validate_json(output)
    if len(probe.streams) != 1:
        raise ValueError("Expected a visual media stream")
    stream = probe.streams[0]
    if (
        stream.width < 1
        or stream.height < 1
        or stream.width * stream.height > MAX_DECODED_PIXELS
    ):
        raise ValueError("Source decoded pixel budget exceeded")
    duration = None
    if mime.startswith("video/"):
        duration = float(probe.format.duration or stream.duration or "nan")
        if not math.isfinite(duration) or not 0 < duration <= MAX_VIDEO_SECONDS:
            raise ValueError("Video duration exceeds observation policy")
    return MediaInspection(stream.width, stream.height, duration)


def presentation(body: bytes, timestamp: float | None) -> bytes:
    arguments = [
        "ffmpeg",
        "-v",
        "error",
        "-max_alloc",
        "100000000",
        "-threads",
        "1",
        "-protocol_whitelist",
        "fd,pipe",
        "-fd",
        "0",
        "-i",
        "fd:",
    ]
    if timestamp is not None:
        arguments.extend(["-ss", str(timestamp)])
    arguments.extend(
        [
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-frames:v",
            "1",
            "-threads",
            "1",
            "-vf",
            "scale=w='min(1024,iw)':h='min(1024,ih)':force_original_aspect_ratio=decrease",
            "-q:v",
            "5",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-fs",
            str(MAX_PRESENTATION_BYTES),
            "pipe:1",
        ]
    )
    output = _run(arguments, body)
    if not output.startswith(b"\xff\xd8\xff") or len(output) > MAX_PRESENTATION_BYTES:
        raise ValueError("Presentation image exceeds its byte budget or is unavailable")
    return output


class JavaMediaObserver:
    def __init__(self, client: httpx.AsyncClient) -> None:
        if any(shutil.which(tool) is None for tool in ("ffmpeg", "ffprobe")):
            raise ValueError("Media observation requires ffmpeg and ffprobe")
        self.client = client

    async def _read(self, path: str, limit: int, mime: str | None = None) -> bytes:
        try:
            async with self.client.stream(
                "GET", path, follow_redirects=False
            ) as response:
                if response.status_code != 200:
                    raise ValueError("Authorized Asset is unavailable")
                if (
                    mime
                    and response.headers.get("content-type", "").split(";")[0] != mime
                ):
                    raise ValueError(
                        "Asset content MIME differs from immutable metadata"
                    )
                result = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(result) + len(chunk) > limit:
                        raise ValueError("Asset download exceeds its read budget")
                    result.extend(chunk)
                return bytes(result)
        except httpx.HTTPError:
            raise ValueError("Authorized Asset is unavailable") from None

    async def _asset(
        self, user: str, selection: MediaSelection
    ) -> tuple[JavaAssetMetadata, bytes, MediaInspection]:
        path = f"/internal/v1/users/{user}/assets/{selection.asset_id}"
        asset = JavaAssetMetadata.model_validate_json(await self._read(path, 65536))
        if asset.id != selection.asset_id or asset.user_id != user:
            raise ValueError("Asset belongs to another owner or identity")
        if (
            selection.source.task_id is not None
            and asset.task_id != selection.source.task_id
        ):
            raise ValueError("Asset belongs to another Task")
        if asset.media_type not in (
            "image/png",
            "image/jpeg",
            "video/mp4",
            "video/quicktime",
        ):
            raise ValueError(
                "Unsupported media observation; audio observation is unsupported"
            )
        if asset.media_type.startswith("image/") and asset.size_bytes > MAX_IMAGE_BYTES:
            raise ValueError("Image exceeds its source byte budget")
        body = await self._read(path + "/content", asset.size_bytes, asset.media_type)
        if (
            len(body) != asset.size_bytes
            or hashlib.sha256(body).hexdigest() != asset.sha256
        ):
            raise ValueError("Asset length or checksum differs from immutable metadata")
        inspected = await asyncio.to_thread(inspect_media, body, asset.media_type)
        return asset, body, inspected

    @staticmethod
    def _selections(request: ModelRequest) -> tuple[MediaSelection, ...]:
        selections: list[MediaSelection] = []
        latest_tasks: dict[str, tuple[TaskQuerySnapshot, str]] = {}
        task_calls: dict[str, GetTaskArguments] = {}
        # Only fresh inputs and the latest assistant call batch trigger observation.
        # Historical messages remain byte-for-byte intact in the provider history.
        last_assistant = max(
            (
                index
                for index, message in enumerate(request.messages)
                if isinstance(message, AIMessage)
            ),
            default=-1,
        )
        recent_inputs = [
            message.id
            for message in request.messages[last_assistant + 1 :]
            if "lux_input" in message.additional_kwargs
        ]
        input_ids = request.media_input_ids or tuple(recent_inputs[-1:])
        task_boundary = max(
            last_assistant,
            max(
                (
                    index
                    for index, message in enumerate(request.messages)
                    if "lux_input" in message.additional_kwargs
                ),
                default=-1,
            ),
        )
        for position, message in enumerate(request.messages):
            if isinstance(message, AIMessage):
                latest_tasks.clear()
                task_calls.update(
                    {
                        call["id"]: GetTaskArguments.model_validate(call["args"])
                        for call in message.tool_calls
                        if call["name"] == "get_task"
                    }
                )
            raw = message.additional_kwargs.get("lux_input")
            if raw is not None and message.id in input_ids:
                item = AcceptedInputItem.model_validate(raw, strict=False)
                if item.id != message.id:
                    raise ValueError("Observation input identity differs")
                selections.extend(
                    MediaSelection(block.asset_id, ObservationSource(item_id=item.id))
                    for block in item.content.blocks
                    if isinstance(block, AssetReferenceBlock)
                )
            if (
                position > task_boundary
                and isinstance(message, ToolMessage)
                and message.tool_call_id in task_calls
            ):
                result = decode_tool_result(message.content)
                if isinstance(result, TaskQuerySnapshot) and message.id is not None:
                    if result.task_id != task_calls[message.tool_call_id].task_id:
                        raise ValueError(
                            "Task snapshot differs from original call Task"
                        )
                    previous = latest_tasks.get(result.task_id)
                    if previous is None or result.version >= previous[0].version:
                        latest_tasks[result.task_id] = (result, message.id)
        for task, item in latest_tasks.values():
            selections.extend(
                MediaSelection(
                    asset.asset_id,
                    ObservationSource(
                        item_id=item,
                        task_id=task.task_id,
                        task_version=task.version,
                    ),
                )
                for asset in task.assets
            )
        if len(selections) > MAX_OBSERVATIONS:
            raise ValueError("Invocation observation count exceeds policy")
        return tuple(selections)

    async def prepare(self, request: ModelRequest) -> ModelRequest:
        selections = self._selections(request)
        new_assets = {selection.asset_id for selection in selections}
        carried = [
            observation
            for observation in request.media_observations
            if observation.asset.id not in new_assets
        ]
        # Keep the most recent compact context, reserving one slot per fresh Asset.
        retained_count = MAX_OBSERVATIONS - len(selections)
        observations = carried[-retained_count:] if retained_count else []
        for position, selection in enumerate(selections):
            asset, body, inspected = await self._asset(request.user_id, selection)
            frame_count = min(
                8,
                MAX_OBSERVATIONS - len(observations) - (len(selections) - position - 1),
            )
            timestamps = (
                (None,)
                if inspected.duration_seconds is None
                else tuple(
                    inspected.duration_seconds * index / frame_count
                    for index in range(frame_count)
                )
            )
            if len(observations) + len(timestamps) > MAX_OBSERVATIONS:
                raise ValueError("Invocation observation count exceeds policy")
            for timestamp in timestamps:
                # A frozen descriptor represents a decodable presentation, not a header.
                await asyncio.to_thread(presentation, body, timestamp)
                observations.append(
                    MediaObservation(
                        kind="image" if timestamp is None else "video_frame",
                        user_id=request.user_id,
                        session_id=request.session_id,
                        source=selection.source,
                        asset=asset,
                        width=inspected.width,
                        height=inspected.height,
                        duration_seconds=inspected.duration_seconds,
                        timestamp_seconds=timestamp,
                    )
                )
        return request.model_copy(update={"media_observations": tuple(observations)})

    async def invocation_messages(
        self, request: ModelRequest
    ) -> tuple[AnyMessage, ...]:
        if len(request.media_observations) > MAX_OBSERVATIONS:
            raise ValueError("Invocation observation count exceeds policy")
        additions: list[AnyMessage] = []
        cached: dict[str, tuple[JavaAssetMetadata, bytes, MediaInspection]] = {}
        sources: set[MediaSelection] = set()
        for message in request.messages:
            raw = message.additional_kwargs.get("lux_input")
            if raw is not None:
                item = AcceptedInputItem.model_validate(raw, strict=False)
                sources.update(
                    MediaSelection(block.asset_id, ObservationSource(item_id=item.id))
                    for block in item.content.blocks
                    if isinstance(block, AssetReferenceBlock)
                )
            if isinstance(message, ToolMessage) and message.id:
                try:
                    snapshot = decode_tool_result(message.content)
                except ValueError:
                    continue
                if isinstance(snapshot, TaskQuerySnapshot):
                    sources.update(
                        MediaSelection(
                            asset.asset_id,
                            ObservationSource(
                                item_id=message.id,
                                task_id=snapshot.task_id,
                                task_version=snapshot.version,
                            ),
                        )
                        for asset in snapshot.assets
                    )
        for observation in request.media_observations:
            selection = MediaSelection(observation.asset.id, observation.source)
            if (
                observation.user_id != request.user_id
                or observation.session_id != request.session_id
                or selection not in sources
            ):
                raise ValueError("Observation request/source ownership differs")
            if observation.asset.id not in cached:
                cached[observation.asset.id] = await self._asset(
                    request.user_id, selection
                )
            asset, body, inspected = cached[observation.asset.id]
            if (
                asset != observation.asset
                or inspected.width != observation.width
                or inspected.height != observation.height
                or inspected.duration_seconds != observation.duration_seconds
            ):
                raise ValueError("Frozen observation metadata differs")
            jpeg = await asyncio.to_thread(
                presentation, body, observation.timestamp_seconds
            )
            label = (
                f"Observation of Asset {asset.id}; "
                f"source Item {observation.source.item_id}"
            )
            if observation.source.task_id:
                label += (
                    f"; Task {observation.source.task_id} "
                    f"snapshot {observation.source.task_version}"
                )
            if observation.timestamp_seconds is not None:
                label += (
                    f"; frame timestamp {observation.timestamp_seconds:.3f} seconds"
                )
            # LangChain owns this content-block mapping; bytes remain invocation-local.
            additions.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": label},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,"
                                + base64.b64encode(jpeg).decode("ascii"),
                                "detail": "auto",
                            },
                        },
                    ]
                )
            )
        return (*request.messages, *additions)
