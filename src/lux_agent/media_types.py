"""Compact immutable references; presentation bytes never enter durable state."""

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from lux_agent.protocol import ResourceId, Timestamp, WireModel

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_PRESENTATION_BYTES = 1024 * 1024
MAX_OBSERVATIONS = 8
MAX_DECODED_PIXELS = 40_000_000
MAX_VIDEO_SECONDS = 120


class ObservationSource(WireModel):
    item_id: ResourceId
    task_id: ResourceId | None = None
    task_version: StrictInt | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def task_binding(self) -> Self:
        if (self.task_id is None) != (self.task_version is None):
            raise ValueError("Task observation requires its original snapshot version")
        return self


class JavaAssetMetadata(WireModel):
    id: ResourceId
    user_id: ResourceId
    task_id: ResourceId | None
    filename: str
    media_type: str
    size_bytes: StrictInt = Field(ge=1, le=MAX_VIDEO_BYTES)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: Timestamp


class MediaObservation(WireModel):
    kind: Literal["image", "video_frame"]
    user_id: ResourceId
    session_id: ResourceId
    source: ObservationSource
    asset: JavaAssetMetadata
    width: StrictInt = Field(ge=1)
    height: StrictInt = Field(ge=1)
    duration_seconds: float | None = Field(default=None, gt=0, le=MAX_VIDEO_SECONDS)
    timestamp_seconds: float | None = Field(default=None, ge=0, lt=MAX_VIDEO_SECONDS)

    @model_validator(mode="after")
    def observation_binding(self) -> Self:
        if self.user_id != self.asset.user_id:
            raise ValueError("Observation owner differs from Asset owner")
        if (
            self.source.task_id is not None
            and self.source.task_id != self.asset.task_id
        ):
            raise ValueError("Observation Task differs from Asset Task")
        if self.width * self.height > MAX_DECODED_PIXELS:
            raise ValueError("Source decoded pixel budget exceeded")
        if self.kind == "image":
            if self.timestamp_seconds is not None or self.duration_seconds is not None:
                raise ValueError("Image observations cannot carry video timing")
        elif (
            self.timestamp_seconds is None
            or self.duration_seconds is None
            or self.timestamp_seconds >= self.duration_seconds
        ):
            raise ValueError("Video frame must belong to its bounded source duration")
        return self
