"""Explicit generation-plan v1 wire contracts shared with the Java owner."""

from datetime import datetime
from typing import Annotated, Literal, get_args, get_origin

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)


class PlanWireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator(
        "schema_version", "accepted", "file_count", mode="before", check_fields=False
    )
    @classmethod
    def literal_types(cls, value, info: ValidationInfo):
        annotation = cls.model_fields[info.field_name].annotation
        if get_origin(annotation) is Literal and not any(
            type(value) is type(expected) and value == expected
            for expected in get_args(annotation)
        ):
            raise ValueError("Literal value requires its exact JSON type")
        return value


def valid_instant(value: str) -> str:
    datetime.fromisoformat(value)
    return value


Uuid = Annotated[
    StrictStr,
    Field(pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"),
]

Instant = Annotated[
    StrictStr,
    Field(pattern="^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(\\.\\d{1,6})?Z$"),
    AfterValidator(valid_instant),
]

Sha256 = Annotated[StrictStr, Field(pattern="^[0-9a-f]{64}$")]

MediaType = Literal["IMAGE", "VIDEO"]

PlanState = Literal["DRAFT", "FROZEN"]

ImageMode = Literal["IMAGE"]

VideoMode = Literal["VIDEO"]

ImageOutputFormat = Literal["PNG", "JPEG"]

VideoOutputFormat = Literal["MP4", "MOV"]

ParameterKey = Literal[
    "PROMPT",
    "ASPECT_RATIO",
    "RESOLUTION_ID",
    "PIXEL_DIMENSIONS",
    "REFERENCES",
    "OUTPUT_FORMAT",
    "DURATION_SECONDS",
    "GENERATE_AUDIO",
]


class AspectRatio(PlanWireModel):
    width_units: Annotated[StrictInt, Field(ge=1, le=2147483647)]
    height_units: Annotated[StrictInt, Field(ge=1, le=2147483647)]


class PixelDimensions(PlanWireModel):
    width: Annotated[StrictInt, Field(ge=1, le=2147483647)]
    height: Annotated[StrictInt, Field(ge=1, le=2147483647)]


class RatioSize(PlanWireModel):
    type: Literal["ratio_size"]
    aspect_ratio: AspectRatio | None
    resolution_id: Annotated[StrictStr, Field(min_length=1, max_length=64)] | None


class PixelSize(PlanWireModel):
    type: Literal["pixel_size"]
    dimensions: PixelDimensions


DraftOutputSize = Annotated[RatioSize | PixelSize, Field(discriminator="type")]


class ReferenceAsset(PlanWireModel):
    type: Literal["reference_asset"]
    asset_id: Uuid


class ImageParameters(PlanWireModel):
    type: Literal["image_parameters"]
    prompt: Annotated[StrictStr, Field(min_length=1, max_length=64000)] | None
    size: DraftOutputSize | None
    references: Annotated[
        tuple[ReferenceAsset, ...], Field(min_length=0, max_length=64)
    ]
    output_format: ImageOutputFormat | None


class VideoParameters(PlanWireModel):
    type: Literal["video_parameters"]
    prompt: Annotated[StrictStr, Field(min_length=1, max_length=64000)] | None
    size: DraftOutputSize | None
    references: Annotated[
        tuple[ReferenceAsset, ...], Field(min_length=0, max_length=64)
    ]
    output_format: VideoOutputFormat | None
    duration_seconds: Annotated[StrictInt, Field(ge=1, le=2147483647)] | None
    generate_audio: StrictBool | None


class ImageRequirements(PlanWireModel):
    type: Literal["image_requirements"]
    prompt: Annotated[StrictStr, Field(min_length=1, max_length=64000)] | None
    size: DraftOutputSize | None
    requested_file_count: Annotated[StrictInt, Field(ge=1, le=2147483647)] | None
    references: (
        Annotated[tuple[ReferenceAsset, ...], Field(min_length=0, max_length=64)] | None
    )
    output_format: ImageOutputFormat | None
    generation_mode: ImageMode | None


class VideoRequirements(PlanWireModel):
    type: Literal["video_requirements"]
    prompt: Annotated[StrictStr, Field(min_length=1, max_length=64000)] | None
    size: DraftOutputSize | None
    requested_file_count: Annotated[StrictInt, Field(ge=1, le=2147483647)] | None
    references: (
        Annotated[tuple[ReferenceAsset, ...], Field(min_length=0, max_length=64)] | None
    )
    output_format: VideoOutputFormat | None
    duration_seconds: Annotated[StrictInt, Field(ge=1, le=2147483647)] | None
    generate_audio: StrictBool | None
    generation_mode: VideoMode | None


class ImageIntent(PlanWireModel):
    type: Literal["image_intent"]
    description: Annotated[StrictStr, Field(min_length=1, max_length=64000)]
    requirements: ImageRequirements


class VideoIntent(PlanWireModel):
    type: Literal["video_intent"]
    description: Annotated[StrictStr, Field(min_length=1, max_length=64000)]
    requirements: VideoRequirements


class ModelChoice(PlanWireModel):
    model_id: Annotated[StrictStr, Field(min_length=1, max_length=255)]
    profile_id: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    expected_capability_revision: Annotated[
        StrictInt, Field(ge=1, le=9223372036854775807)
    ]


class ModelSelection(PlanWireModel):
    model_id: Annotated[StrictStr, Field(min_length=1, max_length=255)]
    profile_id: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    capability_revision: Annotated[StrictInt, Field(ge=1, le=9223372036854775807)]


class ParameterOrigin(PlanWireModel):
    field: ParameterKey
    source: Literal["EXPLICIT", "MODEL_DEFAULT"]
    source_model: ModelSelection | None

    @model_validator(mode="after")
    def origin_source(self):
        if (self.source == "MODEL_DEFAULT") != (self.source_model is not None):
            raise ValueError("Parameter origin source and source_model disagree")
        return self


class ImagePlanConfiguration(PlanWireModel):
    type: Literal["image_plan_configuration"]
    model_selection: ModelSelection | None
    intent: ImageIntent
    parameters: ImageParameters
    parameter_origins: Annotated[
        tuple[ParameterOrigin, ...], Field(min_length=0, max_length=11)
    ]


class VideoPlanConfiguration(PlanWireModel):
    type: Literal["video_plan_configuration"]
    model_selection: ModelSelection | None
    intent: VideoIntent
    parameters: VideoParameters
    parameter_origins: Annotated[
        tuple[ParameterOrigin, ...], Field(min_length=0, max_length=11)
    ]


PlanConfiguration = Annotated[
    ImagePlanConfiguration | VideoPlanConfiguration, Field(discriminator="type")
]


class SizePreset(PlanWireModel):
    preset_id: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    aspect_ratio: AspectRatio
    resolution_id: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    pixel_dimensions: PixelDimensions


class PixelBounds(PlanWireModel):
    min_width: Annotated[StrictInt, Field(ge=1, le=2147483647)]
    max_width: Annotated[StrictInt, Field(ge=1, le=2147483647)]
    min_height: Annotated[StrictInt, Field(ge=1, le=2147483647)]
    max_height: Annotated[StrictInt, Field(ge=1, le=2147483647)]
    max_pixels: Annotated[StrictInt, Field(ge=1, le=9223372036854775807)]


class CustomPixelRules(PlanWireModel):
    bounds: PixelBounds
    width_step: Annotated[StrictInt, Field(ge=1, le=2147483647)]
    height_step: Annotated[StrictInt, Field(ge=1, le=2147483647)]
    allowed_aspect_ratios: Annotated[tuple[AspectRatio, ...], Field(min_length=1)]


class SizeRules(PlanWireModel):
    presets: Annotated[tuple[SizePreset, ...], Field(min_length=0)]
    custom_pixels: CustomPixelRules | None


class DiscreteIntDomain(PlanWireModel):
    type: Literal["discrete_int_domain"]
    values: Annotated[
        tuple[Annotated[StrictInt, Field(ge=1, le=2147483647)], ...],
        Field(min_length=1),
    ]


class SteppedIntDomain(PlanWireModel):
    type: Literal["stepped_int_domain"]
    minimum: Annotated[StrictInt, Field(ge=1, le=2147483647)]
    maximum: Annotated[StrictInt, Field(ge=1, le=2147483647)]
    step: Annotated[StrictInt, Field(ge=1, le=2147483647)]


PositiveIntDomain = Annotated[
    DiscreteIntDomain | SteppedIntDomain, Field(discriminator="type")
]


class PromptRules(PlanWireModel):
    max_utf8_bytes: Annotated[StrictInt, Field(ge=1, le=262144)]


ReferenceMediaType = Literal["IMAGE", "VIDEO", "AUDIO"]


class AspectRatioRange(PlanWireModel):
    minimum: AspectRatio
    maximum: AspectRatio


class ReferenceMediaLimit(PlanWireModel):
    media_type: ReferenceMediaType
    minimum_count: Annotated[StrictInt, Field(ge=0, le=64)]
    maximum_count: Annotated[StrictInt, Field(ge=0, le=64)]
    mime_types: Annotated[
        tuple[Annotated[StrictStr, Field(min_length=1, max_length=128)], ...],
        Field(min_length=1),
    ]
    max_asset_bytes: Annotated[StrictInt, Field(ge=1, le=9223372036854775807)]
    dimensions: PixelBounds | None
    max_duration_milliseconds: (
        Annotated[StrictInt, Field(ge=1, le=9223372036854775807)] | None
    )
    # Omitted legacy fields must round-trip without changing persisted receipts.
    aspect_ratio_range: AspectRatioRange | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("aspect_ratio_range")
    @classmethod
    def provided_ratio_is_nonnull(
        cls, value: AspectRatioRange | None
    ) -> AspectRatioRange:
        if value is None:
            raise ValueError("Omit an unconstrained ratio range instead of null")
        return value


class ReferenceRules(PlanWireModel):
    minimum_total_count: Annotated[StrictInt, Field(ge=0, le=64)]
    maximum_total_count: Annotated[StrictInt, Field(ge=0, le=64)]
    media_limits: Annotated[
        tuple[ReferenceMediaLimit, ...], Field(min_length=0, max_length=3)
    ]
    max_total_video_duration_milliseconds: (
        Annotated[StrictInt, Field(ge=1, le=9223372036854775807)] | None
    )
    max_total_audio_duration_milliseconds: (
        Annotated[StrictInt, Field(ge=1, le=9223372036854775807)] | None
    )
    max_input_video_plus_output_duration_milliseconds: (
        Annotated[StrictInt, Field(ge=1, le=9223372036854775807)] | None
    )


class CapabilityEvidence(PlanWireModel):
    source_kind: Literal[
        "OFFICIAL_DOCUMENTATION", "CONTRACT_VERIFICATION", "PRODUCT_POLICY"
    ]
    reference: Annotated[StrictStr, Field(min_length=1, max_length=2048)]
    checked_at: Instant
    note: Annotated[StrictStr, Field(min_length=1, max_length=2048)]


class ImageDefaults(PlanWireModel):
    size: DraftOutputSize | None
    output_format: ImageOutputFormat | None


class VideoDefaults(PlanWireModel):
    size: DraftOutputSize | None
    output_format: VideoOutputFormat | None
    duration_seconds: Annotated[StrictInt, Field(ge=1, le=2147483647)] | None
    generate_audio: StrictBool | None


class VideoOutputOption(PlanWireModel):
    size_preset_id: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    duration_seconds: PositiveIntDomain


class VerifiedImageProfile(PlanWireModel):
    type: Literal["verified_image_profile"]
    profile_id: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    generation_mode: ImageMode
    prompt_rules: PromptRules
    size_rules: SizeRules
    reference_rules: ReferenceRules
    output_formats: Annotated[tuple[ImageOutputFormat, ...], Field(min_length=1)]
    defaults: ImageDefaults
    evidence: Annotated[tuple[CapabilityEvidence, ...], Field(min_length=1)]


class VerifiedVideoProfile(PlanWireModel):
    type: Literal["verified_video_profile"]
    profile_id: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    generation_mode: VideoMode
    prompt_rules: PromptRules
    size_rules: SizeRules
    output_options: Annotated[tuple[VideoOutputOption, ...], Field(min_length=1)]
    reference_rules: ReferenceRules
    output_formats: Annotated[tuple[VideoOutputFormat, ...], Field(min_length=1)]
    generate_audio_values: Annotated[
        tuple[StrictBool, ...], Field(min_length=1, max_length=2)
    ]
    duration_tolerance_milliseconds: Annotated[
        StrictInt, Field(ge=0, le=9223372036854775807)
    ]
    defaults: VideoDefaults
    evidence: Annotated[tuple[CapabilityEvidence, ...], Field(min_length=1)]


class UnavailableImageProfile(PlanWireModel):
    type: Literal["unavailable_image_profile"]
    profile_id: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    generation_mode: ImageMode
    status: Literal["UNKNOWN", "UNSUPPORTED"]
    reason: Annotated[StrictStr, Field(min_length=1, max_length=2048)]


class UnavailableVideoProfile(PlanWireModel):
    type: Literal["unavailable_video_profile"]
    profile_id: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    generation_mode: VideoMode
    status: Literal["UNKNOWN", "UNSUPPORTED"]
    reason: Annotated[StrictStr, Field(min_length=1, max_length=2048)]


ImageProfile = Annotated[
    VerifiedImageProfile | UnavailableImageProfile, Field(discriminator="type")
]

VideoProfile = Annotated[
    VerifiedVideoProfile | UnavailableVideoProfile, Field(discriminator="type")
]


class ProviderBinding(PlanWireModel):
    provider_id: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    provider_model_id: Annotated[StrictStr, Field(min_length=1, max_length=255)]
    adapter_id: Annotated[StrictStr, Field(min_length=1, max_length=128)]

    @field_validator("provider_id", "provider_model_id", "adapter_id")
    @classmethod
    def nonblank_binding_field(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Provider binding fields must contain non-whitespace text")
        return value


class ImageModelDefinition(PlanWireModel):
    type: Literal["image_model_definition"]
    model_id: Annotated[StrictStr, Field(min_length=1, max_length=255)]
    display_name: Annotated[StrictStr, Field(min_length=1, max_length=255)]
    capability_revision: Annotated[StrictInt, Field(ge=1, le=9223372036854775807)]
    enabled: StrictBool
    provider_binding: ProviderBinding
    profiles: Annotated[tuple[ImageProfile, ...], Field(min_length=1)]


class VideoModelDefinition(PlanWireModel):
    type: Literal["video_model_definition"]
    model_id: Annotated[StrictStr, Field(min_length=1, max_length=255)]
    display_name: Annotated[StrictStr, Field(min_length=1, max_length=255)]
    capability_revision: Annotated[StrictInt, Field(ge=1, le=9223372036854775807)]
    enabled: StrictBool
    provider_binding: ProviderBinding
    profiles: Annotated[tuple[VideoProfile, ...], Field(min_length=1)]


ModelDefinition = Annotated[
    ImageModelDefinition | VideoModelDefinition, Field(discriminator="type")
]


class GenerationSubmissionPolicy(PlanWireModel):
    maximum_tasks_per_submission: Annotated[StrictInt, Field(ge=1, le=2147483647)]


class ModelCatalogSnapshot(PlanWireModel):
    type: Literal["model_catalog_snapshot"]
    catalog_revision: Annotated[StrictInt, Field(ge=1, le=9223372036854775807)]
    submission_policy: GenerationSubmissionPolicy
    models: Annotated[tuple[ModelDefinition, ...], Field(min_length=0)]


class ResolvedPresetSize(PlanWireModel):
    type: Literal["resolved_preset_size"]
    preset: SizePreset


ExecutableSize = Annotated[ResolvedPresetSize | PixelSize, Field(discriminator="type")]


class ImageExecutionParameters(PlanWireModel):
    type: Literal["image_execution_parameters"]
    prompt: Annotated[StrictStr, Field(min_length=1, max_length=64000)]
    size: ExecutableSize
    references: Annotated[
        tuple[ReferenceAsset, ...], Field(min_length=0, max_length=64)
    ]
    output_format: ImageOutputFormat
    generation_mode: ImageMode


class VideoExecutionParameters(PlanWireModel):
    type: Literal["video_execution_parameters"]
    prompt: Annotated[StrictStr, Field(min_length=1, max_length=64000)]
    size: ResolvedPresetSize
    references: Annotated[
        tuple[ReferenceAsset, ...], Field(min_length=0, max_length=64)
    ]
    output_format: VideoOutputFormat
    duration_seconds: Annotated[StrictInt, Field(ge=1, le=2147483647)]
    generate_audio: StrictBool
    generation_mode: VideoMode


class ImageResultExpectations(PlanWireModel):
    type: Literal["image_result_expectations"]
    file_count: Literal[1]
    size: ExecutableSize
    output_format: ImageOutputFormat


class VideoResultExpectations(PlanWireModel):
    type: Literal["video_result_expectations"]
    file_count: Literal[1]
    size: ResolvedPresetSize
    output_format: VideoOutputFormat
    duration_seconds: Annotated[StrictInt, Field(ge=1, le=2147483647)]
    duration_tolerance_milliseconds: Annotated[
        StrictInt, Field(ge=0, le=9223372036854775807)
    ]
    expected_audio: StrictBool


class ImageGenerationRequest(PlanWireModel):
    request_type: Literal["image_generation_plan_v1"]
    schema_version: Literal[1]
    model_selection: ModelSelection
    provider_binding: ProviderBinding
    parameters: ImageExecutionParameters
    result_expectations: ImageResultExpectations


class VideoGenerationRequest(PlanWireModel):
    request_type: Literal["video_generation_plan_v1"]
    schema_version: Literal[1]
    model_selection: ModelSelection
    provider_binding: ProviderBinding
    parameters: VideoExecutionParameters
    result_expectations: VideoResultExpectations


NormalizedGenerationRequest = ImageGenerationRequest | VideoGenerationRequest


class GenerationPlan(PlanWireModel):
    type: Literal["generation_plan"]
    schema_version: Literal[1]
    id: Uuid
    user_id: Uuid
    session_id: Uuid
    revision: Annotated[StrictInt, Field(ge=1, le=9223372036854775807)]
    state: PlanState
    configuration: PlanConfiguration
    frozen_request: NormalizedGenerationRequest | None
    created_at: Instant
    updated_at: Instant
    frozen_at: Instant | None

    @model_validator(mode="after")
    def frozen_state(self):
        if self.state == "DRAFT":
            if self.frozen_request is not None or self.frozen_at is not None:
                raise ValueError("Draft plans cannot have frozen metadata")
        elif self.frozen_request is None or self.frozen_at is None:
            raise ValueError("Frozen plans require frozen metadata")
        return self


class SessionPlanFocus(PlanWireModel):
    session_id: Uuid
    active_plan_id: Uuid | None
    revision: Annotated[StrictInt, Field(ge=0, le=9223372036854775807)]


IssueCode = Literal[
    "MODEL_REQUIRED",
    "MODEL_DISABLED",
    "MODEL_NOT_FOUND",
    "CAPABILITY_REVISION_UNAVAILABLE",
    "CAPABILITIES_CHANGED",
    "PROFILE_UNKNOWN",
    "PROFILE_UNSUPPORTED",
    "REQUIRED_PARAMETER_MISSING",
    "PARAMETER_UNSUPPORTED",
    "REQUIREMENT_UNSATISFIED",
    "REFERENCE_UNAVAILABLE",
    "REFERENCE_TYPE_MISMATCH",
    "REFERENCE_METADATA_UNAVAILABLE",
    "REFERENCE_LIMIT_EXCEEDED",
    "REFERENCE_COMBINATION_UNSUPPORTED",
    "SIZE_DURATION_COMBINATION_UNSUPPORTED",
    "EXECUTION_ADAPTER_UNAVAILABLE",
    "RESULT_VERIFIER_UNAVAILABLE",
    "REQUESTED_FILE_COUNT_UNSUPPORTED",
]

IssueField = Literal[
    "MODEL",
    "PROFILE",
    "GENERATION_MODE",
    "INTENT_DESCRIPTION",
    "PROMPT",
    "ASPECT_RATIO",
    "RESOLUTION_ID",
    "PIXEL_DIMENSIONS",
    "REQUESTED_FILE_COUNT",
    "REFERENCES",
    "OUTPUT_FORMAT",
    "DURATION_SECONDS",
    "GENERATE_AUDIO",
    "PROVIDER_BINDING",
]


class ValidationIssue(PlanWireModel):
    code: IssueCode
    field: IssueField
    reference_index: Annotated[StrictInt, Field(ge=0, le=63)] | None
    message: Annotated[StrictStr, Field(min_length=1, max_length=2048)]
    model_selection: ModelSelection | None


class PlanValidation(PlanWireModel):
    can_submit: StrictBool
    issues: Annotated[tuple[ValidationIssue, ...], Field(min_length=0)]
    validated_catalog_revision: Annotated[
        StrictInt, Field(ge=1, le=9223372036854775807)
    ]


class GenerationPlanSnapshot(PlanWireModel):
    plan: GenerationPlan
    validation: PlanValidation


class PlansCursor(PlanWireModel):
    before_created_at: Instant
    before_id: Uuid


class FocusedPlanQuery(PlanWireModel):
    type: Literal["focused"]


class PlanByIdQuery(PlanWireModel):
    type: Literal["by_id"]
    plan_id: Uuid


class SessionPlansQuery(PlanWireModel):
    type: Literal["list"]
    cursor: PlansCursor | None
    limit: Annotated[StrictInt, Field(ge=1, le=50)]


PlanQuery = Annotated[
    FocusedPlanQuery | PlanByIdQuery | SessionPlansQuery, Field(discriminator="type")
]


class ImageTarget(PlanWireModel):
    media_type: Literal["IMAGE"]
    model_choice: ModelChoice | None


class VideoTarget(PlanWireModel):
    media_type: Literal["VIDEO"]
    model_choice: ModelChoice | None


GenerationTarget = Annotated[
    ImageTarget | VideoTarget, Field(discriminator="media_type")
]


class KeepChange(PlanWireModel):
    op: Literal["KEEP"]


class ClearChange(PlanWireModel):
    op: Literal["CLEAR"]


class SetDescription(PlanWireModel):
    op: Literal["SET"]
    value: Annotated[StrictStr, Field(min_length=1, max_length=64000)]


DescriptionChange = Annotated[KeepChange | SetDescription, Field(discriminator="op")]


class SetPrompt(PlanWireModel):
    op: Literal["SET"]
    value: Annotated[StrictStr, Field(min_length=1, max_length=64000)]


PromptChange = Annotated[
    KeepChange | SetPrompt | ClearChange, Field(discriminator="op")
]


class SetSize(PlanWireModel):
    op: Literal["SET"]
    value: DraftOutputSize


SizeChange = Annotated[KeepChange | SetSize | ClearChange, Field(discriminator="op")]


class SetImageOutputFormat(PlanWireModel):
    op: Literal["SET"]
    value: ImageOutputFormat


ImageOutputFormatChange = Annotated[
    KeepChange | SetImageOutputFormat | ClearChange, Field(discriminator="op")
]


class SetVideoOutputFormat(PlanWireModel):
    op: Literal["SET"]
    value: VideoOutputFormat


VideoOutputFormatChange = Annotated[
    KeepChange | SetVideoOutputFormat | ClearChange, Field(discriminator="op")
]


class SetDuration(PlanWireModel):
    op: Literal["SET"]
    value: Annotated[StrictInt, Field(ge=1, le=2147483647)]


DurationChange = Annotated[
    KeepChange | SetDuration | ClearChange, Field(discriminator="op")
]


class SetBoolean(PlanWireModel):
    op: Literal["SET"]
    value: StrictBool


BooleanChange = Annotated[
    KeepChange | SetBoolean | ClearChange, Field(discriminator="op")
]


class SetImageMode(PlanWireModel):
    op: Literal["SET"]
    value: ImageMode


ImageModeChange = Annotated[
    KeepChange | SetImageMode | ClearChange, Field(discriminator="op")
]


class SetVideoMode(PlanWireModel):
    op: Literal["SET"]
    value: VideoMode


VideoModeChange = Annotated[
    KeepChange | SetVideoMode | ClearChange, Field(discriminator="op")
]


class SetReferences(PlanWireModel):
    op: Literal["SET"]
    value: Annotated[tuple[ReferenceAsset, ...], Field(min_length=0, max_length=64)]


ReferencesChange = Annotated[
    KeepChange | SetReferences | ClearChange, Field(discriminator="op")
]


class ImageParametersPatch(PlanWireModel):
    prompt: PromptChange
    size: SizeChange
    references: ReferencesChange
    output_format: ImageOutputFormatChange


class VideoParametersPatch(PlanWireModel):
    prompt: PromptChange
    size: SizeChange
    references: ReferencesChange
    output_format: VideoOutputFormatChange
    duration_seconds: DurationChange
    generate_audio: BooleanChange


class SetRequestedFileCount(PlanWireModel):
    op: Literal["SET"]
    value: Annotated[StrictInt, Field(ge=1, le=2147483647)]


RequestedFileCountChange = Annotated[
    KeepChange | SetRequestedFileCount | ClearChange, Field(discriminator="op")
]


class ImageRequirementsPatch(PlanWireModel):
    prompt: PromptChange
    size: SizeChange
    requested_file_count: RequestedFileCountChange
    references: ReferencesChange
    output_format: ImageOutputFormatChange
    generation_mode: ImageModeChange


class VideoRequirementsPatch(PlanWireModel):
    prompt: PromptChange
    size: SizeChange
    requested_file_count: RequestedFileCountChange
    references: ReferencesChange
    output_format: VideoOutputFormatChange
    duration_seconds: DurationChange
    generate_audio: BooleanChange
    generation_mode: VideoModeChange


class ImagePlanPatch(PlanWireModel):
    type: Literal["image_plan_patch"]
    description: DescriptionChange
    requirements: ImageRequirementsPatch
    parameters: ImageParametersPatch


class VideoPlanPatch(PlanWireModel):
    type: Literal["video_plan_patch"]
    description: DescriptionChange
    requirements: VideoRequirementsPatch
    parameters: VideoParametersPatch


PlanPatch = Annotated[ImagePlanPatch | VideoPlanPatch, Field(discriminator="type")]


class GetModelCatalogArguments(PlanWireModel):
    media_type: MediaType | None


class ReadGenerationPlanArguments(PlanWireModel):
    query: PlanQuery


class CreateImagePlanArguments(PlanWireModel):
    target: ImageTarget
    intent: ImageIntent
    parameters: ImageParameters
    expected_focus_revision: Annotated[StrictInt, Field(ge=0, le=9223372036854775807)]


class CreateVideoPlanArguments(PlanWireModel):
    target: VideoTarget
    intent: VideoIntent
    parameters: VideoParameters
    expected_focus_revision: Annotated[StrictInt, Field(ge=0, le=9223372036854775807)]


CreateGenerationPlanArguments = CreateImagePlanArguments | CreateVideoPlanArguments


class UpdateGenerationPlanArguments(PlanWireModel):
    plan_id: Uuid
    expected_plan_revision: Annotated[StrictInt, Field(ge=1, le=9223372036854775807)]
    expected_focus_revision: Annotated[StrictInt, Field(ge=0, le=9223372036854775807)]
    patch: PlanPatch


class ChangeGenerationTargetArguments(PlanWireModel):
    plan_id: Uuid
    expected_plan_revision: Annotated[StrictInt, Field(ge=1, le=9223372036854775807)]
    expected_focus_revision: Annotated[StrictInt, Field(ge=0, le=9223372036854775807)]
    target: GenerationTarget


class CloneGenerationPlanArguments(PlanWireModel):
    source_plan_id: Uuid
    expected_source_plan_revision: Annotated[
        StrictInt, Field(ge=1, le=9223372036854775807)
    ]
    expected_focus_revision: Annotated[StrictInt, Field(ge=0, le=9223372036854775807)]


class FocusGenerationPlanArguments(PlanWireModel):
    plan_id: Uuid
    expected_focus_revision: Annotated[StrictInt, Field(ge=0, le=9223372036854775807)]


class SubmitGenerationPlanArguments(PlanWireModel):
    plan_id: Uuid
    expected_plan_revision: Annotated[StrictInt, Field(ge=1, le=9223372036854775807)]
    expected_focus_revision: Annotated[StrictInt, Field(ge=0, le=9223372036854775807)]


GenerationPlanToolName = Literal[
    "get_model_catalog",
    "read_generation_plan",
    "create_generation_plan",
    "update_generation_plan",
    "change_generation_target",
    "clone_generation_plan",
    "focus_generation_plan",
    "submit_generation_plan",
]


class GetModelCatalogInvocation(PlanWireModel):
    type: Literal["generation_plan_tool_invocation"]
    operation_id: Uuid
    session_id: Uuid
    tool_name: Literal["get_model_catalog"]
    arguments: GetModelCatalogArguments


class ReadGenerationPlanInvocation(PlanWireModel):
    type: Literal["generation_plan_tool_invocation"]
    operation_id: Uuid
    session_id: Uuid
    tool_name: Literal["read_generation_plan"]
    arguments: ReadGenerationPlanArguments


class CreateGenerationPlanInvocation(PlanWireModel):
    type: Literal["generation_plan_tool_invocation"]
    operation_id: Uuid
    session_id: Uuid
    tool_name: Literal["create_generation_plan"]
    arguments: CreateGenerationPlanArguments


class UpdateGenerationPlanInvocation(PlanWireModel):
    type: Literal["generation_plan_tool_invocation"]
    operation_id: Uuid
    session_id: Uuid
    tool_name: Literal["update_generation_plan"]
    arguments: UpdateGenerationPlanArguments


class ChangeGenerationTargetInvocation(PlanWireModel):
    type: Literal["generation_plan_tool_invocation"]
    operation_id: Uuid
    session_id: Uuid
    tool_name: Literal["change_generation_target"]
    arguments: ChangeGenerationTargetArguments


class CloneGenerationPlanInvocation(PlanWireModel):
    type: Literal["generation_plan_tool_invocation"]
    operation_id: Uuid
    session_id: Uuid
    tool_name: Literal["clone_generation_plan"]
    arguments: CloneGenerationPlanArguments


class FocusGenerationPlanInvocation(PlanWireModel):
    type: Literal["generation_plan_tool_invocation"]
    operation_id: Uuid
    session_id: Uuid
    tool_name: Literal["focus_generation_plan"]
    arguments: FocusGenerationPlanArguments


class SubmitGenerationPlanInvocation(PlanWireModel):
    type: Literal["generation_plan_tool_invocation"]
    operation_id: Uuid
    session_id: Uuid
    tool_name: Literal["submit_generation_plan"]
    arguments: SubmitGenerationPlanArguments


JavaPlanToolInvocation = Annotated[
    GetModelCatalogInvocation
    | ReadGenerationPlanInvocation
    | CreateGenerationPlanInvocation
    | UpdateGenerationPlanInvocation
    | ChangeGenerationTargetInvocation
    | CloneGenerationPlanInvocation
    | FocusGenerationPlanInvocation
    | SubmitGenerationPlanInvocation,
    Field(discriminator="tool_name"),
]


class GenerationPlanReadResult(PlanWireModel):
    type: Literal["generation_plan_read_result"]
    focus: SessionPlanFocus
    plans: Annotated[
        tuple[GenerationPlanSnapshot, ...], Field(min_length=0, max_length=50)
    ]
    next_cursor: PlansCursor | None


class AppliedDefault(PlanWireModel):
    field: ParameterKey
    source_model: ModelSelection


class PlanMutationReceipt(PlanWireModel):
    type: Literal["generation_plan_mutation_receipt"]
    action: Literal[
        "CREATED", "UPDATED", "MODEL_CHANGED", "MEDIA_PLAN_CREATED", "CLONED"
    ]
    source_plan_id: Uuid | None
    snapshot: GenerationPlanSnapshot
    focus: SessionPlanFocus
    cleared_execution_fields: Annotated[
        tuple[ParameterKey, ...], Field(min_length=0, max_length=11)
    ]
    non_migrated_requirement_fields: Annotated[
        tuple[IssueField, ...], Field(min_length=0, max_length=14)
    ]
    applied_defaults: Annotated[
        tuple[AppliedDefault, ...], Field(min_length=0, max_length=11)
    ]


class PlanFocusReceipt(PlanWireModel):
    type: Literal["generation_plan_focus_receipt"]
    focus: SessionPlanFocus


class SubmittedTask(PlanWireModel):
    index: Annotated[StrictInt, Field(ge=0, le=2147483646)]
    task_id: Uuid
    task_operation_id: Uuid


class PlanTaskSubmissionReceipt(PlanWireModel):
    type: Literal["plan_task_submission_receipt"]
    plan_id: Uuid
    frozen_plan_revision: Annotated[StrictInt, Field(ge=1, le=9223372036854775807)]
    accepted: Literal[True]
    requested_file_count: Annotated[StrictInt, Field(ge=1, le=2147483647)]
    tasks: Annotated[tuple[SubmittedTask, ...], Field(min_length=1)]
    focus: SessionPlanFocus


PlanToolErrorCode = Literal[
    "INVALID_ARGUMENT",
    "RESOURCE_NOT_FOUND",
    "PLAN_FROZEN",
    "PLAN_REVISION_CONFLICT",
    "FOCUS_REVISION_CONFLICT",
    "PLAN_NOT_FOCUSED",
    "TARGET_NOT_AVAILABLE",
    "CAPABILITIES_CHANGED",
    "PLAN_NOT_SUBMITTABLE",
    "OPERATION_IDENTITY_CONFLICT",
    "CONCURRENT_MODIFICATION",
    "DEPENDENT_TOOL_BATCH",
]


class PlanToolRejection(PlanWireModel):
    type: Literal["generation_plan_tool_rejection"]
    code: PlanToolErrorCode
    message: Annotated[StrictStr, Field(min_length=1, max_length=2048)]
    current_plan_revision: (
        Annotated[StrictInt, Field(ge=1, le=9223372036854775807)] | None
    )
    current_focus_revision: (
        Annotated[StrictInt, Field(ge=0, le=9223372036854775807)] | None
    )
    issues: Annotated[tuple[ValidationIssue, ...], Field(min_length=0)]
    business_execution: Literal["NOT_PERFORMED"]


GenerationPlanToolResult = Annotated[
    ModelCatalogSnapshot
    | GenerationPlanReadResult
    | PlanMutationReceipt
    | PlanFocusReceipt
    | PlanTaskSubmissionReceipt
    | PlanToolRejection,
    Field(discriminator="type"),
]


class JavaPlanToolOperationReceipt(PlanWireModel):
    type: Literal["generation_plan_tool_operation_receipt"]
    schema_version: Literal[1]
    operation_id: Uuid
    session_id: Uuid
    tool_name: GenerationPlanToolName
    request_hash: Sha256
    result: GenerationPlanToolResult


class VerifiedImageMetadata(PlanWireModel):
    type: Literal["verified_image_metadata"]
    asset_id: Uuid
    sha256: Sha256
    size_bytes: Annotated[StrictInt, Field(ge=1, le=9223372036854775807)]
    detected_mime_type: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    dimensions: PixelDimensions


class VerifiedVideoMetadata(PlanWireModel):
    type: Literal["verified_video_metadata"]
    asset_id: Uuid
    sha256: Sha256
    size_bytes: Annotated[StrictInt, Field(ge=1, le=9223372036854775807)]
    detected_mime_type: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    dimensions: PixelDimensions
    duration_milliseconds: Annotated[StrictInt, Field(ge=1, le=9223372036854775807)]
    has_audio: StrictBool


class VerifiedAudioMetadata(PlanWireModel):
    type: Literal["verified_audio_metadata"]
    asset_id: Uuid
    sha256: Sha256
    size_bytes: Annotated[StrictInt, Field(ge=1, le=9223372036854775807)]
    detected_mime_type: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    duration_milliseconds: Annotated[StrictInt, Field(ge=1, le=9223372036854775807)]


VerifiedMediaMetadata = Annotated[
    VerifiedImageMetadata | VerifiedVideoMetadata | VerifiedAudioMetadata,
    Field(discriminator="type"),
]


class GenerationPlanContext(PlanWireModel):
    focus: SessionPlanFocus
    plan: GenerationPlanSnapshot | None
    catalog: ModelCatalogSnapshot


INVOCATION_ADAPTER = TypeAdapter(JavaPlanToolInvocation)
RESULT_ADAPTER = TypeAdapter(GenerationPlanToolResult)
