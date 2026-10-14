"""The additive catalog ratio constraint preserves archived legacy wire values."""

import json

import pytest
from pydantic import ValidationError

from lux_agent.generation_plan import ReferenceMediaLimit


def reference_limit():
    return {
        "media_type": "IMAGE",
        "minimum_count": 0,
        "maximum_count": 30,
        "mime_types": ["image/png", "image/jpeg"],
        "max_asset_bytes": 1572864,
        "dimensions": None,
        "max_duration_milliseconds": None,
    }


def test_old_and_extended_reference_capabilities_round_trip_without_hash_drift():
    old = reference_limit()
    assert (
        json.loads(
            ReferenceMediaLimit.model_validate_json(json.dumps(old)).model_dump_json()
        )
        == old
    )
    current = old | {
        "aspect_ratio_range": {
            "minimum": {"width_units": 2, "height_units": 5},
            "maximum": {"width_units": 5, "height_units": 2},
        }
    }
    assert (
        json.loads(
            ReferenceMediaLimit.model_validate_json(
                json.dumps(current)
            ).model_dump_json()
        )
        == current
    )


def test_new_ratio_range_rejects_explicit_null_like_the_shared_schema():
    with pytest.raises(ValidationError):
        ReferenceMediaLimit.model_validate_json(
            json.dumps(reference_limit() | {"aspect_ratio_range": None})
        )
