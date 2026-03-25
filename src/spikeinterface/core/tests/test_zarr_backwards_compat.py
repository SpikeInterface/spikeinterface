"""
Tests for zarr format backward compatibility.

Loads zarr v2 fixtures (generated with spikeinterface==0.104.0 and zarr<3) using
the current spikeinterface version, which uses zarr>=3.

The fixtures directory is passed via the ZARR_V2_FIXTURES_PATH environment variable.
These tests are skipped when that variable is not set (i.e. in normal CI runs).

To run locally:
    ZARR_V2_FIXTURES_PATH=/tmp/zarr_v2_fixtures pytest test_zarr_backwards_compat.py -v
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import spikeinterface as si

FIXTURES_PATH = os.environ.get("ZARR_V2_FIXTURES_PATH")

pytestmark = pytest.mark.skipif(
    FIXTURES_PATH is None,
    reason="ZARR_V2_FIXTURES_PATH environment variable not set",
)


@pytest.fixture(scope="module")
def fixtures_dir() -> Path:
    return Path(FIXTURES_PATH)


@pytest.fixture(scope="module")
def expected(fixtures_dir: Path) -> dict:
    with open(fixtures_dir / "expected_values.json") as f:
        return json.load(f)


def test_load_recording(fixtures_dir, expected):
    recording = si.load(fixtures_dir / "recording.zarr")
    exp = expected["recording"]

    assert recording.get_num_channels() == exp["num_channels"]
    assert recording.get_num_segments() == exp["num_segments"]
    assert recording.get_sampling_frequency() == exp["sampling_frequency"]
    assert str(recording.get_dtype()) == exp["dtype"]

    for seg in range(recording.get_num_segments()):
        assert recording.get_num_samples(seg) == exp["num_samples_per_segment"][seg]

    assert list(recording.get_channel_ids()) == exp["channel_ids"]

    traces = recording.get_traces(start_frame=0, end_frame=10, segment_index=0)
    np.testing.assert_array_equal(traces, np.array(exp["traces_seg0_first10"]))


def test_load_sorting(fixtures_dir, expected):
    sorting = si.load(fixtures_dir / "sorting.zarr")
    exp = expected["sorting"]

    assert sorting.get_num_segments() == exp["num_segments"]
    assert sorting.get_sampling_frequency() == exp["sampling_frequency"]
    assert list(sorting.get_unit_ids()) == exp["unit_ids"]

    for uid in sorting.unit_ids:
        spike_train = sorting.get_unit_spike_train(unit_id=uid, segment_index=0)
        np.testing.assert_array_equal(spike_train, np.array(exp["spike_trains_seg0"][str(uid)]))


def test_load_sorting_analyzer(fixtures_dir, expected):
    analyzer = si.load(fixtures_dir / "analyzer.zarr")
    exp = expected["analyzer"]

    assert analyzer.get_num_units() == exp["num_units"]
    assert analyzer.get_num_channels() == exp["num_channels"]

    templates_ext = analyzer.get_extension("templates")
    assert templates_ext is not None, "templates extension not found in analyzer"

    templates = templates_ext.get_data()
    assert list(templates.shape) == exp["templates_shape"]
