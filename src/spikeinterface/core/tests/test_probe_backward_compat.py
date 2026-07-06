"""
Backward compatibility tests: load recordings saved with spikeinterface 0.104.*
using the current code and verify that probe information is correctly preserved.

Fixtures are produced by running:
    python .github/scripts/create_probe_compat_fixtures.py <output_dir>
with spikeinterface==0.104.* installed.

The GH Action workflow probe_backward_compat.yml does this automatically.
Set SI_PROBE_COMPAT_FIXTURES_DIR to point at the fixture directory if running locally.
"""

import os
import numpy as np
import pytest
from pathlib import Path

from spikeinterface.core import load

FIXTURES_DIR = Path(os.environ.get("SI_PROBE_COMPAT_FIXTURES_DIR", "probe_compat_fixtures"))

pytestmark = pytest.mark.skipif(
    not FIXTURES_DIR.exists(),
    reason=(
        f"Probe compatibility fixtures not found at '{FIXTURES_DIR}'. "
        "Run .github/scripts/create_probe_compat_fixtures.py with spikeinterface==0.104.* first, "
        "or set SI_PROBE_COMPAT_FIXTURES_DIR to the fixture directory."
    ),
)


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------


def _check_single_probe(rec):
    assert rec.has_probe(), "Recording must have a probe after loading"
    assert rec.get_num_channels() == 8
    probes = rec.get_probes()
    assert len(probes) == 1
    probe = probes[0]
    assert probe.annotations.get("name") == "test_probe"
    assert probe.annotations.get("manufacturer") == "test_vendor"
    assert list(probe.contact_ids) == [f"e{i}" for i in range(8)]
    # After loading, device_channel_indices must be sorted 0..N-1
    assert np.array_equal(probe.device_channel_indices, np.arange(8))


def _check_two_probes(rec):
    assert rec.has_probe()
    assert rec.get_num_channels() == 16
    probes = rec.get_probes()
    assert len(probes) == 2, "Both probes must survive after loading"
    probe_names = {p.annotations.get("name") for p in probes}
    assert probe_names == {"probe_A", "probe_B"}, "Per-probe names must be preserved"
    manufacturers = {p.annotations.get("manufacturer") for p in probes}
    assert manufacturers == {"vendor_X", "vendor_Y"}, "Per-probe manufacturers must be preserved"
    all_contact_ids = set()
    for p in probes:
        all_contact_ids.update(p.contact_ids.tolist())
    assert all_contact_ids == {f"a{i}" for i in range(8)} | {f"b{i}" for i in range(8)}
    groups = rec.get_property("group")
    assert len(np.unique(groups)) == 2, "Each probe must have its own group"


def _check_shuffled_probe(rec):
    assert rec.has_probe()
    assert rec.get_num_channels() == 8
    probe = rec.get_probes()[0]
    assert probe.annotations.get("name") == "shuffled_probe"
    assert probe.annotations.get("manufacturer") == "shuffle_vendor"
    # After the old set_probe sorted contacts by device_channel_indices and
    # normalised them, the stored probegroup has dci = 0..7.
    assert np.array_equal(probe.device_channel_indices, np.arange(8))
    traces = rec.get_traces(segment_index=0)
    assert traces.shape == (1000, 8)


# ---------------------------------------------------------------------------
# Binary folder fixtures
# ---------------------------------------------------------------------------


def test_single_probe_binary_compat():
    _check_single_probe(load(FIXTURES_DIR / "single_probe_binary"))


def test_two_probe_binary_compat():
    _check_two_probes(load(FIXTURES_DIR / "two_probe_binary"))


def test_shuffled_probe_binary_compat():
    _check_shuffled_probe(load(FIXTURES_DIR / "shuffled_probe_binary"))


def test_interleaved_probe_binary_compat():
    _check_two_probes(load(FIXTURES_DIR / "two_probe_interleaved_binary"))


# ---------------------------------------------------------------------------
# Zarr dump fixtures
# ---------------------------------------------------------------------------


def test_single_probe_zarr_compat():
    _check_single_probe(load(FIXTURES_DIR / "single_probe.zarr"))


def test_two_probe_zarr_compat():
    _check_two_probes(load(FIXTURES_DIR / "two_probe.zarr"))


def test_shuffled_probe_zarr_compat():
    _check_shuffled_probe(load(FIXTURES_DIR / "shuffled_probe.zarr"))


def test_interleaved_probe_zarr_compat():
    _check_two_probes(load(FIXTURES_DIR / "two_probe_interleaved.zarr"))


# ---------------------------------------------------------------------------
# JSON dump fixtures
# ---------------------------------------------------------------------------


def test_single_probe_json_compat():
    _check_single_probe(load(FIXTURES_DIR / "single_probe.json"))


def test_two_probe_json_compat():
    _check_two_probes(load(FIXTURES_DIR / "two_probe.json"))


def test_shuffled_probe_json_compat():
    _check_shuffled_probe(load(FIXTURES_DIR / "shuffled_probe.json"))


def test_interleaved_probe_json_compat():
    _check_two_probes(load(FIXTURES_DIR / "two_probe_interleaved.json"))


# ---------------------------------------------------------------------------
# Pickle dump fixtures
# ---------------------------------------------------------------------------


def test_single_probe_pickle_compat():
    _check_single_probe(load(FIXTURES_DIR / "single_probe.pkl"))


def test_two_probe_pickle_compat():
    _check_two_probes(load(FIXTURES_DIR / "two_probe.pkl"))


def test_shuffled_probe_pickle_compat():
    _check_shuffled_probe(load(FIXTURES_DIR / "shuffled_probe.pkl"))


def test_interleaved_probe_pickle_compat():
    _check_two_probes(load(FIXTURES_DIR / "two_probe_interleaved.pkl"))
