import json
import unittest

import numpy as np
import pytest

from spikeinterface.core import BaseSorting, NumpyRecording
from spikeinterface.sorters import VanillaSortSorter, available_sorters, run_sorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


def make_recording(multi_segment=False):
    """Short deterministic four-channel fixture with enough events for default K."""
    rng = np.random.default_rng(42)
    traces = rng.normal(size=(7500, 4)).astype("float32")
    t = np.arange(-30, 30)
    waveform = -np.exp(-((t / 3.0) ** 2)) + 0.35 * np.exp(-(((t - 9) / 6.0) ** 2))
    for index, center in enumerate(range(100, len(traces) - 100, 150)):
        amplitudes = np.roll([8.0, 5.0, 2.0, 1.0], index % 4)
        traces[center - 30 : center + 30] += (waveform[:, None] * amplitudes).astype("float32")
    segments = [traces, traces[:3750].copy()] if multi_segment else [traces]
    recording = NumpyRecording(segments, 30000, channel_ids=["a", "b", "c", "d"])
    recording.set_dummy_probe_from_locations(np.array([[0, 0], [20, 0], [0, 20], [20, 20]]))
    return recording


@pytest.mark.skipif(not VanillaSortSorter.is_installed(), reason="vanillasort >= 0.1.0 not installed")
class VanillaSortCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = VanillaSortSorter

    def setUp(self):
        # Keep the common tests, using 0.25 s instead of the suite's 60 s fixture.
        self.recording = make_recording().save(folder=self.cache_folder / "rec", verbose=False, format="binary")


@pytest.mark.skipif(not VanillaSortSorter.is_installed(), reason="vanillasort >= 0.1.0 not installed")
def test_vanillasort_direct_and_wrapper(tmp_path):
    import vanillasort

    assert "vanillasort" in available_sorters()
    assert VanillaSortSorter.get_sorter_version() == vanillasort.__version__
    assert set(VanillaSortSorter.default_params()) == set(VanillaSortSorter.params_description())
    assert not VanillaSortSorter.use_gpu({"device": "cpu"})
    recording = make_recording(multi_segment=True).save(folder=tmp_path / "rec", verbose=False)
    params = dict(components=4, device="cpu", seed=0)
    direct = vanillasort.sort(recording, verbose=False, **params)
    folder = tmp_path / "wrapped"
    wrapped = run_sorter("vanillasort", recording, folder=folder, verbose=True, **params)
    assert isinstance(wrapped, BaseSorting)
    assert wrapped.get_num_segments() == 2
    assert wrapped.to_spike_vector().size > 0
    np.testing.assert_array_equal(wrapped.unit_ids, direct.unit_ids)
    np.testing.assert_array_equal(wrapped.to_spike_vector(), direct.to_spike_vector())
    np.testing.assert_array_equal(wrapped.get_property("main_channel_id"), direct.get_property("main_channel_id"))
    restored = VanillaSortSorter.get_result_from_folder(folder)
    np.testing.assert_array_equal(restored.to_spike_vector(), direct.to_spike_vector())
    np.testing.assert_array_equal(restored.get_property("main_channel_id"), direct.get_property("main_channel_id"))
    assert restored.has_recording()
    assert restored.sorting_info["params"]["sorter_params"]["seed"] == 0
    log = json.loads((folder / "spikeinterface_log.json").read_text())
    assert log["sorter_version"] == vanillasort.__version__
    assert log["error"] is False and log["run_time"] > 0
    assert any("Completed:" in line for line in log["runtime_trace"])

    # BaseSorter retains normal error logs for incompatible model input.
    bad = NumpyRecording([np.zeros((200, 4), dtype="float32")], 20000)
    bad.set_dummy_probe_from_locations(recording.get_channel_locations())
    bad = bad.save(folder=tmp_path / "bad_rec", verbose=False)
    failed_folder = tmp_path / "failed"
    failed = run_sorter("vanillasort", bad, folder=failed_folder, raise_error=False, with_output=False, **params)
    assert failed is None
    failed_log = json.loads((failed_folder / "spikeinterface_log.json").read_text())
    assert failed_log["error"] is True
    assert "30 kHz" in "\n".join(failed_log["error_trace"])
