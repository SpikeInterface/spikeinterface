import builtins
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pytest

from spikeinterface.core import BaseSorting, NumpyRecording
from spikeinterface.sorters import VanillaSortSorter, available_sorters, run_sorter
from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


def test_vanillasort_discovery_without_heavy_imports(monkeypatch):
    from spikeinterface.sorters.external import vanillasort

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"vanillasort", "torch"}, f"Discovery imported {name}"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(vanillasort.importlib.util, "find_spec", lambda name: None)
    assert not VanillaSortSorter.is_installed()

    monkeypatch.setattr(vanillasort.importlib.util, "find_spec", lambda name: object())
    installed_version = "test-version"
    monkeypatch.setattr(vanillasort, "version", lambda name: installed_version)
    assert VanillaSortSorter.is_installed()
    assert VanillaSortSorter.get_sorter_version() == installed_version
    assert VanillaSortSorter.gpu_capability == "nvidia-optional"
    for params, expected in [
        ({}, True),
        ({"device": "auto"}, True),
        ({"device": "cpu"}, False),
        ({"device": "cuda"}, True),
        ({"device": "cuda:1"}, True),
    ]:
        assert VanillaSortSorter.use_gpu(params) is expected

    # Version checks must also stay lightweight and fail before inference.
    for installed_version in ("0.1.0", "0.2.0"):
        with pytest.raises(RuntimeError, match="VanillaSort >= 0.2.1.*pip install --upgrade vanillasort"):
            VanillaSortSorter._check_params(None, None, VanillaSortSorter.default_params())
    for installed_version in ("0.2.1", "0.10.0"):
        params = VanillaSortSorter.default_params()
        assert VanillaSortSorter._check_params(None, None, params) == params


@pytest.fixture
def tiny_model(tmp_path):
    """Use real small model architectures; external CI never downloads weights."""
    import torch
    from vanillasort.checkpoints import load_config
    from vanillasort.models import VanillaDet, HuiduRep

    config = load_config()
    config["detector_architecture"].update(
        d_model=8, nhead=2, num_layers=1, dim_ff=16, frontend_channels=8, max_len=5000
    )
    config["huidurep_architecture"].update(embedding_dim=16, n_heads=2, ff_dim=16, num_layers=3)
    config["chunk_len"] = 128
    config["profiles"]["d1"].update(base_threshold=0.0001, direct_keep=0.0001)
    model_path = tmp_path / "tiny.pt"
    with torch.random.fork_rng():
        torch.manual_seed(0)
        torch.save(
            {
                "config": config,
                "detector_state_dict": VanillaDet(**config["detector_architecture"]).state_dict(),
                "huidurep_state_dict": HuiduRep(**config["huidurep_architecture"]).state_dict(),
            },
            model_path,
        )
    return model_path


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


@pytest.mark.skipif(not VanillaSortSorter.is_installed(), reason="vanillasort not installed")
class VanillaSortCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = VanillaSortSorter

    @pytest.fixture(autouse=True)
    def configure_model(self, tiny_model):
        self.model_path = tiny_model

    def setUp(self):
        defaults = patch.dict(
            self.SorterClass._default_params,
            {
                "model_path": str(self.model_path),
                "components": 2,
                "device": "cpu",
            },
        )
        defaults.start()
        self.addCleanup(defaults.stop)
        # Keep the common tests, using 0.25 s instead of the suite's 60 s fixture.
        self.recording = make_recording().save(folder=self.cache_folder / "rec", verbose=False, format="binary")


@pytest.mark.skipif(not VanillaSortSorter.is_installed(), reason="vanillasort not installed")
def test_vanillasort_direct_and_wrapper(tmp_path, tiny_model, capsys):
    import vanillasort

    assert "vanillasort" in available_sorters()
    assert VanillaSortSorter.get_sorter_version() == vanillasort.__version__
    assert set(VanillaSortSorter.default_params()) == set(VanillaSortSorter.params_description())
    assert not VanillaSortSorter.use_gpu({"device": "cpu"})
    recording = make_recording(multi_segment=True).save(folder=tmp_path / "rec", verbose=False)
    params = dict(components=4, device="cpu", seed=0, model_path=str(tiny_model))
    direct = vanillasort.sort(recording, verbose=False, **params)
    assert capsys.readouterr().out == ""
    folder = tmp_path / "wrapped"
    stdout = sys.stdout
    wrapped = run_sorter("vanillasort", recording, folder=folder, verbose=True, **params)
    assert sys.stdout is stdout
    progress = capsys.readouterr().out
    for message in ("VanillaSort ", "Preprocessing ", "Detector:", "Completed:"):
        assert message in progress
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
    assert all(line in progress for line in log["runtime_trace"])

    quiet_folder = tmp_path / "quiet"
    quiet = run_sorter("vanillasort", recording, folder=quiet_folder, verbose=False, **params)
    assert sys.stdout is stdout
    assert capsys.readouterr().out == ""
    np.testing.assert_array_equal(quiet.to_spike_vector(), direct.to_spike_vector())
    quiet_log = json.loads((quiet_folder / "spikeinterface_log.json").read_text())
    assert any("Completed:" in line for line in quiet_log["runtime_trace"])

    # BaseSorter retains normal error logs for incompatible model input.
    bad = NumpyRecording([np.zeros((200, 4), dtype="float32")], 20000)
    bad.set_dummy_probe_from_locations(recording.get_channel_locations())
    bad = bad.save(folder=tmp_path / "bad_rec", verbose=False)
    failed_folder = tmp_path / "failed"
    failed = run_sorter(
        "vanillasort", bad, folder=failed_folder, verbose=True, raise_error=False, with_output=False, **params
    )
    assert sys.stdout is stdout
    assert failed is None
    failed_log = json.loads((failed_folder / "spikeinterface_log.json").read_text())
    assert failed_log["error"] is True
    assert "30 kHz" in "\n".join(failed_log["error_trace"])


@pytest.mark.skipif(not VanillaSortSorter.is_installed(), reason="vanillasort not installed")
def test_vanillasort_extended_params_roundtrip(tmp_path, tiny_model):
    import vanillasort

    base = make_recording(multi_segment=True)
    segments = []
    for segment in range(base.get_num_segments()):
        traces = base.get_traces(segment_index=segment)
        segments.append(np.column_stack([traces, 0.8 * np.roll(traces, 13, axis=0)]))
    recording = NumpyRecording(segments, 30000, channel_ids=list("abcdefgh"))
    recording.set_dummy_probe_from_locations(np.column_stack([np.tile([0, 20], 4), np.repeat([0, 20, 40, 60], 2)]))
    recording = recording.save(folder=tmp_path / "rec", verbose=False)
    options = dict(
        enable_gate=False,
        threshold=0.0002,
        direct_keep=0.75,
        primary_snr=4,
        secondary_snr=2.5,
        min_secondary_channels=2,
        neighborhood_mode="cover",
        target_units=1,
        postprocessing={"core_filter": False, "on_shortfall": "return_available", "min_cosine": 0.8},
    )
    params = dict(components=4, device="cpu", seed=0, model_path=str(tiny_model), **options)
    direct = vanillasort.sort(recording, verbose=False, **params)
    folder = tmp_path / "wrapped"
    wrapped = run_sorter("vanillasort", recording, folder=folder, verbose=False, **params)
    np.testing.assert_array_equal(wrapped.unit_ids, direct.unit_ids)
    np.testing.assert_array_equal(wrapped.to_spike_vector(), direct.to_spike_vector())
    np.testing.assert_array_equal(wrapped.get_property("main_channel_id"), direct.get_property("main_channel_id"))
    assert wrapped.get_num_segments() == 2 and wrapped.get_num_units() == 2
    for segment in range(2):
        active_units = [u for u in wrapped.unit_ids if len(wrapped.get_unit_spike_train(u, segment_index=segment))]
        assert len(active_units) == 1

    saved = json.loads((folder / "spikeinterface_params.json").read_text())["sorter_params"]
    assert {name: saved[name] for name in options} == options
    inference = folder / "sorter_output" / "inference"
    audit = json.loads((inference / "run.json").read_text())
    assert audit["neighborhood_mode"] == "cover" and audit["target_units"] == 1
    assert len(audit["runs"]) == 4  # two covering patches in each segment
    gate = wrapped.get_annotation("detection_gate")
    for name in ("enable_gate", "threshold", "direct_keep", "primary_snr", "secondary_snr", "min_secondary_channels"):
        assert gate[name] == options[name]
    assert gate["mode"] == "off"
    for segment, diagnostic in enumerate(wrapped.get_annotation("fixed_k_postprocessing")):
        assert diagnostic["requested_units"] == diagnostic["output_units"] == 1
        for name, value in options["postprocessing"].items():
            assert diagnostic["config"][name] == value
        assert (inference / f"postprocessing_segment{segment}.json").is_file()

    # A high score threshold produces empty results with an explicit shortfall policy.
    empty_params = dict(params, threshold=1.0)
    empty = run_sorter("vanillasort", recording, folder=tmp_path / "empty", verbose=False, **empty_params)
    assert empty.get_num_units() == 0 and empty.get_num_segments() == 2
    assert all(d["status"] == "partial" for d in empty.get_annotation("fixed_k_postprocessing"))

    failed_folder = tmp_path / "failed"
    failed = run_sorter(
        "vanillasort",
        recording,
        folder=failed_folder,
        verbose=False,
        raise_error=False,
        with_output=False,
        **dict(empty_params, postprocessing={"core_filter": False}),
    )
    assert failed is None
    failure = json.loads((failed_folder / "spikeinterface_log.json").read_text())
    assert failure["error"] and "InsufficientAnchorsError" in "\n".join(failure["error_trace"])
    diagnostic = json.loads(
        (failed_folder / "sorter_output" / "inference" / "postprocessing_segment0.json").read_text()
    )
    assert diagnostic["failure_reason"] == "empty_input"

    with pytest.raises(ValueError, match="Invalid parameters.*gate_fn"):
        VanillaSortSorter.set_params_to_folder(recording, tmp_path, {"gate_fn": lambda context: []}, verbose=False)
