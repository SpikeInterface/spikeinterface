import pytest
from pathlib import Path

import json
import shutil

import numpy as np

from spikeinterface.core import generate_ground_truth_recording, SortingAnalyzer
from spikeinterface.core.core_tools import SIJsonEncoder

from spikeinterface.core.waveforms_extractor_backwards_compatibility import MockWaveformExtractor
from spikeinterface.core.waveforms_extractor_backwards_compatibility import extract_waveforms as mock_extract_waveforms
from spikeinterface.core.waveforms_extractor_backwards_compatibility import load_waveforms as load_waveforms_backwards

# remove this when WaveformsExtractor will be removed
from spikeinterface.core import extract_waveforms as old_extract_waveforms


def get_dataset():
    recording, sorting = generate_ground_truth_recording(
        durations=[30.0, 20.0],
        sampling_frequency=16000.0,
        num_channels=4,
        num_units=5,
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        generate_unit_locations_kwargs=dict(
            margin_um=5.0,
            minimum_z=5.0,
            maximum_z=20.0,
        ),
        generate_templates_kwargs=dict(
            unit_params=dict(
                alpha=(100.0, 500.0),
            )
        ),
        noise_kwargs=dict(noise_levels=5.0, strategy="tile_pregenerated"),
        seed=2406,
    )
    return recording, sorting


def test_extract_waveforms(create_cache_folder):
    cache_folder = create_cache_folder
    recording, sorting = get_dataset()

    folder = cache_folder / "old_waveforms_extractor"
    if folder.exists():
        shutil.rmtree(folder)

    we_kwargs = dict(sparse=True, max_spikes_per_unit=30)

    we_old = old_extract_waveforms(recording, sorting, folder=folder, **we_kwargs)
    print(we_old)

    folder = cache_folder / "mock_waveforms_extractor"
    if folder.exists():
        shutil.rmtree(folder)

    we_mock = mock_extract_waveforms(recording, sorting, folder=folder, **we_kwargs)
    print(we_mock)

    for we in (we_old, we_mock):

        selected_spikes = we.get_sampled_indices(unit_id=sorting.unit_ids[0])
        # print(selected_spikes.size, selected_spikes.dtype)

        wfs = we.get_waveforms(sorting.unit_ids[0])
        # print(wfs.shape)

        wfs = we.get_waveforms(sorting.unit_ids[0], force_dense=True)
        # print(wfs.shape)

        templates = we.get_all_templates()
        # print(templates.shape)

    # test reading old WaveformsExtractor folder
    folder = cache_folder / "old_waveforms_extractor"
    sorting_analyzer_from_we = load_waveforms_backwards(folder, output="SortingAnalyzer")
    print(sorting_analyzer_from_we)
    mock_loaded_we_old = load_waveforms_backwards(folder, output="MockWaveformExtractor")
    print(mock_loaded_we_old)


@pytest.mark.skip("This test is run locally")
def test_read_old_waveforms_extractor_binary():
    import pandas as pd

    folder = Path(__file__).parent / "old_waveforms"
    mock_waveforms = load_waveforms_backwards(folder / "we-0.100.0")
    sorting_analyzer = load_waveforms_backwards(folder / "we-0.100.0", output="SortingAnalyzer")

    assert isinstance(mock_waveforms, MockWaveformExtractor)
    assert isinstance(sorting_analyzer, SortingAnalyzer)

    for ext_name in sorting_analyzer.get_loaded_extension_names():
        print(ext_name)
        keys = sorting_analyzer.get_extension(ext_name).data.keys()
        print(keys)
        data = sorting_analyzer.get_extension(ext_name).get_data()
        if isinstance(data, np.ndarray):
            print(data.shape)
        elif isinstance(data, pd.DataFrame):
            print(data.columns)
        else:
            print(type(data))


def _create_legacy_we_folder(recording, sorting, folder):
    """Build a minimal legacy WaveformExtractor binary folder on disk.

    Creates just enough structure for ``_read_old_waveforms_extractor_binary``
    to load: the top-level ``params.json``, ``recording_info/``, and a
    serialised sorting object.  No waveform data is written – only the
    skeleton that extension sub-folders hang off of.
    """
    from spikeinterface.core.recording_tools import get_rec_attributes

    folder.mkdir(parents=True, exist_ok=True)

    # params.json
    params = {
        "ms_before": 1.0,
        "ms_after": 2.0,
        "return_scaled": True,
        "dtype": "float32",
    }
    with open(folder / "params.json", "w") as f:
        json.dump(params, f)

    # recording_info/
    rec_info_folder = folder / "recording_info"
    rec_info_folder.mkdir()
    rec_attributes = get_rec_attributes(recording)
    rec_attributes["probegroup"] = None
    with open(rec_info_folder / "recording_attributes.json", "w") as f:
        json.dump(rec_attributes, f, cls=SIJsonEncoder)

    # No need to serialize the sorting on disk – the test passes it
    # directly via the ``sorting`` argument of ``load_waveforms_backwards``.

    return folder


def _add_legacy_quality_metrics(folder, unit_ids):
    """Add a ``quality_metrics/`` sub-folder with deprecated 0.100-era params."""
    import pandas as pd

    ext_folder = folder / "quality_metrics"
    ext_folder.mkdir()

    deprecated_params = {
        "metric_names": [
            "num_spikes",
            "firing_rate",
            "snr",
            "isolation_distance",
            "l_ratio",
        ],
        "qm_params": {
            "num_spikes": {},
            "firing_rate": {},
            "snr": {"peak_sign": "neg", "peak_mode": "extremum"},
            "isolation_distance": {},
            "l_ratio": {},
            "amplitude_cutoff": {"peak_sign": "neg"},
            "amplitude_median": {"peak_sign": "neg"},
        },
        "peak_sign": "neg",
        "seed": None,
        "skip_pc_metrics": False,
    }
    with open(ext_folder / "params.json", "w") as f:
        json.dump(deprecated_params, f)

    metrics_df = pd.DataFrame(index=unit_ids, columns=["num_spikes", "firing_rate", "snr"])
    metrics_df["num_spikes"] = 100
    metrics_df["firing_rate"] = 5.0
    metrics_df["snr"] = 10.0
    metrics_df.to_csv(ext_folder / "metrics.csv")


def _add_legacy_template_metrics(folder, unit_ids):
    """Add a ``template_metrics/`` sub-folder with deprecated 0.100-era params."""
    import pandas as pd

    ext_folder = folder / "template_metrics"
    ext_folder.mkdir()

    deprecated_params = {
        "metric_names": [
            "peak_to_valley",
            "peak_trough_ratio",
            "half_width",
        ],
        "metrics_kwargs": {
            "upsampling_factor": 10,
            "window_slope_ms": 0.7,
        },
    }
    with open(ext_folder / "params.json", "w") as f:
        json.dump(deprecated_params, f)

    metrics_df = pd.DataFrame(index=unit_ids, columns=["peak_to_valley", "peak_trough_ratio", "half_width"])
    metrics_df["peak_to_valley"] = 0.5
    metrics_df["peak_trough_ratio"] = 2.0
    metrics_df["half_width"] = 0.3
    metrics_df.to_csv(ext_folder / "metrics.csv")


def test_load_legacy_we_with_deprecated_metrics(create_cache_folder, tmp_path):
    """Regression test for GH-4508.

    A legacy WaveformExtractor folder whose ``quality_metrics/params.json``
    or ``template_metrics/params.json`` contains deprecated metric names
    (e.g. ``l_ratio``, ``peak_to_valley``) must load without raising a
    ``ValueError``.  The backward-compatibility handler must migrate the
    deprecated names before validation runs.
    """
    recording, sorting = get_dataset()

    we_folder = tmp_path / "legacy_we_deprecated_metrics"
    _create_legacy_we_folder(recording, sorting, we_folder)
    _add_legacy_quality_metrics(we_folder, sorting.unit_ids)
    _add_legacy_template_metrics(we_folder, sorting.unit_ids)

    # This would raise ValueError on main before the fix
    sorting_analyzer = load_waveforms_backwards(we_folder, sorting=sorting, output="SortingAnalyzer")
    assert isinstance(sorting_analyzer, SortingAnalyzer)

    # quality_metrics: deprecated names should be migrated
    qm = sorting_analyzer.get_extension("quality_metrics")
    assert qm is not None
    qm_names = qm.params["metric_names"]
    # The compat handler should have removed the deprecated names
    assert "l_ratio" not in qm_names
    assert "isolation_distance" not in qm_names
    # qm_params should have been renamed to metric_params
    assert "qm_params" not in qm.params
    assert "metric_params" in qm.params

    # template_metrics: deprecated names should be migrated
    tm = sorting_analyzer.get_extension("template_metrics")
    assert tm is not None
    tm_names = tm.params["metric_names"]
    assert "peak_to_valley" not in tm_names
    assert "peak_trough_ratio" not in tm_names
    # metrics_kwargs should have been renamed to metric_params
    assert "metrics_kwargs" not in tm.params
    assert "metric_params" in tm.params


# @pytest.mark.skip("This test is run locally")
# def test_read_old_waveforms_extractor_zarr():
#     import pandas as pd

#     folder = Path(__file__).parent / "old_waveforms"
#     mock_waveforms = load_waveforms_backwards(folder / "we-0.100.0.zarr")
#     sorting_analyzer = load_waveforms_backwards(folder / "we-0.100.0.zarr", output="SortingAnalyzer")

#     assert isinstance(mock_waveforms, MockWaveformExtractor)
#     assert isinstance(sorting_analyzer, SortingAnalyzer)

#     for ext_name in sorting_analyzer.get_loaded_extension_names():
#         print(ext_name)
#         keys = sorting_analyzer.get_extension(ext_name).data.keys()
#         print(keys)
#         data = sorting_analyzer.get_extension(ext_name).get_data()
#         if isinstance(data, np.ndarray):
#             print(data.shape)
#         elif isinstance(data, pd.DataFrame):
#             print(data.columns)
#         else:
#             print(type(data))


if __name__ == "__main__":
    test_read_old_waveforms_extractor_binary()
    # test_read_old_waveforms_extractor_binary()
