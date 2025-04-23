import pytest

from spikeinterface.preprocessing import bandpass_filter, decimate
from spikeinterface.exporters import export_to_ibl_gui

from spikeinterface.exporters.tests.common import (
    make_sorting_analyzer,
    sorting_analyzer_sparse_for_export,
)

required_output_files = [
    "spikes.times.npy",
    "spikes.clusters.npy",
    "spikes.depths.npy",
    "spikes.amps.npy",
    "clusters.waveforms.npy",
    "clusters.peakToTrough.npy",
    "clusters.channels.npy",
    "clusters.metrics.csv",
    "channels.localCoordinates.npy",
    "channels.rawInd.npy",
]
ap_output_files = ["_iblqc_ephysTimeRmsAP.rms.npy", "_iblqc_ephysTimeRmsAP.timestamps.npy"]
lfp_output_files = [
    "_iblqc_ephysTimeRmsLF.rms.npy",
    "_iblqc_ephysTimeRmsLF.timestamps.npy",
    "_iblqc_ephysSpectralDensityLF.power.npy",
    "_iblqc_ephysSpectralDensityLF.freqs.npy",
]

good_units_query = "amplitude_median < -30"


def test_export_ap_to_ibl(sorting_analyzer_sparse_for_export, create_cache_folder):
    cache_folder = create_cache_folder
    output_folder = cache_folder / "ibl_ap_output"

    sorting_analyzer = sorting_analyzer_sparse_for_export
    # AP, but no LFP
    export_to_ibl_gui(
        sorting_analyzer,
        output_folder,
        # good_units_query=good_units_query,
        verbose=True,
        n_jobs=-1,
    )
    for f in required_output_files:
        assert (output_folder / f).exists(), f"Missing file: {f}"
    for f in ap_output_files:
        assert (output_folder / f).exists(), f"Missing file: {f}"
    for f in lfp_output_files:
        assert not (output_folder / f).exists(), f"Unexpected file: {f}"


def test_export_recordingless_to_ibl(sorting_analyzer_sparse_for_export, create_cache_folder):
    cache_folder = create_cache_folder
    output_folder = cache_folder / "ibl_recordingless_output"

    sorting_analyzer = sorting_analyzer_sparse_for_export
    recording = sorting_analyzer.recording
    sorting_analyzer._recording = None

    # AP, but no LFP
    export_to_ibl_gui(sorting_analyzer_sparse_for_export, output_folder, good_units_query=good_units_query, n_jobs=-1)
    for f in required_output_files:
        assert (output_folder / f).exists(), f"Missing file: {f}"
    for f in ap_output_files:
        assert not (output_folder / f).exists(), f"Missing file: {f}"
    for f in lfp_output_files:
        assert not (output_folder / f).exists(), f"Unexpected file: {f}"

    sorting_analyzer._recording = recording


def test_export_lfp_to_ibl(sorting_analyzer_sparse_for_export, create_cache_folder):
    cache_folder = create_cache_folder
    output_folder = cache_folder / "ibl_lfp_output"

    sorting_analyzer = sorting_analyzer_sparse_for_export
    recording = sorting_analyzer.recording
    recording_lfp = bandpass_filter(recording, freq_min=0.5, freq_max=300)
    recording_lfp = decimate(recording_lfp, 10)
    # LFP, but no AP
    export_to_ibl_gui(
        sorting_analyzer, output_folder, lfp_recording=recording_lfp, good_units_query=good_units_query, n_jobs=-1
    )
    for f in required_output_files:
        assert (output_folder / f).exists(), f"Missing file: {f}"
    for f in ap_output_files:
        assert (output_folder / f).exists(), f"Unexpected file: {f}"
    for f in lfp_output_files:
        assert (output_folder / f).exists(), f"Missing file: {f}"


def test_missing_info(sorting_analyzer_sparse_for_export, create_cache_folder):
    cache_folder = create_cache_folder
    output_folder = cache_folder / "ibl_missing_info_output"

    sorting_analyzer = sorting_analyzer_sparse_for_export

    # missing metrics
    good_units_query = "rp_violations < 0.2"

    with pytest.raises(ValueError, match="Missing required quality metrics"):
        export_to_ibl_gui(sorting_analyzer, output_folder, good_units_query=good_units_query, n_jobs=-1)

    sorting_analyzer.delete_extension("spike_amplitudes")

    with pytest.raises(ValueError, match="Missing required extension"):
        export_to_ibl_gui(sorting_analyzer, output_folder, n_jobs=-1)


if __name__ == "__main__":
    sorting_analyzer = make_sorting_analyzer(sparse=True)
    test_export_ap_to_ibl(sorting_analyzer)
