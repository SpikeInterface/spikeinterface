import pytest
import numpy as np
from pathlib import Path
import shutil

from spikeinterface import create_sorting_analyzer, generate_ground_truth_recording
from spikeinterface.core.base import spike_peak_dtype
from spikeinterface.core.job_tools import divide_time_series_into_chunks

# from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
    PeakRetriever,
    SpikeRetriever,
    PipelineNode,
    ExtractDenseWaveforms,
    sorting_to_peaks,
)


class AmplitudeExtractionNode(PipelineNode):
    def __init__(self, recording, parents=None, return_output=True, param0=5.5):
        PipelineNode.__init__(self, recording, parents=parents, return_output=return_output)
        self.param0 = param0
        self._dtype = np.dtype([("abs_amplitude", recording.get_dtype())])

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, peaks):
        amps = np.zeros(peaks.size, dtype=self._dtype)
        amps["abs_amplitude"] = np.abs(peaks["amplitude"])
        return amps

    def get_margin(self):
        return 5


class WaveformDenoiser(PipelineNode):
    # waveform smoother
    def __init__(self, recording, return_output=True, parents=None):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)

    def get_dtype(self):
        return np.dtype("float32")

    def compute(self, traces, peaks, waveforms):
        kernel = np.array([0.1, 0.8, 0.1])
        denoised_waveforms = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=waveforms)
        return denoised_waveforms


class WaveformsRootMeanSquare(PipelineNode):
    def __init__(self, recording, return_output=True, parents=None):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)

    def get_dtype(self):
        return np.dtype("float32")

    def compute(self, traces, peaks, waveforms):
        rms_by_channels = np.sum(waveforms**2, axis=1)
        return rms_by_channels


@pytest.fixture(scope="module")
def cache_folder_creation(tmp_path_factory):
    cache_folder = tmp_path_factory.mktemp("cache_folder")
    return cache_folder


def test_run_node_pipeline(cache_folder_creation):
    cache_folder = cache_folder_creation
    recording, sorting = generate_ground_truth_recording(num_channels=10, num_units=10, durations=[10.0])

    # job_kwargs = dict(chunk_duration="0.5s", n_jobs=2, progress_bar=False)
    job_kwargs = dict(chunk_duration="0.5s", n_jobs=1, progress_bar=False)

    spikes = sorting.to_spike_vector()

    # create peaks from spikes
    sorting_analyzer = create_sorting_analyzer(sorting, recording, format="memory")
    sorting_analyzer.compute(["random_spikes", "templates"], **job_kwargs)
    main_channel_indices = sorting_analyzer.get_main_channels(outputs="index", with_dict=False)

    peaks = sorting_to_peaks(sorting, main_channel_indices, spike_peak_dtype)
    # print(peaks.size)

    peak_retriever = PeakRetriever(recording, peaks)
    # this test when no spikes in last chunks
    peak_retriever_few = PeakRetriever(recording, peaks[: peaks.size // 2])

    # channel index is from template
    spike_retriever_T = SpikeRetriever(sorting, recording, channel_from_template=True)
    # channel index is per spike
    spike_retriever_S = SpikeRetriever(
        sorting,
        recording,
        channel_from_template=False,
        radius_um=50,
        peak_sign="neg",
    )

    # test with 3 different first nodes
    for loop, peak_source in enumerate((peak_retriever, peak_retriever_few, spike_retriever_T, spike_retriever_S)):
        # one step only : squeeze output
        nodes = [
            peak_source,
            AmplitudeExtractionNode(recording, parents=[peak_source], param0=6.6),
        ]
        step_one = run_node_pipeline(recording, nodes, job_kwargs, squeeze_output=True)
        if loop == 0:
            assert np.allclose(np.abs(peaks["amplitude"]), step_one["abs_amplitude"])

        # 3 nodes two have outputs
        ms_before = 0.5
        ms_after = 1.0
        peak_retriever = PeakRetriever(recording, peaks)
        dense_waveforms = ExtractDenseWaveforms(
            recording, parents=[peak_source], ms_before=ms_before, ms_after=ms_after, return_output=False
        )
        waveform_denoiser = WaveformDenoiser(recording, parents=[peak_source, dense_waveforms], return_output=False)
        amplitue_extraction = AmplitudeExtractionNode(recording, parents=[peak_source], param0=6.6, return_output=True)
        waveforms_rms = WaveformsRootMeanSquare(recording, parents=[peak_source, dense_waveforms], return_output=True)
        denoised_waveforms_rms = WaveformsRootMeanSquare(
            recording, parents=[peak_source, waveform_denoiser], return_output=True
        )

        nodes = [
            peak_source,
            dense_waveforms,
            waveform_denoiser,
            amplitue_extraction,
            waveforms_rms,
            denoised_waveforms_rms,
        ]

        # gather memory mode
        output = run_node_pipeline(recording, nodes, job_kwargs, gather_mode="memory")
        amplitudes, waveforms_rms, denoised_waveforms_rms = output

        num_peaks = peaks.shape[0]
        num_channels = recording.get_num_channels()
        if peak_source != peak_retriever_few:
            assert waveforms_rms.shape[0] == num_peaks
        assert waveforms_rms.shape[1] == num_channels

        if peak_source != peak_retriever_few:
            assert waveforms_rms.shape[0] == num_peaks
        assert waveforms_rms.shape[1] == num_channels

        # gather npy mode
        folder = cache_folder / f"pipeline_folder_{loop}"
        if folder.is_dir():
            shutil.rmtree(folder)
        output = run_node_pipeline(
            recording,
            nodes,
            job_kwargs,
            gather_mode="npy",
            folder=folder,
            names=["amplitudes", "waveforms_rms", "denoised_waveforms_rms"],
        )
        amplitudes2, waveforms_rms2, denoised_waveforms_rms2 = output

        amplitudes_file = folder / "amplitudes.npy"
        assert amplitudes_file.is_file()
        amplitudes3 = np.load(amplitudes_file)
        assert np.array_equal(amplitudes, amplitudes2)
        assert np.array_equal(amplitudes2, amplitudes3)

        waveforms_rms_file = folder / "waveforms_rms.npy"
        assert waveforms_rms_file.is_file()
        waveforms_rms3 = np.load(waveforms_rms_file)
        assert np.array_equal(waveforms_rms, waveforms_rms2)
        assert np.array_equal(waveforms_rms2, waveforms_rms3)

        denoised_waveforms_rms_file = folder / "denoised_waveforms_rms.npy"
        assert denoised_waveforms_rms_file.is_file()
        denoised_waveforms_rms3 = np.load(denoised_waveforms_rms_file)
        assert np.array_equal(denoised_waveforms_rms, denoised_waveforms_rms2)
        assert np.array_equal(denoised_waveforms_rms2, denoised_waveforms_rms3)

        # gather zarr mode
        import zarr

        zarr_folder = cache_folder / f"pipeline_folder_{loop}.zarr"
        if zarr_folder.is_dir():
            shutil.rmtree(zarr_folder)
        output = run_node_pipeline(
            recording,
            nodes,
            job_kwargs,
            gather_mode="zarr",
            folder=zarr_folder,
            names=["amplitudes", "waveforms_rms", "denoised_waveforms_rms"],
        )
        amplitudes_z, waveforms_rms_z, denoised_waveforms_rms_z = output

        # values must match the memory gather
        assert np.array_equal(amplitudes, amplitudes_z[:])
        assert np.array_equal(waveforms_rms, waveforms_rms_z[:])
        assert np.array_equal(denoised_waveforms_rms, denoised_waveforms_rms_z[:])

        # arrays must be persisted on disk and re-openable
        zarr_root = zarr.open(str(zarr_folder), mode="r")
        for name in ("amplitudes", "waveforms_rms", "denoised_waveforms_rms"):
            assert name in zarr_root
        assert np.array_equal(amplitudes, zarr_root["amplitudes"][:])
        assert np.array_equal(waveforms_rms, zarr_root["waveforms_rms"][:])
        assert np.array_equal(denoised_waveforms_rms, zarr_root["denoised_waveforms_rms"][:])

        # gather npy mode with an explicit list of file paths (final location)
        npy_files_folder = cache_folder / f"pipeline_npy_files_{loop}"
        if npy_files_folder.is_dir():
            shutil.rmtree(npy_files_folder)
        npy_files = [
            npy_files_folder / "amp.npy",
            npy_files_folder / "sub" / "rms.npy",
            npy_files_folder / "denoised_rms.npy",
        ]
        output = run_node_pipeline(
            recording,
            nodes,
            job_kwargs,
            gather_mode="npy",
            folder=npy_files,
        )
        amplitudes_f, waveforms_rms_f, denoised_waveforms_rms_f = output
        for npy_file in npy_files:
            assert npy_file.is_file()
        assert np.array_equal(amplitudes, amplitudes_f)
        assert np.array_equal(waveforms_rms, waveforms_rms_f)
        assert np.array_equal(denoised_waveforms_rms, denoised_waveforms_rms_f)

        # gather zarr mode with an explicit list of dataset paths, created on the fly
        # inside an existing store (final location, e.g. an analyzer extension group)
        datasets_store = cache_folder / f"pipeline_zarr_datasets_{loop}.zarr"
        if datasets_store.is_dir():
            shutil.rmtree(datasets_store)
        # pre-existing store that must not be wiped
        root = zarr.open(str(datasets_store), mode="w")
        root.attrs["preexisting"] = True
        dataset_paths = [
            datasets_store / "extensions" / "amplitudes",
            datasets_store / "extensions" / "waveforms_rms",
            datasets_store / "extensions" / "denoised_waveforms_rms",
        ]
        output = run_node_pipeline(
            recording,
            nodes,
            job_kwargs,
            gather_mode="zarr",
            folder=dataset_paths,
        )
        amplitudes_d, waveforms_rms_d, denoised_waveforms_rms_d = output
        assert np.array_equal(amplitudes, amplitudes_d[:])
        assert np.array_equal(waveforms_rms, waveforms_rms_d[:])
        assert np.array_equal(denoised_waveforms_rms, denoised_waveforms_rms_d[:])
        # data must be persisted at the passed final location and the store not wiped
        root_reopen = zarr.open(str(datasets_store), mode="r")
        assert root_reopen.attrs.get("preexisting", False)
        assert np.array_equal(amplitudes, root_reopen["extensions"]["amplitudes"][:])
        assert np.array_equal(waveforms_rms, root_reopen["extensions"]["waveforms_rms"][:])
        assert np.array_equal(denoised_waveforms_rms, root_reopen["extensions"]["denoised_waveforms_rms"][:])

        # Test pickle mechanism
        for node in nodes:
            import pickle

            pickled_node = pickle.dumps(node)
            unpickled_node = pickle.loads(pickled_node)


def test_gather_to_zarr_chunking(tmp_path):
    # the zarr chunk size along the first axis must be picked from a byte target (not from the
    # size of the first gathered buffer), so it stays sensible for billions of spikes and never
    # collapses to 1 row per chunk on a quiet first chunk.
    recording, sorting = generate_ground_truth_recording(num_channels=8, num_units=5, durations=[20.0], seed=7)

    # small chunks + n_jobs>1 so the first non-empty buffer is small (would give chunk0==1 with the
    # old first-buffer heuristic)
    job_kwargs = dict(chunk_duration="0.1s", n_jobs=2, progress_bar=False)

    spikes = sorting.to_spike_vector()
    peaks = np.zeros(spikes.size, dtype=spike_peak_dtype)
    peaks["sample_index"] = spikes["sample_index"]
    peaks["segment_index"] = spikes["segment_index"]

    peak_retriever = PeakRetriever(recording, peaks)
    ms_before, ms_after = 0.5, 1.0
    dense_waveforms = ExtractDenseWaveforms(
        recording, parents=[peak_retriever], ms_before=ms_before, ms_after=ms_after, return_output=True
    )
    nodes = [peak_retriever, dense_waveforms]

    # default byte target (10 MiB)
    target_bytes = 10 * 1024 * 1024
    waveforms = run_node_pipeline(
        recording, nodes, job_kwargs, gather_mode="zarr", folder=tmp_path / "wfs.zarr", names=["waveforms"]
    )
    nbefore_after = dense_waveforms.nbefore + dense_waveforms.nafter
    row_nbytes = nbefore_after * recording.get_num_channels() * np.dtype("float32").itemsize
    expected_chunk0 = max(1, target_bytes // row_nbytes)
    assert waveforms.chunks[0] == expected_chunk0
    assert waveforms.chunks[0] > 1
    assert waveforms.chunks[1:] == waveforms.shape[1:]

    # explicit override is respected
    waveforms2 = run_node_pipeline(
        recording,
        nodes,
        job_kwargs,
        gather_mode="zarr",
        folder=tmp_path / "wfs2.zarr",
        names=["waveforms"],
        gather_kwargs={"zarr_chunk_size": 1234},
    )
    assert waveforms2.chunks[0] == 1234
    assert np.array_equal(waveforms[:], waveforms2[:])


def test_skip_after_n_peaks_and_recording_slices():
    recording, sorting = generate_ground_truth_recording(num_channels=10, num_units=10, durations=[10.0], seed=2205)

    # job_kwargs = dict(chunk_duration="0.5s", n_jobs=2, progress_bar=False)
    job_kwargs = dict(chunk_duration="0.5s", n_jobs=1, progress_bar=False)

    spikes = sorting.to_spike_vector()

    # create peaks from spikes
    sorting_analyzer = create_sorting_analyzer(sorting, recording, format="memory")
    sorting_analyzer.compute(["random_spikes", "templates"], **job_kwargs)
    main_channel_indices = sorting_analyzer.get_main_channels(outputs="index", with_dict=False)

    peaks = sorting_to_peaks(sorting, main_channel_indices, spike_peak_dtype)
    # print(peaks.size)

    node0 = PeakRetriever(recording, peaks)
    node1 = AmplitudeExtractionNode(recording, parents=[node0], param0=6.6, return_output=True)
    nodes = [node0, node1]

    # skip
    skip_after_n_peaks = 30
    some_amplitudes = run_node_pipeline(
        recording, nodes, job_kwargs, gather_mode="memory", skip_after_n_peaks=skip_after_n_peaks
    )
    assert some_amplitudes.size >= skip_after_n_peaks
    assert some_amplitudes.size < spikes.size

    # slices : 1 every 4
    recording_slices = divide_time_series_into_chunks(recording, 10_000)
    recording_slices = recording_slices[::4]
    some_amplitudes = run_node_pipeline(recording, nodes, job_kwargs, gather_mode="memory", slices=recording_slices)
    tolerance = 1.2
    assert some_amplitudes.size < (spikes.size // 4) * tolerance


# the following is for testing locally with python or ipython. It is not used in ci or with pytest.
if __name__ == "__main__":
    # folder = Path("./cache_folder/core")
    # test_run_node_pipeline(folder)

    test_skip_after_n_peaks_and_recording_slices()
