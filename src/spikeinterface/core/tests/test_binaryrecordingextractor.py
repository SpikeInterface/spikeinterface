import pytest
import numpy as np
from pathlib import Path
import sys

from spikeinterface.core import BinaryRecordingExtractor
from spikeinterface.core.numpyextractors import NumpyRecording
from spikeinterface.core.core_tools import measure_memory_allocation
from spikeinterface.core.generate import GeneratorRecording

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_BinaryRecordingExtractor():
    num_seg = 2
    num_channels = 3
    num_samples = 30
    sampling_frequency = 10000
    dtype = "int16"

    file_paths = [cache_folder / f"test_BinaryRecordingExtractor_{i}.raw" for i in range(num_seg)]
    for i in range(num_seg):
        np.memmap(file_paths[i], dtype=dtype, mode="w+", shape=(num_samples, num_channels))

    rec = BinaryRecordingExtractor(
        file_paths=file_paths,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype=dtype,
    )

    file_paths = [cache_folder / f"test_BinaryRecordingExtractor_copied_{i}.raw" for i in range(num_seg)]
    BinaryRecordingExtractor.write_recording(rec, file_paths)

    file_paths = [cache_folder / f"test_BinaryRecordingExtractor_{i}.raw" for i in range(num_seg)]
    assert (cache_folder / "test_BinaryRecordingExtractor_copied_0.raw").is_file()


def test_round_trip(tmp_path):
    num_channels = 10
    num_samples = 50

    traces = np.arange(num_channels * num_samples, dtype="int16").reshape(num_samples, num_channels)
    sampling_frequency = 30_000.0
    recording = NumpyRecording(traces_list=[traces], sampling_frequency=sampling_frequency)

    file_path = tmp_path / "test_BinaryRecordingExtractor.raw"
    dtype = recording.get_dtype()
    BinaryRecordingExtractor.write_recording(recording=recording, dtype=dtype, file_paths=file_path)

    sampling_frequency = recording.get_sampling_frequency()
    num_channels = recording.get_num_channels()
    binary_recorder = BinaryRecordingExtractor(
        file_paths=file_path,
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype=dtype,
    )

    # Test for full traces
    assert np.allclose(recording.get_traces(), binary_recorder.get_traces())

    # Ttest for a sub-set of the traces
    start_frame = 20
    end_frame = 40
    smaller_traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    binary_smaller_traces = binary_recorder.get_traces(start_frame=start_frame, end_frame=end_frame)

    np.allclose(smaller_traces, binary_smaller_traces)


@pytest.fixture(scope="module")
def folder_with_binary_files(tmpdir_factory):
    tmp_path = Path(tmpdir_factory.mktemp("spike_interface_test"))
    folder = tmp_path / "test_binary_recording"
    num_channels = 32
    sampling_frequency = 30_000.0
    dtype = "float32"
    recording = GeneratorRecording(
        durations=[1.0],
        sampling_frequency=sampling_frequency,
        num_channels=num_channels,
        dtype=dtype,
    )
    dtype = recording.get_dtype()
    recording.save(folder=folder, overwrite=True)

    return folder


def test_sequential_reading_of_small_traces(folder_with_binary_files):
    # Test that memmap is readed correctly when pointing to specific frames
    folder = folder_with_binary_files
    num_channels = 32
    sampling_frequency = 30_000.0
    dtype = "float32"

    file_paths = [folder / "traces_cached_seg0.raw"]
    recording = BinaryRecordingExtractor(
        num_chan=num_channels,
        file_paths=file_paths,
        sampling_frequency=sampling_frequency,
        dtype=dtype,
    )

    full_traces = recording.get_traces()

    # Test for a sub-set of the traces
    start_frame = 10
    end_frame = 15
    small_traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    expected_traces = full_traces[start_frame:end_frame, :]
    assert np.allclose(small_traces, expected_traces)

    # Test for a sub-set of the traces
    start_frame = 1000
    end_frame = 1100
    small_traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    expected_traces = full_traces[start_frame:end_frame, :]
    assert np.allclose(small_traces, expected_traces)

    # Test for a sub-set of the traces
    start_frame = 10_000
    end_frame = 11_000
    small_traces = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    expected_traces = full_traces[start_frame:end_frame, :]
    assert np.allclose(small_traces, expected_traces)


def test_memory_effcienty(folder_with_binary_files):
    "This test that memory is freed afte reading the traces"
    folder = folder_with_binary_files
    num_channels = 32
    sampling_frequency = 30_000.0
    dtype = "float32"

    file_paths = [folder / "traces_cached_seg0.raw"]
    recording = BinaryRecordingExtractor(
        num_chan=num_channels,
        file_paths=file_paths,
        sampling_frequency=sampling_frequency,
        dtype=dtype,
    )

    memory_before_traces_bytes = measure_memory_allocation()
    traces = recording.get_traces(start_frame=1000, end_frame=10_000)
    memory_after_traces_bytes = measure_memory_allocation()
    traces_size_bytes = traces.nbytes

    expected_memory_usage = memory_before_traces_bytes + traces_size_bytes
    expected_memory_usage_GiB = expected_memory_usage / 1024**3
    memory_after_traces_bytes_GiB = memory_after_traces_bytes / 1024**3

    ratio = memory_after_traces_bytes_GiB / expected_memory_usage_GiB

    assertion_msg = (
        f"Peak memory {memory_after_traces_bytes_GiB} GiB usage is {ratio:.2f} times"
        f"the expected memory usage of {expected_memory_usage_GiB} GiB."
    )
    assert ratio <= 1.05, assertion_msg

    assert memory_after_traces_bytes_GiB == pytest.approx(expected_memory_usage_GiB, rel=0.1)


def measure_peak_memory_usage():
    """
    Measure the peak memory usage in bytes for the current process.

    The `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` command is used to get the peak memory usage.
    The `ru_maxrss` attribute represents the maximum resident set size used (in kilobytes on Linux and bytes on MacOS),
    which is the maximum memory used by the process since it was started.

    This function only works on Unix systems (including Linux and MacOS).

    Returns
    -------
    int
        Peak memory usage in bytes.

    Raises
    ------
    NotImplementedError
        If the function is called on a Windows system.
    """

    if sys.platform == "win32":
        raise NotImplementedError("Resource module not available on Windows")

    import resource

    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # If ru_maxrss returns memory in kilobytes (like on Linux), convert to bytes
    if hasattr(resource, "RLIMIT_AS"):
        mem_usage = mem_usage * 1024

    return mem_usage


@pytest.mark.skipif(sys.platform == "win32", reason="resource module not available on Windows")
def test_peak_memory_usage(folder_with_binary_files):
    "This tests that there are no spikes in memory usage when reading traces."
    folder = folder_with_binary_files
    num_channels = 32
    sampling_frequency = 30_000.0
    dtype = "float32"

    file_paths = [folder / "traces_cached_seg0.raw"]
    recording = BinaryRecordingExtractor(
        num_chan=num_channels,
        file_paths=file_paths,
        sampling_frequency=sampling_frequency,
        dtype=dtype,
    )

    memory_before_traces_bytes = measure_memory_allocation()
    traces = recording.get_traces(start_frame=1000, end_frame=2000)
    traces_size_bytes = traces.nbytes

    expected_memory_usage = memory_before_traces_bytes + traces_size_bytes
    peak_memory_MiB = measure_peak_memory_usage() / 1024**2
    expected_memory_usage_MiB = expected_memory_usage / 1024**2
    ratio = peak_memory_MiB / expected_memory_usage_MiB
    assertion_msg = (
        f"Peak memory {peak_memory_MiB} MiB usage is {ratio:.2f} times"
        f"the expected memory usage of {expected_memory_usage_MiB} MiB."
    )
    assert ratio <= 1.05, assertion_msg


if __name__ == "__main__":
    test_BinaryRecordingExtractor()
