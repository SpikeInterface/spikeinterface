import pytest
import psutil

import numpy as np

from spikeinterface.core import load_extractor, extract_waveforms
from spikeinterface.core.generate import (generate_recording, NoiseGeneratorRecording, generate_recording_by_size, 
                                          InjectTemplatesRecording, generate_single_fake_waveform, generate_templates,
                                          generate_channel_locations, generate_unit_locations,
                                          toy_example)


from spikeinterface.core.core_tools import convert_bytes_to_str

from spikeinterface.core.testing import check_recordings_equal

strategy_list = ["tile_pregenerated", "on_the_fly"]

def measure_memory_allocation(measure_in_process: bool = True) -> float:
    """
    A local utility to measure memory allocation at a specific point in time.
    Can measure either the process resident memory or system wide memory available

    Uses psutil package.

    Parameters
    ----------
    measure_in_process : bool, True by default
        Mesure memory allocation in the current process only, if false then measures at the system
        level.
    """

    if measure_in_process:
        process = psutil.Process()
        memory = process.memory_info().rss
    else:
        mem_info = psutil.virtual_memory()
        memory = mem_info.total - mem_info.available

    return memory


@pytest.mark.parametrize("strategy", strategy_list)
def test_noise_generator_memory(strategy):
    # Test that get_traces does not consume more memory than allocated.

    bytes_to_MiB_factor = 1024**2
    relative_tolerance = 0.05  # relative tolerance of 5 per cent

    sampling_frequency = 30000  # Hz
    durations = [2.0]
    dtype = np.dtype("float32")
    num_channels = 384
    seed = 0

    num_samples = int(durations[0] * sampling_frequency)
    # Around 100 MiB  4 bytes per sample * 384 channels * 30000  samples * 2 seconds duration
    expected_trace_size_MiB = dtype.itemsize * num_channels * num_samples / bytes_to_MiB_factor

    initial_memory_MiB = measure_memory_allocation() / bytes_to_MiB_factor
    lazy_recording = NoiseGeneratorRecording(
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        durations=durations,        
        dtype=dtype,
        seed=seed,
        strategy=strategy,
    )

    memory_after_instanciation_MiB = measure_memory_allocation() / bytes_to_MiB_factor
    expected_memory_usage_MiB = initial_memory_MiB
    if strategy == "tile_pregenerated":
        expected_memory_usage_MiB += 50  # 50 MiB for the white noise generator

    ratio = memory_after_instanciation_MiB * 1.0 / expected_memory_usage_MiB
    assertion_msg = (
        f"Memory after instantation is {memory_after_instanciation_MiB} MiB and is {ratio:.2f} times"
        f"the expected memory usage of {expected_memory_usage_MiB} MiB."
    )
    assert ratio <= 1.0 + relative_tolerance, assertion_msg

    traces = lazy_recording.get_traces()
    expected_traces_shape = (int(durations[0] * sampling_frequency), num_channels)

    traces_size_MiB = traces.nbytes / bytes_to_MiB_factor
    assert traces_size_MiB == expected_trace_size_MiB
    assert traces.shape == expected_traces_shape

    memory_after_traces_MiB = measure_memory_allocation() / bytes_to_MiB_factor

    expected_memory_usage_MiB = memory_after_instanciation_MiB + traces_size_MiB
    ratio = memory_after_traces_MiB * 1.0 / expected_memory_usage_MiB
    assertion_msg = (
        f"Memory after loading traces is {memory_after_traces_MiB} MiB and is {ratio:.2f} times"
        f"the expected memory usage of {expected_memory_usage_MiB} MiB."
    )
    assert ratio <= 1.0 + relative_tolerance, assertion_msg


def test_noise_generator_under_giga():
    # Test that the recording has the correct size in memory when calling smaller than 1 GiB
    # This is a week test that only measures the size of the traces and not the  memory used
    recording = generate_recording_by_size(full_traces_size_GiB=0.5)
    recording_total_memory = convert_bytes_to_str(recording.get_memory_size())
    assert recording_total_memory == "512.00 MiB"

    recording = generate_recording_by_size(full_traces_size_GiB=0.3)
    recording_total_memory = convert_bytes_to_str(recording.get_memory_size())
    assert recording_total_memory == "307.20 MiB"

    recording = generate_recording_by_size(full_traces_size_GiB=0.1)
    recording_total_memory = convert_bytes_to_str(recording.get_memory_size())
    assert recording_total_memory == "102.40 MiB"


@pytest.mark.parametrize("strategy", strategy_list)
def test_noise_generator_correct_shape(strategy):
    # Test that the recording has the correct size in shape
    sampling_frequency = 30000  # Hz
    durations = [1.0]
    dtype = np.dtype("float32")
    num_channels = 2
    seed = 0

    lazy_recording = NoiseGeneratorRecording(
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        durations=durations,
       dtype=dtype,
        seed=seed,
        strategy=strategy,
    )

    num_frames = lazy_recording.get_num_frames(segment_index=0)
    assert num_frames == sampling_frequency * durations[0]

    traces = lazy_recording.get_traces()

    assert traces.shape == (num_frames, num_channels)


@pytest.mark.parametrize("strategy", strategy_list)
@pytest.mark.parametrize(
    "start_frame, end_frame",
    [
        (0, None),
        (0, 80),
        (20_000, 30_000),
        (0, 30_000),
        (15_000, 30_0000),
    ],
)
def test_noise_generator_consistency_across_calls(strategy, start_frame, end_frame):
    # Calling the get_traces twice should return the same result
    sampling_frequency = 30000  # Hz
    durations = [2.0]
    dtype = np.dtype("float32")
    num_channels = 2
    seed = 0

    lazy_recording = NoiseGeneratorRecording(
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        durations=durations,        
        dtype=dtype,
        seed=seed,
        strategy=strategy,
    )

    traces = lazy_recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    same_traces = lazy_recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    assert np.allclose(traces, same_traces)


@pytest.mark.parametrize("strategy", strategy_list)
@pytest.mark.parametrize(
    "start_frame, end_frame, extra_samples",
    [
        (0, 1000, 10),
        (0, 20_000, 10_000),
        (1_000, 2_000, 300),
        (250, 750, 800),
        (10_000, 25_000, 3_000),
        (0, 60_000, 10_000),
    ],
)
def test_noise_generator_consistency_across_traces(strategy, start_frame, end_frame, extra_samples):
    # Test that the generated traces behave like true arrays. Calling a larger array and then slicing it should
    # give the same result as calling the slice directly
    sampling_frequency = 30000  # Hz
    durations = [10.0]
    dtype = np.dtype("float32")
    num_channels = 2
    seed = start_frame + end_frame + extra_samples  # To make sure that the seed is different for each test

    lazy_recording = NoiseGeneratorRecording(
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        durations=durations,
        dtype=dtype,
        seed=seed,
        strategy=strategy,
    )

    traces = lazy_recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    end_frame_larger_array = end_frame + extra_samples
    larger_traces = lazy_recording.get_traces(start_frame=start_frame, end_frame=end_frame_larger_array)
    equivalent_trace_from_larger_traces = larger_traces[:-extra_samples, :]  # Remove the extra samples
    assert np.allclose(traces, equivalent_trace_from_larger_traces)


@pytest.mark.parametrize("strategy", strategy_list)
@pytest.mark.parametrize("seed", [None, 42])
def test_noise_generator_consistency_after_dump(strategy, seed):
    # test same noise after dump even with seed=None
    rec0 = NoiseGeneratorRecording(
        num_channels=2,
        sampling_frequency=30000.,
        durations=[2.0],
        dtype="float32",
        seed=seed,
        strategy=strategy,
    )
    traces0 = rec0.get_traces()
    
    rec1 = load_extractor(rec0.to_dict())
    traces1 = rec1.get_traces()

    assert np.allclose(traces0, traces1)



def test_generate_recording():
    # check the high level function
    rec = generate_recording(mode="lazy")
    rec = generate_recording(mode="legacy")


def test_generate_single_fake_waveform():
    sampling_frequency = 30000.
    ms_before = 1.
    ms_after = 3.
    wf = generate_single_fake_waveform(ms_before=ms_before, ms_after=ms_after, sampling_frequency=sampling_frequency)

    # import matplotlib.pyplot as plt
    # times = np.arange(wf.size) / sampling_frequency * 1000 - ms_before
    # fig, ax = plt.subplots()
    # ax.plot(times, wf)
    # ax.axvline(0)
    # plt.show()

def test_generate_templates():

    rng = np.random.default_rng(seed=0)

    num_chans = 12
    num_columns = 1
    num_units = 10
    margin_um= 15.
    channel_locations = generate_channel_locations(num_chans, num_columns, 20.)
    unit_locations = generate_unit_locations(num_units, channel_locations, margin_um, rng)

    
    sampling_frequency = 30000.
    ms_before = 1.
    ms_after = 3.
    templates = generate_templates(channel_locations, unit_locations, sampling_frequency, ms_before, ms_after,
            upsample_factor=None,
            seed=42,
            dtype="float32",
        )
    assert templates.ndim == 3
    assert templates.shape[2] == num_chans
    assert templates.shape[0] == num_units


    # templates = generate_templates(channel_locations, unit_locations, sampling_frequency, ms_before, ms_after,
    #         upsample_factor=3,
    #         seed=42,
    #         dtype="float32",
    #     )
    # assert templates.ndim == 4
    # assert templates.shape[2] == num_chans
    # assert templates.shape[0] == num_units
    # assert templates.shape[3] == 3


    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # for u in range(num_units):
    #     ax.plot(templates[u, :, ].T.flatten())
    # for f in range(templates.shape[3]):
    #     ax.plot(templates[0, :, :, f].T.flatten())
    # plt.show()


def test_inject_templates():
    num_channels = 4
    durations = [5.0, 2.5]

    recording = generate_recording(num_channels=4, durations=durations, mode="lazy")
    recording.annotate(is_filtered=True)
    # recording = recording.save(folder=cache_folder / "recording")

    # npz_filename = cache_folder / "sorting.npz"
    # sorting_npz = create_sorting_npz(num_seg=2, file_path=npz_filename)
    # sorting = NpzSortingExtractor(npz_filename)

    # wvf_extractor = extract_waveforms(recording, sorting, mode="memory", ms_before=3.0, ms_after=3.0)
    # templates = wvf_extractor.get_all_templates()
    # templates[:, 0] = templates[:, -1] = 0.0  # Go around the check for the edge, this is just testing.

    # parent_recording = None
    recording_template_injected = InjectTemplatesRecording(
        sorting,
        templates,
        nbefore=wvf_extractor.nbefore,
        num_samples=[recording.get_num_frames(seg_ind) for seg_ind in range(recording.get_num_segments())],
    )

    assert recording_template_injected.get_traces(end_frame=600, segment_index=0).shape == (600, 4)
    assert recording_template_injected.get_traces(start_frame=100, end_frame=600, segment_index=1).shape == (500, 4)
    assert recording_template_injected.get_traces(
        start_frame=recording.get_num_frames(0) - 200, segment_index=0
    ).shape == (200, 4)

    # parent_recording != None
    recording_template_injected = InjectTemplatesRecording(
        sorting, templates, nbefore=wvf_extractor.nbefore, parent_recording=recording
    )

    assert recording_template_injected.get_traces(end_frame=600, segment_index=0).shape == (600, 4)
    assert recording_template_injected.get_traces(start_frame=100, end_frame=600, segment_index=1).shape == (500, 4)
    assert recording_template_injected.get_traces(
        start_frame=recording.get_num_frames(0) - 200, segment_index=0
    ).shape == (200, 4)

    # Check dumpability
    saved_loaded = load_extractor(recording_template_injected.to_dict())
    check_recordings_equal(recording_template_injected, saved_loaded, return_scaled=False)

    # saved_1job = recording_template_injected.save(folder=cache_folder / "1job")
    # saved_2job = recording_template_injected.save(folder=cache_folder / "2job", n_jobs=2, chunk_duration="1s")
    # check_recordings_equal(recording_template_injected, saved_1job, return_scaled=False)
    # check_recordings_equal(recording_template_injected, saved_2job, return_scaled=False)

def test_toy_example():
    rec, sorting = toy_example(num_segments=2, num_units=10)
    assert rec.get_num_segments() == 2
    assert sorting.get_num_segments() == 2
    assert sorting.get_num_units() == 10

    # rec, sorting = toy_example(num_segments=1, num_channels=16, num_columns=2)
    # assert rec.get_num_segments() == 1
    # assert sorting.get_num_segments() == 1
    # print(rec)
    # print(sorting)

    probe = rec.get_probe()
    # print(probe)


if __name__ == "__main__":
    # strategy = "tile_pregenerated"
    # strategy = "on_the_fly"
    # test_noise_generator_memory(strategy)
    # test_noise_generator_under_giga()
    # test_noise_generator_correct_shape(strategy)
    # test_noise_generator_consistency_across_calls(strategy, 0, 5)
    # test_noise_generator_consistency_across_traces(strategy, 0, 1000, 10)
    # test_noise_generator_consistency_after_dump(strategy)
    # test_generate_recording()
    # test_generate_single_fake_waveform()
    # test_generate_templates()
    # test_inject_templates()

    test_toy_example()
