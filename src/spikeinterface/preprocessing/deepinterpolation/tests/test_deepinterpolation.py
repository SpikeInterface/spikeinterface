import pytest
import numpy as np
from pathlib import Path
from packaging.version import parse
from warnings import warn

import probeinterface
from spikeinterface import generate_recording, append_recordings
from spikeinterface.preprocessing import depth_order, zscore

from spikeinterface.preprocessing.deepinterpolation import train_deepinterpolation, deepinterpolate
from spikeinterface.preprocessing.deepinterpolation.train import train_deepinterpolation_process


try:
    import tensorflow
    import deepinterpolation

    if parse(deepinterpolation.__version__) >= parse("0.2.0"):
        HAVE_DEEPINTERPOLATION = True
    else:
        warn("DeepInterpolation version >=0.2.0 is required for the tests. Skipping...")
        HAVE_DEEPINTERPOLATION = False
except ImportError:
    HAVE_DEEPINTERPOLATION = False


def recording_and_shape():
    num_cols = 2
    num_rows = 64
    probe = probeinterface.generate_multi_columns_probe(num_columns=num_cols, num_contact_per_column=num_rows)
    probe.set_device_channel_indices(np.arange(num_cols * num_rows))
    recording = generate_recording(num_channels=num_cols * num_rows, durations=[10.0], sampling_frequency=30000)
    recording.set_probe(probe, in_place=True)
    recording = depth_order(recording)
    recording = zscore(recording)
    desired_shape = (num_rows, num_cols)
    return recording, desired_shape


@pytest.fixture(scope="module")
def recording_and_shape_fixture():
    return recording_and_shape()


@pytest.mark.skipif(not HAVE_DEEPINTERPOLATION, reason="requires deepinterpolation")
def test_deepinterpolation_generator_borders(recording_and_shape_fixture):
    """Test that the generator avoids borders in multi-segment and recording lists cases"""
    from spikeinterface.preprocessing.deepinterpolation.generators import SpikeInterfaceRecordingGenerator

    recording, desired_shape = recording_and_shape_fixture
    recording_multi_segment = append_recordings([recording, recording, recording])
    recording_list = [recording, recording, recording]
    recording_multi_list = [recording_multi_segment, recording_multi_segment, recording_multi_segment]

    gen_multi_segment = SpikeInterfaceRecordingGenerator(recording_multi_segment, desired_shape=desired_shape)
    gen_list = SpikeInterfaceRecordingGenerator(recording_list, desired_shape=desired_shape)
    gen_multi_list = SpikeInterfaceRecordingGenerator(recording_multi_list, desired_shape=desired_shape)

    # exclude in between segments
    assert len(gen_multi_segment.exclude_intervals) == 2
    # exclude in between recordings
    assert len(gen_list.exclude_intervals) == 2
    # exclude in between recordings and segments
    assert len(gen_multi_list.exclude_intervals) == 2 * len(recording_multi_list) + 2


@pytest.mark.skipif(not HAVE_DEEPINTERPOLATION, reason="requires deepinterpolation")
@pytest.mark.dependency()
def test_deepinterpolation_training(recording_and_shape_fixture, create_cache_folder):
    recording, desired_shape = recording_and_shape_fixture

    cache_folder = create_cache_folder
    model_folder = Path(cache_folder) / "training"
    # train
    model_path = train_deepinterpolation(
        recording,
        model_folder=model_folder,
        model_name="training",
        train_start_s=1,
        train_end_s=10,
        train_duration_s=0.02,
        test_start_s=0,
        test_end_s=1,
        test_duration_s=0.01,
        pre_frame=20,
        post_frame=20,
        run_uid="si_test",
        pre_post_omission=1,
        desired_shape=desired_shape,
    )
    print(model_path)


@pytest.mark.skipif(not HAVE_DEEPINTERPOLATION, reason="requires deepinterpolation")
@pytest.mark.dependency(depends=["test_deepinterpolation_training"])
def test_deepinterpolation_transfer(recording_and_shape_fixture, tmp_path, create_cache_folder):
    recording, desired_shape = recording_and_shape_fixture
    cache_folder = create_cache_folder

    existing_model_path = Path(cache_folder) / "training" / "si_test_training_model.h5"
    model_folder = Path(tmp_path) / "transfer"

    # train
    model_path = train_deepinterpolation(
        recording,
        model_folder=model_folder,
        model_name="si_test_transfer",
        existing_model_path=existing_model_path,
        train_start_s=1,
        train_end_s=10,
        train_duration_s=0.02,
        test_start_s=0,
        test_end_s=1,
        test_duration_s=0.01,
        pre_frame=20,
        post_frame=20,
        pre_post_omission=1,
        desired_shape=desired_shape,
    )
    print(model_path)


@pytest.mark.skipif(not HAVE_DEEPINTERPOLATION, reason="requires deepinterpolation")
@pytest.mark.dependency(depends=["test_deepinterpolation_training"])
def test_deepinterpolation_inference(recording_and_shape_fixture, create_cache_folder):
    recording, _ = recording_and_shape_fixture
    pre_frame = post_frame = 20
    cache_folder = create_cache_folder
    existing_model_path = Path(cache_folder) / "training" / "si_test_training_model.h5"

    recording_di = deepinterpolate(
        recording, model_path=existing_model_path, pre_frame=pre_frame, post_frame=post_frame, pre_post_omission=1
    )
    print(recording_di)
    traces_original_first = recording.get_traces(start_frame=0, end_frame=100)
    traces_di_first = recording_di.get_traces(start_frame=0, end_frame=100)
    assert traces_di_first.shape == (100, recording.get_num_channels())
    # first 20 frames should be the same
    np.testing.assert_array_equal(traces_di_first[:pre_frame], traces_original_first[:pre_frame])
    np.any(np.not_equal(traces_di_first[pre_frame:], traces_original_first[pre_frame:]))

    num_samples = recording.get_num_samples()
    traces_original_last = recording.get_traces(start_frame=num_samples - 100, end_frame=num_samples)
    traces_di_last = recording_di.get_traces(start_frame=num_samples - 100, end_frame=num_samples)
    # last 20 frames should be the same
    np.testing.assert_array_equal(traces_di_last[-post_frame:], traces_original_last[-post_frame:])
    np.any(np.not_equal(traces_di_last[:-post_frame:], traces_original_last[:-post_frame:]))


@pytest.mark.skipif(not HAVE_DEEPINTERPOLATION, reason="requires deepinterpolation")
@pytest.mark.dependency(depends=["test_deepinterpolation_training"])
def test_deepinterpolation_inference_multi_job(recording_and_shape_fixture, create_cache_folder):
    recording, _ = recording_and_shape_fixture
    pre_frame = post_frame = 20
    cache_folder = create_cache_folder
    existing_model_path = Path(cache_folder) / "training" / "si_test_training_model.h5"

    recording_di = deepinterpolate(
        recording,
        model_path=existing_model_path,
        pre_frame=pre_frame,
        post_frame=post_frame,
        pre_post_omission=1,
        use_gpu=False,
    )
    print(recording_di)
    recording_di_slice = recording_di.frame_slice(start_frame=0, end_frame=int(0.5 * recording.sampling_frequency))

    recording_di_slice.save(folder=Path(cache_folder) / "di_slice", n_jobs=2, chunk_duration="50ms")
    traces_chunks = recording_di_slice.get_traces()
    traces_full = recording_di_slice.get_traces()
    np.testing.assert_array_equal(traces_chunks, traces_full)


if __name__ == "__main__":
    recording_shape = recording_and_shape()
    test_deepinterpolation_training(recording_shape)
    # test_deepinterpolation_transfer()
    # test_deepinterpolation_inference(recording_shape)
    # test_deepinterpolation_inference_multi_job()
    # test_deepinterpolation_generator_borders(recording_shape)
