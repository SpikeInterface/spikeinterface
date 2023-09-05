import pytest
import numpy as np
from pathlib import Path

import probeinterface as pi
from spikeinterface import download_dataset, generate_recording
from spikeinterface.extractors import read_mearec, read_spikeglx, read_openephys
from spikeinterface.preprocessing import depth_order, zscore

from spikeinterface.preprocessing.deepinterpolation import train_deepinterpolation, deepinterpolate


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "preprocessing"
else:
    cache_folder = Path("cache_folder") / "preprocessing"


def recording_and_shape():
    num_cols = 2
    num_rows = 64
    probe = pi.generate_multi_columns_probe(num_columns=num_cols, num_contact_per_column=num_rows)
    probe.set_device_channel_indices(np.arange(num_cols * num_rows))
    recording = generate_recording(num_channels=num_cols * num_rows, durations=[10.0], sampling_frequency=30000)
    recording.set_probe(probe, in_place=True)
    recording = depth_order(recording)
    recording = zscore(recording)
    desired_shape = (num_rows, num_cols)
    return recording, desired_shape


@pytest.fixture
def recording_and_shape_fixture():
    return recording_and_shape()


def test_deepinterpolation_training(recording_and_shape_fixture):
    recording, desired_shape = recording_and_shape_fixture

    model_folder = Path(cache_folder) / "training"
    # train
    model_path = train_deepinterpolation(
        recording,
        model_folder=model_folder,
        model_name="training",
        train_start_s=1,
        train_end_s=10,
        train_duration_s=0.1,
        test_start_s=0,
        test_end_s=1,
        test_duration_s=0.05,
        pre_frame=20,
        post_frame=20,
        run_uid="si_test",
        pre_post_omission=1,
        desired_shape=desired_shape,
    )
    print(model_path)


@pytest.mark.dependency(depends=["test_deepinterpolation_training"])
def test_deepinterpolation_transfer(recording_and_shape_fixture, tmp_path):
    recording, desired_shape = recording_and_shape_fixture

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
        train_duration_s=0.1,
        test_start_s=0,
        test_end_s=1,
        test_duration_s=0.05,
        pre_frame=20,
        post_frame=20,
        pre_post_omission=1,
        desired_shape=desired_shape,
    )
    print(model_path)


@pytest.mark.dependency(depends=["test_deepinterpolation_training"])
def test_deepinterpolation_inference(recording_and_shape_fixture):
    recording, desired_shape = recording_and_shape_fixture
    pre_frame = post_frame = 20
    existing_model_path = Path(cache_folder) / "training" / "si_test_training_model.h5"

    recording_di = deepinterpolate(
        recording, model_path=existing_model_path, pre_frame=pre_frame, post_frame=post_frame, pre_post_omission=1
    )
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


@pytest.mark.dependency(depends=["test_deepinterpolation_training"])
def test_deepinterpolation_inference_multi_job(recording_and_shape_fixture):
    recording, desired_shape = recording_and_shape_fixture
    pre_frame = post_frame = 20
    existing_model_path = Path(cache_folder) / "training" / "si_test_training_model.h5"

    recording_di = deepinterpolate(
        recording,
        model_path=existing_model_path,
        pre_frame=pre_frame,
        post_frame=post_frame,
        pre_post_omission=1,
        use_gpu=False,
    )
    recording_di_slice = recording_di.frame_slice(start_frame=0, end_frame=int(0.5 * recording.sampling_frequency))

    recording_di_slice.save(folder=Path(cache_folder) / "di_slice", n_jobs=2, chunk_duration="50ms")
    traces_chunks = recording_di_slice.get_traces()
    traces_full = recording_di_slice.get_traces()
    np.testing.assert_array_equal(traces_chunks, traces_full)
