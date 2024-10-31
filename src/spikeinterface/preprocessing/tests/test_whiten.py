import pytest
import numpy as np

from spikeinterface.core import generate_recording
from spikeinterface.core import BaseRecording, BaseRecordingSegment
from spikeinterface.preprocessing import whiten, scale, compute_whitening_matrix
from spikeinterface.preprocessing.whiten import compute_covariance_matrix

from pathlib import Path
import spikeinterface.full as si  # TOOD: is this a bad idea? remove!


class CustomRecording(BaseRecording):
    """

    """
    def __init__(self, durations, num_channels, channel_ids, sampling_frequency, dtype):
        BaseRecording.__init__(self, sampling_frequency=sampling_frequency, channel_ids=channel_ids, dtype=dtype)

        num_samples = [dur * sampling_frequency for dur in durations]

        for sample_num in num_samples:
            rec_segment = CustomRecordingSegment(
                sample_num,
                num_channels,
                sampling_frequency,
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = {
            "num_channels": num_channels,
            "durations": durations,
            "sampling_frequency": sampling_frequency,
        }

# TODO
# 1) save covariance matrix
# 2)


class CustomRecordingSegment(BaseRecordingSegment):
    """
    """
    def __init__(self, num_samples, num_channels, sampling_frequency):
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.sampling_frequency = sampling_frequency
        self.data = np.zeros((num_samples, num_channels))

        self.t_start = None
        self.time_vector = None

    def set_data(self, data):
        # TODO: do some checks
        self.data = data

    def get_traces(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: list | None = None,
    ):
        return self.data[start_frame:end_frame, channel_indices]

    def get_num_samples(self):
        return self.num_samples


# TODO: return random cghuns scaled vs unscled

@pytest.mark.parametrize("eps", [1e-8, 1])
@pytest.mark.parametrize("num_segments", [1, 2])
@pytest.mark.parametrize("dtype", [np.float32])  # np.int16
def test_compute_whitening_matrix(eps, num_segments, dtype):
    """
    """
    num_channels = 3

    recording = CustomRecording(
        durations=[10, 10] if num_segments == 2 else [10],  # will auto-fill zeros
        num_channels=num_channels,
        channel_ids=np.arange(num_channels),
        sampling_frequency=30000,
        dtype=dtype
    )
    num_samples = recording.get_num_samples(segment_index=0)

    # 1) setup the data with known mean and covariance.
    mean_1 = mean_2 = np.zeros(num_channels)  # TODO: diferent tests
    # mean_2 = np.arange(num_channels)

    # Covariances for simulated data. Limit off-diagonals larger than variances
    # for realism + stability / PSD. Actually, a better way is to just get
    # some random data and compute x.T@x#
    cov_1 = np.array(
        [[1, 0.5, 0],
         [0.5, 1,  -0.25],
         [0,  -0.25, 1]]
    )
    seg_1_data = np.random.multivariate_normal(mean_1, cov_1, recording.get_num_samples(segment_index=0))
    seg_1_data = seg_1_data.astype(dtype)

    recording._recording_segments[0].set_data(seg_1_data)
    assert np.array_equal(recording.get_traces(segment_index=0), seg_1_data), "segment 1 test setup did not work."

    if num_segments == 2:
        recording._recording_segments[1].set_data(
            np.zeros((num_samples, num_channels))
        )

    _, test_cov, test_mean = compute_covariance_matrix(
        recording,
        apply_mean=True,
        regularize=False,
        regularize_kwargs={},
        random_chunk_kwargs=dict(
            num_chunks_per_segment=1,
            chunk_size=recording.get_num_samples(segment_index=0)-1,
        )
    )

    if num_segments == 1:
        assert np.allclose(test_cov, cov_1, rtol=0, atol=0.01)
    else:
        assert np.allclose(test_cov, cov_1 / 2, rtol=0, atol=0.01)

    #  test_cov
    # TOOD: own test for mean

    whitened_recording = si.whiten(
        recording,
        apply_mean=True,
        regularize=False,
        regularize_kwargs={},
        num_chunks_per_segment=1,
        chunk_size=recording.get_num_samples(segment_index=0) - 1,
        eps=eps
    )

    W = whitened_recording._kwargs["W"]
    U, S, Vt = np.linalg.svd(W)
    S_ = (1 / S) ** 2 - eps
    P = U @ np.diag(S_) @ Vt

    if num_segments == 1:
        assert np.allclose(P, cov_1, rtol=0, atol=0.01)
    else:
        assert np.allclose(P, cov_1 / 2, rtol=0, atol=0.01)

    # TODO:
    # 1) test int16, MVN is not going to work. Completely new test that just tests against X.T@T/n
    # 2) test apply mean on / off and means
    # 3) make clear eps is tested above
    # 4) test regularisation (use existing approach). Maybe test directly against sklearn function
    # 5 )test local vs. global
    # 6) monkeypatch regularisation and random kwargs to check they are passed correctly.
    # 7) test radius and int scale in the simplest way
    # 8) test W, M are saved correctly

def test_whiten(create_cache_folder):
    cache_folder = create_cache_folder
    rec = generate_recording(num_channels=4, seed=2205)

    print(rec.get_channel_locations())
    random_chunk_kwargs = {"seed": 2205}
    W1, M = compute_whitening_matrix(rec, "global", random_chunk_kwargs, apply_mean=False, radius_um=None)
    # print(W)
    # print(M)

    with pytest.raises(AssertionError):
        W, M = compute_whitening_matrix(rec, "local", random_chunk_kwargs, apply_mean=False, radius_um=None)
    W, M = compute_whitening_matrix(rec, "local", random_chunk_kwargs, apply_mean=False, radius_um=25)
    # W must be sparse
    np.sum(W == 0) == 6

    rec2 = whiten(rec)
    rec2.save(verbose=False)

    # test dtype
    rec_int = scale(rec2, dtype="int16")
    rec3 = whiten(rec_int, dtype="float16")
    rec3 = rec3.save(folder=cache_folder / "rec1")
    assert rec3.get_dtype() == "float16"

    # test parallel
    rec_par = rec3.save(folder=cache_folder / "rec_par", n_jobs=2)
    np.testing.assert_array_equal(rec3.get_traces(segment_index=0), rec_par.get_traces(segment_index=0))

    with pytest.raises(AssertionError):
        rec4 = whiten(rec_int, dtype=None)
    rec4 = whiten(rec_int, dtype=None, int_scale=256)
    assert rec4.get_dtype() == "int16"
    assert rec4._kwargs["M"] is None

    # test regularization : norm should be smaller
    W2, M = compute_whitening_matrix(rec, "global", random_chunk_kwargs, apply_mean=False, regularize=True)
    assert np.linalg.norm(W1) > np.linalg.norm(W2)
