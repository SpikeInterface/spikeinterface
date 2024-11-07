import pytest
import numpy as np

from spikeinterface.core import generate_recording
from spikeinterface.core import BaseRecording, BaseRecordingSegment
from spikeinterface.preprocessing import whiten, scale, compute_whitening_matrix
from spikeinterface.preprocessing.whiten import compute_sklearn_covariance_matrix

from pathlib import Path
import spikeinterface.full as si  # TOOD: is this a bad idea? remove!


class CustomRecording(BaseRecording):
    """ """

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


class CustomRecordingSegment(BaseRecordingSegment):
    """ """

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


class TestWhiten:

    def get_float_test_data(self, num_segments, dtype, mean=None, covar=None):
        """
        mention the segment thing
        """
        num_channels = 3
        dtype = np.float32

        if mean is None:
            mean = np.zeros(num_channels)

        if covar is None:
            covar = np.array([[1, 0.5, 0], [0.5, 1, -0.25], [0, -0.25, 1]])

        recording = self.get_empty_custom_recording(num_segments, num_channels, dtype)

        seg_1_data = np.random.multivariate_normal(
            mean, covar, recording.get_num_samples(segment_index=0)  # TODO: RENAME!
        )
        if dtype == np.float32:
            seg_1_data = seg_1_data.astype(dtype)
        elif dtype == np.int16:
            seg_1_data = np.round(seg_1_data * 32767).astype(np.int16)
        else:
            raise ValueError("dtype must be float32 or int16")

        recording._recording_segments[0].set_data(seg_1_data)
        assert np.array_equal(recording.get_traces(segment_index=0), seg_1_data), "segment 1 test setup did not work."

        return mean, covar, recording

    def get_empty_custom_recording(self, num_segments, num_channels, dtype):
        """ """
        return CustomRecording(
            durations=[10, 10] if num_segments == 2 else [10],  # will auto-fill zeros
            num_channels=num_channels,
            channel_ids=np.arange(num_channels),
            sampling_frequency=30000,
            dtype=dtype,
        )

    def covar_from_whitening_mat(self, whitened_recording, eps):
        """
        The whitening matrix is computed as the
        inverse square root of the covariance matrix
        (Sigma, 'S' below + some eps for regularising.

        Here the inverse process is performed to compute
        the covariance matrix from the whitening matrix
        for testing purposes. This allows the entire
        workflow to be tested rather than subfunction only.
        """
        W = whitened_recording._kwargs["W"]
        U, D, Vt = np.linalg.svd(W)
        D_new = (1 / D) ** 2 - eps
        S = U @ np.diag(D_new) @ Vt

        return S

    @pytest.mark.parametrize("eps", [1e-8, 1])
    @pytest.mark.parametrize("dtype", [np.float32, np.int16])
    def test_compute_covariance_matrix(self, dtype, eps):
        """ """
        eps = 1e-8
        mean, covar, recording = self.get_float_test_data(num_segments=1, dtype=dtype)

        whitened_recording = si.whiten(
            recording,
            apply_mean=True,
            regularize=False,
            num_chunks_per_segment=1,
            chunk_size=recording.get_num_samples(segment_index=0) - 1,
            eps=eps,
        )

        test_covar = self.covar_from_whitening_mat(whitened_recording, eps)

        assert np.allclose(test_covar, covar, rtol=0, atol=0.01)

        if eps != 1:
            X = whitened_recording.get_traces()
            X = X - np.mean(X, axis=0)
            S = X.T @ X / X.shape[0]

            assert np.allclose(S, np.eye(recording.get_num_channels()), rtol=0, atol=1e-4)

    def test_compute_covariance_matrix_float_2_segments(self):
        """ """
        eps = 1e-8
        mean, covar, recording = self.get_float_test_data(num_segments=2, dtype=np.float32)

        recording._recording_segments[1].set_data(
            np.zeros((recording.get_num_samples(segment_index=0), recording.get_num_channels()))
        )

        whitened_recording = si.whiten(
            recording,
            apply_mean=True,
            regularize=False,
            regularize_kwargs={},
            num_chunks_per_segment=1,
            chunk_size=recording.get_num_samples(segment_index=0) - 1,
            eps=eps,
        )

        test_covar = self.covar_from_whitening_mat(whitened_recording, eps)

        assert np.allclose(test_covar, covar / 2, rtol=0, atol=0.01)

    @pytest.mark.parametrize("apply_mean", [True, False])
    def test_apply_mean(self, apply_mean):

        means = np.array([10, 20, 30])

        eps = 1e-8
        mean, covar, recording = self.get_float_test_data(num_segments=1, dtype=np.float32, mean=means)

        whitened_recording = si.whiten(
            recording,
            apply_mean=apply_mean,
            regularize=False,
            regularize_kwargs={},
            num_chunks_per_segment=1,
            chunk_size=recording.get_num_samples(segment_index=0) - 1,
            eps=eps,
        )

        test_covar = self.covar_from_whitening_mat(whitened_recording, eps)

        if apply_mean:
            assert np.allclose(test_covar, covar, rtol=0, atol=0.01)
        else:
            assert np.allclose(np.diag(test_covar), means**2, rtol=0, atol=5)

        breakpoint()  # TODO: check whitened data is cov identity even when apply_mean=False...

    def test_compute_sklearn_covariance_matrix(self):
        """
        TODO: assume centered is fixed to True
        Test some random stuff

        # TODO: this is not appropraite for all covariance functions. only one with the fit method! e.g. does not work with leodit_wolf
        """
        from sklearn import covariance

        X = np.random.randn(100, 100)

        test_cov = compute_sklearn_covariance_matrix(
            X, {"method": "GraphicalLasso", "alpha": 1, "enet_tol": 0.04}
        )  # RENAME test_cov
        cov = covariance.GraphicalLasso(alpha=1, enet_tol=0.04, assume_centered=True).fit(X).covariance_
        assert np.allclose(test_cov, cov)

        test_cov = compute_sklearn_covariance_matrix(
            X, {"method": "ShrunkCovariance", "shrinkage": 0.3}
        )  # RENAME test_cov
        cov = covariance.ShrunkCovariance(shrinkage=0.3, assume_centered=True).fit(X).covariance_
        assert np.allclose(test_cov, cov)

    def test_whiten_regularisation_norm(self):
        """ """
        from sklearn import covariance

        _, _, recording = self.get_float_test_data(num_segments=1, dtype=np.float32)

        whitened_recording = si.whiten(
            recording,
            regularize=True,
            regularize_kwargs={"method": "ShrunkCovariance", "shrinkage": 0.3},
            apply_mean=True,
            num_chunks_per_segment=1,
            chunk_size=recording.get_num_samples(segment_index=0) - 1,
            eps=1e-8,
        )

        test_covar = self.covar_from_whitening_mat(whitened_recording, eps=1e-8)

        X = recording.get_traces()[:-1, :]
        X = X - np.mean(X, axis=0)

        covar = covariance.ShrunkCovariance(shrinkage=0.3, assume_centered=True).fit(X).covariance_

        assert np.allclose(test_covar, covar, rtol=0, atol=1e-4)

    def test_local_vs_global_whiten(self):
        # Make test data with 4 channels, known covar between all
        # do with radius = 2. compute manually. Will need to change channel locations
        # check matches well.
        pass

    def test_passed_W_and_M(self):
        pass

    def test_all_kwargs(self):
        pass

    def test_saved_to_disk(self):
        # check attributes are saved properly
        pass


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
