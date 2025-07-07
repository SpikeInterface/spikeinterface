import importlib.util

import pytest
import numpy as np

from spikeinterface.core import generate_recording
from spikeinterface.core.numpyextractors import NumpyRecording
from spikeinterface.preprocessing import whiten, scale, compute_whitening_matrix
from spikeinterface.preprocessing.whiten import compute_sklearn_covariance_matrix

sklearn_spec = importlib.util.find_spec("sklearn")
if sklearn_spec is not None:
    from sklearn import covariance as sklearn_covariance

    HAS_SKLEARN = True
else:
    HAS_SKLEARN = False


class TestWhiten:
    """
    Test the whitening preprocessing step.

    The strategy is to generate a recording that has data
    with a known covariance matrix, then testing that the
    covariance matrix is computed properly and that the
    returned data is indeed white.
    """

    def get_test_recording(self, dtype, means=None):
        """
        Generate a set of test data with known covariance matrix and mean.
        Test data is drawn from a multivariate Gaussian distribute with
        means `mean` and covariance matrix `cov_mat`.

        A fixture is not used because we often want to change the options,
        and it is very quick to generate this test data.

        The number of channels (3) and covariance matrix is fixed
        and directly tested against in below tests.

        Parameters
        ----------

        dtype : np.float32 | np.int16
            Datatype of the generated recording.

        means : None | np.ndarray
            The `means` should be an array of length 3 (num samples)
            or `None`. If `None`, means will be zero.
        """
        sampling_frequency = 30000
        num_samples = int(10 * sampling_frequency)  # 10 s recording

        means, cov_mat, data = self.get_test_data_with_known_distribution(num_samples, dtype, means)

        recording = NumpyRecording([data], sampling_frequency)

        return means, cov_mat, recording

    def get_test_data_with_known_distribution(self, num_samples, dtype, means=None, seed=0):
        """
        Create multivariate normal data with known means and covariance matrixs.
        If `dtype` is int16, scale to full range of int16 before cast.
        """
        num_channels = 3

        if means is None:
            means = np.zeros(num_channels)

        cov_mat = np.array([[1, 0.5, 0], [0.5, 1, -0.25], [0, -0.25, 1]])

        rng = np.random.RandomState(seed)
        data = rng.multivariate_normal(means, cov_mat, num_samples)

        # Set the dtype, if `int16`, first scale to +/- 1 then cast to int16 range.
        if dtype == np.float32:
            data = data.astype(dtype)

        elif dtype == np.int16:
            data /= data.max()
            data = np.round((data) * 32767).astype(np.int16)
        else:
            raise ValueError("dtype must be float32 or int16")

        return means, cov_mat, data

    def cov_mat_from_whitening_mat(self, whitened_recording, eps):
        """
        The whitening matrix is computed as the
        inverse square root of the covariance matrix
        (Sigma, 'S' below + some eps for regularising.)

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

    def assert_recording_is_white(self, recording):
        """
        Compute the covariance matrix of the recording,
        and assert that it is close to identity.
        """
        X = recording.get_traces()
        S = self.compute_cov_mat(X)
        assert np.allclose(S, np.eye(recording.get_num_channels()), rtol=0, atol=1e-4)

    def compute_cov_mat(self, X):
        """
        Estimate the covariance matrix as the sample covariance.
        """
        X = X - np.mean(X, axis=0)
        S = X.T @ X / X.shape[0]
        return S

    ###################################################################################
    # Tests
    ###################################################################################

    @pytest.mark.parametrize("dtype", [np.float32, np.int16])
    def test_compute_covariance_matrix(self, dtype):
        """
        Test that the covariance matrix is computed as expected and
        data is white after whitening step. Test against float32 and
        int16, testing int16 is important to ensure data
        is cast to float before computing the covariance matrix,
        otherwise it can overflow.
        """
        eps = 1e-16
        _, cov_mat, recording = self.get_test_recording(dtype=dtype)

        whitened_recording = whiten(
            recording,
            apply_mean=False,
            regularize=False,
            num_chunks_per_segment=1,
            chunk_size=recording.get_num_samples(segment_index=0) - 1,
            eps=eps,
            dtype=np.float32,
        )

        test_cov_mat = self.cov_mat_from_whitening_mat(whitened_recording, eps)

        # If the data is in `int16` the covariance matrix will be scaled up
        # as data is set to +/32767 range before cast.
        if dtype == np.int16:
            test_cov_mat /= test_cov_mat[0][0]
        assert np.allclose(test_cov_mat, cov_mat, rtol=0, atol=0.01)

        self.assert_recording_is_white(whitened_recording)

    def test_non_default_eps(self):
        """
        Try a new non-default eps and check that it is correctly
        propagated to the matrix computation. The test is that
        the `cov_mat_from_whitening_mat` should recovery exctly
        the cov mat if the correct eps is used.
        """
        eps = 1
        _, cov_mat, recording = self.get_test_recording(dtype=np.float32)

        whitened_recording = whiten(
            recording,
            apply_mean=False,
            regularize=False,
            num_chunks_per_segment=1,
            chunk_size=recording.get_num_samples(segment_index=0) - 1,
            eps=eps,
        )
        test_cov_mat = self.cov_mat_from_whitening_mat(whitened_recording, eps)
        assert np.allclose(test_cov_mat, cov_mat, rtol=0, atol=0.01)

    def test_compute_covariance_matrix_2_segments(self):
        """
        Check that the covariance marix is estimated from across
        segments in a multi-segment recording. This is done simply
        by setting the second segment as all zeros and checking the
        estimated covariances are all halved. This makes sense as
        the zeros do not affect the covariance estimation
        but the covariance matrix is scaled by 1 / N.
        """
        eps = 1e-16
        sampling_frequency = 30000
        num_samples = 10 * sampling_frequency

        _, cov_mat, data = self.get_test_data_with_known_distribution(num_samples, np.float32)

        traces_list = [data, np.zeros_like(data)]

        recording = NumpyRecording(traces_list, sampling_frequency)

        whitened_recording = whiten(
            recording,
            apply_mean=True,
            regularize=False,
            num_chunks_per_segment=1,
            chunk_size=recording.get_num_samples(segment_index=0) - 1,
            eps=eps,
        )

        test_cov_mat = self.cov_mat_from_whitening_mat(whitened_recording, eps)

        assert np.allclose(test_cov_mat, cov_mat / 2, rtol=0, atol=0.01)

    @pytest.mark.parametrize("apply_mean", [True, False])
    def test_apply_mean(self, apply_mean):
        """
        Test that the `apply_mean` argument is propagated correctly.
        Note that in the case `apply_mean=False`, the covariance matrix
        is in unusual scaling and so the varainces alone are checked.
        Also, the data is not as well whitened and so this is not
        tested against.
        """
        means = np.array([10, 20, 30])

        eps = 1e-16
        _, cov_mat, recording = self.get_test_recording(dtype=np.float32, means=means)

        whitened_recording = whiten(
            recording,
            apply_mean=apply_mean,
            regularize=False,
            num_chunks_per_segment=1,
            chunk_size=recording.get_num_samples(segment_index=0) - 1,
            eps=eps,
        )

        test_cov_mat = self.cov_mat_from_whitening_mat(whitened_recording, eps)

        if apply_mean:
            assert np.allclose(test_cov_mat, cov_mat, rtol=0, atol=0.01)
        else:
            assert np.allclose(np.diag(test_cov_mat), means**2, rtol=0, atol=5)

        # Note the recording is typically not white if the mean is
        # not removed prior to covariance matrix estimation.
        if apply_mean:
            self.assert_recording_is_white(whitened_recording)

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn must be installed.")
    def test_compute_sklearn_covariance_matrix(self):
        """
        A basic check that the `compute_sklearn_covariance_matrix`
        function from `whiten.py` computes the same matrix
        as using the sklearn function directly for some
        arbitraily chosen methods / parameters.

        Note that the function-style sklearn covariance
        methods are not supported.
        """
        X = np.random.randn(100, 100)

        test_cov = compute_sklearn_covariance_matrix(X, {"method": "GraphicalLasso", "alpha": 1, "enet_tol": 0.04})
        cov = sklearn_covariance.GraphicalLasso(alpha=1, enet_tol=0.04, assume_centered=True).fit(X).covariance_
        assert np.allclose(test_cov, cov)

        test_cov = compute_sklearn_covariance_matrix(X, {"method": "ShrunkCovariance", "shrinkage": 0.3})
        cov = sklearn_covariance.ShrunkCovariance(shrinkage=0.3, assume_centered=True).fit(X).covariance_
        assert np.allclose(test_cov, cov)

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn must be installed.")
    def test_whiten_regularisation_norm(self):
        """
        Check that the covariance matrix estimated by the
        whitening preprocessing is the same as the one
        computed from sklearn when regularise kwargs are given.
        """
        _, _, recording = self.get_test_recording(dtype=np.float32)

        whitened_recording = whiten(
            recording,
            regularize=True,
            regularize_kwargs={"method": "ShrunkCovariance", "shrinkage": 0.3},
            apply_mean=True,
            num_chunks_per_segment=1,
            chunk_size=recording.get_num_samples(segment_index=0) - 1,
            eps=1e-16,
        )

        test_cov_mat = self.cov_mat_from_whitening_mat(whitened_recording, eps=1e-16)

        # Compute covariance matrix using sklearn directly and compare.
        X = recording.get_traces()[:-1, :]
        X = X - np.mean(X, axis=0)
        cov_mat = sklearn_covariance.ShrunkCovariance(shrinkage=0.3, assume_centered=True).fit(X).covariance_

        assert np.allclose(test_cov_mat, cov_mat, rtol=0, atol=1e-4)

    def test_local_vs_global_whiten(self):
        """
        Generate a set of channels each separated by y_dist. Set the
        radius_um to just above y_dist such that only neighbouring
        channels are considered for whitening. Test that whitening
        is correct for the first pair and last pair.
        """
        _, _, recording = self.get_test_recording(dtype=np.float32)

        y_dist = 2
        recording.set_channel_locations(
            [
                [0.0, 0],
                [0.0, y_dist * 1],
                [0.0, y_dist * 2],
            ]
        )

        results = {"global": None, "local": None}

        for mode in ["global", "local"]:
            whitened_recording = whiten(
                recording,
                apply_mean=True,
                num_chunks_per_segment=1,
                chunk_size=recording.get_num_samples(segment_index=0) - 1,
                eps=1e-16,
                mode=mode,
                radius_um=y_dist + 1e-01,
            )
            results[mode] = whitened_recording

        # In local, parts of the covariance matrix are exactly zero
        # (when pairs of channels are not in the same group).
        assert results["local"]._kwargs["W"][0][2] == 0.0
        assert results["global"]._kwargs["W"][0][2] != 0.0

        # Parse out the data into two pairs of channels
        # from which the local variance was computed.
        whitened_data = results["local"].get_traces()

        set_1 = whitened_data[:, :2] - np.mean(whitened_data[:, :2], axis=0)
        set_2 = whitened_data[:, 1:] - np.mean(whitened_data[:, 1:], axis=0)

        # Check that the pairs of close channels
        # whitened together are white
        assert np.allclose(np.eye(2), self.compute_cov_mat(set_1), rtol=0, atol=1e-2)
        assert np.allclose(np.eye(2), self.compute_cov_mat(set_2), rtol=0, atol=1e-2)

        # Check that the data overall is not white
        assert not np.allclose(np.eye(3), self.compute_cov_mat(whitened_data), rtol=0, atol=1e-2)

    def test_passed_W_and_M(self):
        """
        Check that passing W (whitening matrix) and M (means) is
        sucessfully propagated to the relevant segments and stored
        on the kwargs. It is assumed if this is true, they will
        be used for the actual whitening computation.
        """
        num_chan = 4
        num_samples = 10000

        recording = NumpyRecording(
            [np.zeros((num_samples, num_chan))] * 2,
            sampling_frequency=30000,
        )

        test_W = np.random.normal(size=(num_chan, num_chan))
        test_M = np.random.normal(size=num_chan)

        whitened_recording = whiten(recording, W=test_W, M=test_M)

        for seg_idx in [0, 1]:
            assert np.array_equal(whitened_recording._recording_segments[seg_idx].W, test_W)
            assert np.array_equal(whitened_recording._recording_segments[seg_idx].M, test_M)

        assert whitened_recording._kwargs["W"] == test_W.tolist()
        assert whitened_recording._kwargs["M"] == test_M.tolist()

    def test_whiten_general(self, create_cache_folder):
        """
        Perform some general tests on the whitening functionality.

        First, perform smoke test that `compute_whitening_matrix` is running,
        check recording output datatypes are as expected. Check that
        saving preseves datatype, `int_scale` is propagated, and
        regularisation reduces the norm.
        """
        cache_folder = create_cache_folder
        rec = generate_recording(num_channels=4, seed=2205)

        random_chunk_kwargs = {"seed": 2205}
        W1, M = compute_whitening_matrix(rec, "global", random_chunk_kwargs, apply_mean=False, radius_um=None)

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
            rec4 = whiten(rec_int, dtype=None)  # int_scale should be applied
        rec4 = whiten(rec_int, dtype=None, int_scale=256)
        assert rec4.get_dtype() == "int16"
        assert rec4._kwargs["M"] is None

        # test regularization : norm should be smaller
        if HAS_SKLEARN:
            W2, M = compute_whitening_matrix(rec, "global", random_chunk_kwargs, apply_mean=False, regularize=True)
            assert np.linalg.norm(W1) > np.linalg.norm(W2)
