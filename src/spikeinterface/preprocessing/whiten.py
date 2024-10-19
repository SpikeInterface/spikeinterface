from __future__ import annotations

import numpy as np

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core.core_tools import define_function_from_class

from ..core import get_random_data_chunks, get_channel_distances
from .filter import fix_dtype
from ..core.globals import get_global_job_kwargs


class WhitenRecording(BasePreprocessor):
    """
    Whitens the recording extractor traces.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to be whitened.
    dtype : None or dtype, default: None
        If None the the parent dtype is kept.
        For integer dtype a int_scale must be also given.
    mode : "global" | "local", default: "global"
        "global" use the entire covariance matrix to compute the W matrix
        "local" use local covariance (by radius) to compute the W matrix
    radius_um : None or float, default: None
        Used for mode = "local" to get the neighborhood
    apply_mean : bool, default: False
        Substract or not the mean matrix M before the dot product with W.
    int_scale : None or float, default: None
        Apply a scaling factor to fit the integer range.
        This is used when the dtype is an integer, so that the output is scaled.
        For example, a value of `int_scale=200` will scale the traces value to a standard deviation of 200.
    eps : float or None, default: None
        Small epsilon to regularize SVD.
        If None, eps is defaulted to 1e-8. If the data is float type and scaled down to very small values,
        then the eps is automatically set to a small fraction (1e-3) of the median of the squared data.
    W : 2d np.array or None, default: None
        Pre-computed whitening matrix
    M : 1d np.array or None, default: None
        Pre-computed means.
        M can be None when previously computed with apply_mean=False
    regularize : bool, default: False
        Boolean to decide if we want to regularize the covariance matrix, using a chosen method
        of sklearn, specified in regularize_kwargs. Default is GraphicalLassoCV
    regularize_kwargs : {'method' : 'GraphicalLassoCV'}
        Dictionary of the parameters that could be provided to the method of sklearn, if
        the covariance matrix needs to be regularized.
    **random_chunk_kwargs : Keyword arguments for `spikeinterface.core.get_random_data_chunk()` function

    Returns
    -------
    whitened_recording : WhitenRecording
        The whitened recording extractor
    """

    def __init__(
        self,
        recording,
        dtype=None,
        apply_mean=False,
        regularize=False,
        regularize_kwargs=None,
        mode="global",
        radius_um=100.0,
        int_scale=None,
        eps=None,
        W=None,
        M=None,
        **random_chunk_kwargs,
    ):
        # fix dtype
        dtype_ = fix_dtype(recording, dtype)

        if dtype_.kind == "i":
            assert int_scale is not None, "For recording with dtype=int you must set dtype=float32 OR set a int_scale"

        if W is not None:
            W = np.asarray(W)
            if M is not None:
                M = np.asarray(M)
        else:
            W, M = compute_whitening_matrix(
                recording,
                mode,
                random_chunk_kwargs,
                apply_mean,
                radius_um=radius_um,
                eps=eps,
                regularize=regularize,
                regularize_kwargs=regularize_kwargs,
            )

        BasePreprocessor.__init__(self, recording, dtype=dtype_)

        for parent_segment in recording._recording_segments:
            rec_segment = WhitenRecordingSegment(parent_segment, W, M, dtype_, int_scale)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording,
            dtype=dtype,
            mode=mode,
            radius_um=radius_um,
            apply_mean=apply_mean,
            regularize=regularize,
            regularize_kwargs=regularize_kwargs,
            int_scale=float(int_scale) if int_scale is not None else None,
            M=M.tolist() if M is not None else None,
            W=W.tolist(),
        )
        self._kwargs.update(random_chunk_kwargs)


class WhitenRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, W, M, dtype, int_scale):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.W = W
        self.M = M
        self.dtype = dtype
        self.int_scale = int_scale

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))
        traces_dtype = traces.dtype
        # if uint --> force int
        if traces_dtype.kind == "u":
            traces = traces.astype("float32")

        if self.M is not None:
            whiten_traces = (traces - self.M) @ self.W
        else:
            whiten_traces = traces @ self.W

        whiten_traces = whiten_traces[:, channel_indices]

        if self.int_scale is not None:
            whiten_traces *= self.int_scale

        return whiten_traces.astype(self.dtype)


# function for API
whiten = define_function_from_class(source_class=WhitenRecording, name="whiten")


def compute_whitening_matrix(
    recording, mode, random_chunk_kwargs, apply_mean, radius_um=None, eps=None, regularize=False, regularize_kwargs=None
):
    """
    Compute whitening matrix

    Parameters
    ----------
    recording : BaseRecording
        The recording object
    mode : str
        The mode to compute the whitening matrix.

        * "global": compute SVD using all channels
        * "local": compute SVD on local neighborhood (controlled by `radius_um`)

    random_chunk_kwargs : dict
        Keyword arguments for  get_random_data_chunks()
    apply_mean : bool
        If True, the mean is removed prior to computing the covariance
    radius_um : float or None, default: None
        Used for mode = "local" to get the neighborhood
    eps : float or None, default: None
        Small epsilon to regularize SVD. If None, the default is set to 1e-8, but if the data is float type and scaled
        down to very small values, eps is automatically set to a small fraction (1e-3) of the median of the squared data.
    regularize : bool, default: False
        Boolean to decide if we want to regularize the covariance matrix, using a chosen method
        of sklearn, specified in regularize_kwargs. Default is GraphicalLassoCV
    regularize_kwargs : {'method' : 'GraphicalLassoCV'}
        Dictionary of the parameters that could be provided to the method of sklearn, if
        the covariance matrix needs to be regularized.
    Returns
    -------
    W : 2D array
        The whitening matrix
    M : 2D array or None
        The "mean" matrix

    """
    random_data = get_random_data_chunks(recording, concatenated=True, return_scaled=False, **random_chunk_kwargs)

    regularize_kwargs = regularize_kwargs if regularize_kwargs is not None else {"method": "GraphicalLassoCV"}

    if apply_mean:
        M = np.mean(random_data, axis=0)
        M = M[None, :]
        data = random_data - M
    else:
        M = None
        data = random_data

    if not regularize:
        cov = data.T @ data
        cov = cov / data.shape[0]
    else:
        import sklearn.covariance

        method = regularize_kwargs.pop("method")
        regularize_kwargs["assume_centered"] = True
        estimator_class = getattr(sklearn.covariance, method)
        estimator = estimator_class(**regularize_kwargs)
        estimator.fit(data)
        cov = estimator.covariance_

    # Here we determine eps used below to avoid division by zero.
    # Typically we can assume that data is either unscaled integers or in units of
    # uV, but this is not always the case. When data
    # is float type and scaled down to very small values, then the
    # default eps=1e-8 can be too large, resulting in incorrect
    # whitening. We therefore check to see if the data is float
    # type and we estimate a more reasonable eps in the case
    # where the data is on a scale less than 1.
    if eps is None:
        eps = 1e-8
    if data.dtype.kind == "f":
        median_data_sqr = np.median(data**2)  # use the square because cov (and hence S) scales as the square
        if median_data_sqr < 1 and median_data_sqr > 0:
            eps = max(1e-16, median_data_sqr * 1e-3)  # use a small fraction of the median of the squared data

    if mode == "global":
        U, S, Ut = np.linalg.svd(cov, full_matrices=True)
        W = (U @ np.diag(1 / np.sqrt(S + eps))) @ Ut
    elif mode == "local":
        assert radius_um is not None
        n = cov.shape[0]
        distances = get_channel_distances(recording)
        W = np.zeros((n, n), dtype="float64")
        for c in range(n):
            (inds,) = np.nonzero(distances[c, :] <= radius_um)
            cov_local = cov[inds, :][:, inds]
            U, S, Ut = np.linalg.svd(cov_local, full_matrices=True)
            W_local = (U @ np.diag(1 / np.sqrt(S + eps))) @ Ut
            W[inds, c] = W_local[c == inds]
    else:
        raise ValueError(f"compute_whitening_matrix : wrong mode {mode}")

    return W, M
