"""Sorting components: template matching."""

from __future__ import annotations


import numpy as np

from spikeinterface.core import get_noise_levels
from spikeinterface.sortingcomponents.peak_detection import DetectPeakByChannel
from spikeinterface.core.template import Templates


spike_dtype = [
    ("sample_index", "int64"),
    ("channel_index", "int64"),
    ("cluster_index", "int64"),
    ("amplitude", "float64"),
    ("segment_index", "int64"),
]

try:
    import torch
    import torch.nn.functional as F

    HAVE_TORCH = True
    from torch.nn.functional import conv1d
except ImportError:
    HAVE_TORCH = False

from .base import BaseTemplateMatching


def compress_templates(
    templates_array, approx_rank, remove_mean=False, return_new_templates=True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Compress templates using singular value decomposition.

    Parameters
    ----------
    templates : ndarray (num_templates, num_samples, num_channels)
        Spike template waveforms.
    approx_rank : int
        Rank of the compressed template matrices.

    Returns
    -------
    compressed_templates : (ndarray, ndarray, ndarray)
        Templates compressed by singular value decomposition into temporal, singular, and spatial components.
    """
    if remove_mean:
        templates_array -= templates_array.mean(axis=(1, 2))[:, None, None]

    num_templates, num_samples, num_channels = templates_array.shape
    temporal = np.zeros((num_templates, num_samples, approx_rank), dtype=np.float32)
    spatial = np.zeros((num_templates, approx_rank, num_channels), dtype=np.float32)
    singular = np.zeros((num_templates, approx_rank), dtype=np.float32)

    for i in range(num_templates):
        i_temporal, i_singular, i_spatial = np.linalg.svd(templates_array[i], full_matrices=False)
        temporal[i, :, : min(approx_rank, num_channels)] = i_temporal[:, :approx_rank]
        spatial[i, : min(approx_rank, num_channels), :] = i_spatial[:approx_rank, :]
        singular[i, : min(approx_rank, num_channels)] = i_singular[:approx_rank]

    if return_new_templates:
        templates_array = np.matmul(temporal * singular[:, np.newaxis, :], spatial)
    else:
        templates_array = None

    return temporal, singular, spatial, templates_array


def compute_overlaps(templates, num_samples, num_channels, sparsities):
    import scipy.spatial
    import scipy

    num_templates = len(templates)

    dense_templates = np.zeros((num_templates, num_samples, num_channels), dtype=np.float32)
    for i in range(num_templates):
        dense_templates[i, :, sparsities[i]] = templates[i].T

    size = 2 * num_samples - 1

    all_delays = list(range(0, num_samples + 1))

    overlaps = {}

    for delay in all_delays:
        source = dense_templates[:, :delay, :].reshape(num_templates, -1)
        target = dense_templates[:, num_samples - delay :, :].reshape(num_templates, -1)

        overlaps[delay] = scipy.sparse.csr_matrix(source.dot(target.T))

        if delay < num_samples:
            overlaps[size - delay + 1] = overlaps[delay].T.tocsr()

    new_overlaps = []

    for i in range(num_templates):
        data = [overlaps[j][i, :].T for j in range(size)]
        data = scipy.sparse.hstack(data)
        new_overlaps += [data]

    return new_overlaps


class CircusOMPSVDPeeler(BaseTemplateMatching):
    """
    Orthogonal Matching Pursuit inspired from Spyking Circus sorter

    https://elifesciences.org/articles/34518

    This is an Orthogonal Template Matching algorithm. For speed and
    memory optimization, templates are automatically sparsified. Signal
    is convolved with the templates, and as long as some scalar products
    are higher than a given threshold, we use a Cholesky decomposition
    to compute the optimal amplitudes needed to reconstruct the signal.

    IMPORTANT NOTE: small chunks are more efficient for such Peeler,
    consider using 100ms chunk

    Parameters
    ----------
    amplitude : tuple
        (Minimal, Maximal) amplitudes allowed for every template
    max_failures : int
        Stopping criteria of the OMP algorithm, as number of retry while updating amplitudes
    sparse_kwargs : dict
        Parameters to extract a sparsity mask from the waveform_extractor, if not
        already sparse.
    rank : int, default: 5
        Number of components used internally by the SVD
    vicinity : int
        Size of the area surrounding a spike to perform modification (expressed in terms
        of template temporal width)
    engine : string in ["numpy", "torch", "auto"]. Default "auto"
        The engine to use for the convolutions
    torch_device : string in ["cpu", "cuda", None]. Default "cpu"
        Controls torch device if the torch engine is selected
    -----
    """

    _more_output_keys = [
        "norms",
        "temporal",
        "spatial",
        "singular",
        "units_overlaps",
        "unit_overlaps_indices",
        "normed_templates",
        "overlaps",
    ]

    def __init__(
        self,
        recording,
        return_output=True,
        parents=None,
        templates=None,
        amplitudes=[0.6, np.inf],
        stop_criteria="max_failures",
        max_failures=5,
        omp_min_sps=0.1,
        relative_error=5e-5,
        rank=5,
        ignore_inds=[],
        vicinity=2,
        precomputed=None,
        engine="numpy",
        torch_device="cpu",
    ):

        BaseTemplateMatching.__init__(self, recording, templates, return_output=True, parents=None)

        self.num_channels = recording.get_num_channels()
        self.num_samples = templates.num_samples
        self.nbefore = templates.nbefore
        self.nafter = templates.nafter
        self.sampling_frequency = recording.get_sampling_frequency()
        self.vicinity = vicinity * self.num_samples
        assert engine in ["numpy", "torch", "auto"], "engine should be numpy, torch or auto"
        if engine == "auto":
            if HAVE_TORCH:
                self.engine = "torch"
            else:
                self.engine = "numpy"
        else:
            if engine == "torch":
                assert HAVE_TORCH, "please install torch to use the torch engine"
            self.engine = engine

        assert torch_device in ["cuda", "cpu", None]
        self.torch_device = torch_device

        self.amplitudes = amplitudes
        self.stop_criteria = stop_criteria
        self.max_failures = max_failures
        self.omp_min_sps = omp_min_sps
        self.relative_error = relative_error
        self.rank = rank

        self.num_templates = len(templates.unit_ids)

        if precomputed is None:
            self._prepare_templates()
        else:
            for key in self._more_output_keys:
                assert precomputed[key] is not None, "If templates are provided, %d should also be there" % key
                setattr(self, key, precomputed[key])

        self.ignore_inds = np.array(ignore_inds)

        self.unit_overlaps_tables = {}
        for i in range(self.num_templates):
            self.unit_overlaps_tables[i] = np.zeros(self.num_templates, dtype=int)
            self.unit_overlaps_tables[i][self.unit_overlaps_indices[i]] = np.arange(len(self.unit_overlaps_indices[i]))

        self.margin = 2 * self.num_samples
        self.is_pushed = False

    def _prepare_templates(self):

        assert self.stop_criteria in ["max_failures", "omp_min_sps", "relative_error"]

        if self.templates.sparsity is None:
            sparsity = np.ones((self.num_templates, self.num_channels), dtype=bool)
        else:
            sparsity = self.templates.sparsity.mask

        units_overlaps = np.sum(np.logical_and(sparsity[:, np.newaxis, :], sparsity[np.newaxis, :, :]), axis=2)
        self.units_overlaps = units_overlaps > 0
        self.unit_overlaps_indices = {}
        for i in range(self.num_templates):
            self.unit_overlaps_indices[i] = np.flatnonzero(self.units_overlaps[i])

        templates_array = self.templates.get_dense_templates().copy()
        # Then we keep only the strongest components
        self.temporal, self.singular, self.spatial, templates_array = compress_templates(templates_array, self.rank)

        self.normed_templates = np.zeros(templates_array.shape, dtype=np.float32)
        self.norms = np.zeros(self.num_templates, dtype=np.float32)

        # And get the norms, saving compressed templates for CC matrix
        for count in range(self.num_templates):
            template = templates_array[count][:, sparsity[count]]
            self.norms[count] = np.linalg.norm(template)
            self.normed_templates[count][:, sparsity[count]] = template / self.norms[count]

        self.temporal /= self.norms[:, np.newaxis, np.newaxis]
        self.temporal = np.flip(self.temporal, axis=1)

        self.overlaps = []
        self.max_similarity = np.zeros((self.num_templates, self.num_templates), dtype=np.float32)
        for i in range(self.num_templates):
            num_overlaps = np.sum(self.units_overlaps[i])
            overlapping_units = np.flatnonzero(self.units_overlaps[i])

            # Reconstruct unit template from SVD Matrices
            data = self.temporal[i] * self.singular[i][np.newaxis, :]
            template_i = np.matmul(data, self.spatial[i, :, :])
            template_i = np.flipud(template_i)

            unit_overlaps = np.zeros([num_overlaps, 2 * self.num_samples - 1], dtype=np.float32)

            for count, j in enumerate(overlapping_units):
                overlapped_channels = sparsity[j]
                visible_i = template_i[:, overlapped_channels]

                spatial_filters = self.spatial[j, :, overlapped_channels]
                spatially_filtered_template = np.matmul(visible_i, spatial_filters)
                visible_i = spatially_filtered_template * self.singular[j]

                for rank in range(visible_i.shape[1]):
                    unit_overlaps[count, :] += np.convolve(visible_i[:, rank], self.temporal[j][:, rank], mode="full")

                self.max_similarity[i, j] = np.max(unit_overlaps[count])

            self.overlaps.append(unit_overlaps)

        if self.amplitudes is None:
            distances = np.sort(self.max_similarity, axis=1)[:, ::-1]
            distances = 1 - distances[:, 1] / 2
            self.amplitudes = np.zeros((self.num_templates, 2))
            self.amplitudes[:, 0] = distances
            self.amplitudes[:, 1] = np.inf

        self.spatial = np.moveaxis(self.spatial, [0, 1, 2], [1, 0, 2])
        self.temporal = np.moveaxis(self.temporal, [0, 1, 2], [1, 2, 0])
        self.singular = self.singular.T[:, :, np.newaxis]

    def _push_to_torch(self):
        if self.engine == "torch":
            self.spatial = torch.as_tensor(self.spatial, device=self.torch_device)
            self.singular = torch.as_tensor(self.singular, device=self.torch_device)
            self.temporal = torch.as_tensor(self.temporal.copy(), device=self.torch_device).swapaxes(0, 1)
            self.temporal = torch.flip(self.temporal, (2,))
        self.is_pushed = True

    def get_extra_outputs(self):
        output = {}
        for key in self._more_output_keys:
            output[key] = getattr(self, key)
        return output

    def get_trace_margin(self):
        return self.margin

    def compute_matching(self, traces, start_frame, end_frame, segment_index):
        import scipy.spatial
        import scipy
        from scipy import ndimage

        if not self.is_pushed:
            self._push_to_torch()

        (potrs,) = scipy.linalg.get_lapack_funcs(("potrs",), dtype=np.float32)
        (nrm2,) = scipy.linalg.get_blas_funcs(("nrm2",), dtype=np.float32)

        omp_tol = np.finfo(np.float32).eps
        neighbor_window = self.num_samples - 1

        if isinstance(self.amplitudes, list):
            min_amplitude, max_amplitude = self.amplitudes
        else:
            min_amplitude, max_amplitude = self.amplitudes[:, 0], self.amplitudes[:, 1]
            min_amplitude = min_amplitude[:, np.newaxis]
            max_amplitude = max_amplitude[:, np.newaxis]

        if self.engine == "torch":
            blank = np.zeros((neighbor_window, self.num_channels), dtype=np.float32)
            traces = np.vstack((blank, traces, blank))
            num_timesteps = traces.shape[0]
            torch_traces = torch.as_tensor(traces.T[np.newaxis, :, :], device=self.torch_device)
            num_templates, num_channels = self.temporal.shape[0], self.temporal.shape[1]
            spatially_filtered_data = torch.matmul(self.spatial, torch_traces)
            scaled_filtered_data = (spatially_filtered_data * self.singular).swapaxes(0, 1)
            scaled_filtered_data_ = scaled_filtered_data.reshape(1, num_templates * num_channels, num_timesteps)
            scalar_products = conv1d(scaled_filtered_data_, self.temporal, groups=num_templates, padding="valid")
            scalar_products = scalar_products.cpu().numpy()[0, :, self.num_samples - 1 : -neighbor_window]
        else:
            num_timesteps = traces.shape[0]
            num_peaks = num_timesteps - neighbor_window
            conv_shape = (self.num_templates, num_peaks)
            scalar_products = np.zeros(conv_shape, dtype=np.float32)
            # Filter using overlap-and-add convolution
            spatially_filtered_data = np.matmul(self.spatial, traces.T[np.newaxis, :, :])
            scaled_filtered_data = spatially_filtered_data * self.singular
            from scipy import signal

            objective_by_rank = signal.oaconvolve(scaled_filtered_data, self.temporal, axes=2, mode="valid")
            scalar_products += np.sum(objective_by_rank, axis=0)

        num_peaks = scalar_products.shape[1]

        # Filter using overlap-and-add convolution
        if len(self.ignore_inds) > 0:
            scalar_products[self.ignore_inds] = -np.inf
            not_ignored = ~np.isin(np.arange(self.num_templates), self.ignore_inds)

        num_spikes = 0

        spikes = np.empty(scalar_products.size, dtype=spike_dtype)

        M = np.zeros((self.num_templates, self.num_templates), dtype=np.float32)

        all_selections = np.empty((2, scalar_products.size), dtype=np.int32)
        final_amplitudes = np.zeros(scalar_products.shape, dtype=np.float32)
        num_selection = 0

        full_sps = scalar_products.copy()

        all_amplitudes = np.zeros(0, dtype=np.float32)
        is_in_vicinity = np.zeros(0, dtype=np.int32)

        if self.stop_criteria == "omp_min_sps":
            stop_criteria = self.omp_min_sps * np.maximum(self.norms, np.sqrt(self.num_channels * self.num_samples))
        elif self.stop_criteria == "max_failures":
            num_valids = 0
            nb_failures = self.max_failures
        elif self.stop_criteria == "relative_error":
            if len(self.ignore_inds) > 0:
                new_error = np.linalg.norm(scalar_products[not_ignored])
            else:
                new_error = np.linalg.norm(scalar_products)
            delta_error = np.inf

        do_loop = True

        while do_loop:

            best_cluster_inds = np.argmax(scalar_products, axis=0, keepdims=True)
            products = np.take_along_axis(scalar_products, best_cluster_inds, axis=0)

            result = ndimage.maximum_filter(products[0], size=self.vicinity, mode="constant", cval=0)
            cond_1 = products[0] / self.norms[best_cluster_inds[0]] > 0.25
            cond_2 = np.abs(products[0] - result) < 1e-9
            peak_indices = np.flatnonzero(cond_1 * cond_2)

            if len(peak_indices) == 0:
                break

            for peak_index in peak_indices:

                best_cluster_ind = best_cluster_inds[0, peak_index]

                if num_selection > 0:
                    delta_t = selection[1] - peak_index
                    idx = np.flatnonzero((delta_t < self.num_samples) & (delta_t > -self.num_samples))
                    myline = neighbor_window + delta_t[idx]
                    myindices = selection[0, idx]

                    local_overlaps = self.overlaps[best_cluster_ind]
                    overlapping_templates = self.unit_overlaps_indices[best_cluster_ind]
                    table = self.unit_overlaps_tables[best_cluster_ind]

                    if num_selection == M.shape[0]:
                        Z = np.zeros((2 * num_selection, 2 * num_selection), dtype=np.float32)
                        Z[:num_selection, :num_selection] = M
                        M = Z

                    mask = np.isin(myindices, overlapping_templates)
                    a, b = myindices[mask], myline[mask]
                    M[num_selection, idx[mask]] = local_overlaps[table[a], b]

                    if self.vicinity == 0:
                        scipy.linalg.solve_triangular(
                            M[:num_selection, :num_selection],
                            M[num_selection, :num_selection],
                            trans=0,
                            lower=1,
                            overwrite_b=True,
                            check_finite=False,
                        )

                        v = nrm2(M[num_selection, :num_selection]) ** 2
                        Lkk = 1 - v
                        if Lkk <= omp_tol:  # selected atoms are dependent
                            break
                        M[num_selection, num_selection] = np.sqrt(Lkk)
                    else:
                        is_in_vicinity = np.flatnonzero(np.abs(delta_t) < self.vicinity)

                        if len(is_in_vicinity) > 0:
                            L = M[is_in_vicinity, :][:, is_in_vicinity]

                            M[num_selection, is_in_vicinity] = scipy.linalg.solve_triangular(
                                L,
                                M[num_selection, is_in_vicinity],
                                trans=0,
                                lower=1,
                                overwrite_b=True,
                                check_finite=False,
                            )

                            v = nrm2(M[num_selection, is_in_vicinity]) ** 2
                            Lkk = 1 - v
                            if Lkk <= omp_tol:  # selected atoms are dependent
                                break
                            M[num_selection, num_selection] = np.sqrt(Lkk)
                        else:
                            M[num_selection, num_selection] = 1.0
                else:
                    M[0, 0] = 1

                all_selections[:, num_selection] = [best_cluster_ind, peak_index]
                num_selection += 1

                selection = all_selections[:, :num_selection]
                res_sps = full_sps[selection[0], selection[1]]

                if self.vicinity == 0:
                    new_amplitudes, _ = potrs(M[:num_selection, :num_selection], res_sps, lower=True, overwrite_b=False)
                    sub_selection = selection
                    new_amplitudes /= self.norms[sub_selection[0]]
                else:
                    is_in_vicinity = np.append(is_in_vicinity, num_selection - 1)
                    all_amplitudes = np.append(all_amplitudes, np.float32(1))
                    L = M[is_in_vicinity, :][:, is_in_vicinity]
                    new_amplitudes, _ = potrs(L, res_sps[is_in_vicinity], lower=True, overwrite_b=False)
                    sub_selection = selection[:, is_in_vicinity]
                    new_amplitudes /= self.norms[sub_selection[0]]

                diff_amplitudes = new_amplitudes - final_amplitudes[sub_selection[0], sub_selection[1]]
                modified = np.flatnonzero(np.abs(diff_amplitudes) > omp_tol)
                final_amplitudes[sub_selection[0], sub_selection[1]] = new_amplitudes

                for i in modified:
                    tmp_best, tmp_peak = sub_selection[:, i]
                    diff_amp = diff_amplitudes[i] * self.norms[tmp_best]
                    local_overlaps = self.overlaps[tmp_best]
                    overlapping_templates = self.units_overlaps[tmp_best]
                    tmp = tmp_peak - neighbor_window
                    idx = [max(0, tmp), min(num_peaks, tmp_peak + self.num_samples)]
                    tdx = [idx[0] - tmp, idx[1] - tmp]
                    to_add = diff_amp * local_overlaps[:, tdx[0] : tdx[1]]
                    scalar_products[overlapping_templates, idx[0] : idx[1]] -= to_add

            # We stop when updates do not modify the chosen spikes anymore
            if self.stop_criteria == "omp_min_sps":
                is_valid = scalar_products > stop_criteria[:, np.newaxis]
                do_loop = np.any(is_valid)
            elif self.stop_criteria == "max_failures":
                is_valid = (final_amplitudes > min_amplitude) * (final_amplitudes < max_amplitude)
                new_num_valids = np.sum(is_valid)
                if (new_num_valids - num_valids) > 0:
                    nb_failures = self.max_failures
                else:
                    nb_failures -= 1
                num_valids = new_num_valids
                do_loop = nb_failures > 0
            elif self.stop_criteria == "relative_error":
                previous_error = new_error
                if len(self.ignore_inds) > 0:
                    new_error = np.linalg.norm(scalar_products[not_ignored])
                else:
                    new_error = np.linalg.norm(scalar_products)
                delta_error = np.abs(new_error / previous_error - 1)
                do_loop = delta_error > self.relative_error

        is_valid = (final_amplitudes > min_amplitude) * (final_amplitudes < max_amplitude)
        valid_indices = np.where(is_valid)

        num_spikes = len(valid_indices[0])
        spikes["sample_index"][:num_spikes] = valid_indices[1] + self.nbefore
        spikes["channel_index"][:num_spikes] = 0
        spikes["cluster_index"][:num_spikes] = valid_indices[0]
        spikes["amplitude"][:num_spikes] = final_amplitudes[valid_indices[0], valid_indices[1]]

        spikes = spikes[:num_spikes]
        order = np.argsort(spikes["sample_index"])
        spikes = spikes[order]

        return spikes


class CircusPeeler(BaseTemplateMatching):
    """
    Greedy Template-matching ported from the Spyking Circus sorter

    https://elifesciences.org/articles/34518

    This is a Greedy Template Matching algorithm. The idea is to detect
    all the peaks (negative, positive or both) above a certain threshold
    Then, at every peak (plus or minus some jitter) we look if the signal
    can be explained with a scaled template.
    The amplitudes allowed, for every templates, are automatically adjusted
    in an optimal manner, to enhance the Matthew Correlation Coefficient
    between all spikes/templates in the waveformextractor. For speed and
    memory optimization, templates are automatically sparsified if the
    density of the matrix falls below a given threshold

    Parameters
    ----------
    peak_sign : str
        Sign of the peak (neg, pos, or both)
    exclude_sweep_ms : float
        The number of samples before/after to classify a peak (should be low)
    jitter : int
        The number of samples considered before/after every peak to search for
        matches
    detect_threshold : int
        The detection threshold
    noise_levels : array
        The noise levels, for every channels
    random_chunk_kwargs : dict
        Parameters for computing noise levels, if not provided (sub optimal)
    max_amplitude : float
        Maximal amplitude allowed for every template
    min_amplitude : float
        Minimal amplitude allowed for every template
    use_sparse_matrix_threshold : float
        If density of the templates is below a given threshold, sparse matrix
        are used (memory efficient)
    sparse_kwargs : dict
        Parameters to extract a sparsity mask from the waveform_extractor, if not
        already sparse.
    -----


    """

    def __init__(
        self,
        recording,
        return_output=True,
        parents=None,
        templates=None,
        peak_sign="neg",
        exclude_sweep_ms=0.1,
        jitter_ms=0.1,
        detect_threshold=5,
        noise_levels=None,
        random_chunk_kwargs={},
        max_amplitude=1.5,
        min_amplitude=0.5,
        use_sparse_matrix_threshold=0.25,
    ):

        BaseTemplateMatching.__init__(self, recording, templates, return_output=True, parents=None)

        try:
            from sklearn.feature_extraction.image import extract_patches_2d

            HAVE_SKLEARN = True
        except ImportError:
            HAVE_SKLEARN = False

        assert HAVE_SKLEARN, "CircusPeeler needs sklearn to work"

        assert (use_sparse_matrix_threshold >= 0) and (
            use_sparse_matrix_threshold <= 1
        ), f"use_sparse_matrix_threshold should be in [0, 1]"

        self.num_channels = recording.get_num_channels()
        self.num_samples = templates.num_samples
        self.num_templates = len(templates.unit_ids)

        if noise_levels is None:
            print("CircusPeeler : noise should be computed outside")
            noise_levels = get_noise_levels(recording, **d["random_chunk_kwargs"], return_scaled=False)

        self.abs_threholds = noise_levels * detect_threshold

        self.use_sparse_matrix_threshold = use_sparse_matrix_threshold
        self._prepare_templates()
        self.overlaps = compute_overlaps(
            self.normed_templates,
            self.num_samples,
            self.num_channels,
            self.sparsities,
        )

        self.exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.0)

        self.nbefore = templates.nbefore
        self.nafter = templates.nafter
        self.patch_sizes = (templates.num_samples, self.num_channels)
        self.sym_patch = self.nbefore == self.nafter
        self.jitter = int(jitter_ms * recording.get_sampling_frequency() / 1000.0)

        self.amplitudes = np.zeros((self.num_templates, 2), dtype=np.float32)
        self.amplitudes[:, 0] = min_amplitude
        self.amplitudes[:, 1] = max_amplitude

        self.margin = max(self.nbefore, self.nafter) * 2
        self.peak_sign = peak_sign

    def _prepare_templates(self):
        import scipy.spatial
        import scipy

        self.norms = np.zeros(self.num_templates, dtype=np.float32)

        all_units = self.templates.unit_ids

        sparsity = self.templates.sparsity.mask

        templates_array = self.templates.get_dense_templates()
        self.sparsities = {}
        self.normed_templates = {}

        for count, unit_id in enumerate(all_units):
            self.sparsities[count] = np.flatnonzero(sparsity[count])
            self.norms[count] = np.linalg.norm(templates_array[count])
            templates_array[count] /= self.norms[count]
            self.normed_templates[count] = templates_array[count][:, sparsity[count]]

        templates_array = templates_array.reshape(self.num_templates, -1)

        nnz = np.sum(templates_array != 0) / (self.num_templates * self.num_samples * self.num_channels)
        if nnz <= self.use_sparse_matrix_threshold:
            templates_array = scipy.sparse.csr_matrix(templates_array)
            print(f"Templates are automatically sparsified (sparsity level is {nnz})")
            self.is_dense = False
        else:
            self.is_dense = True

        self.circus_templates = templates_array

    def get_trace_margin(self):
        return self.margin

    def compute_matching(self, traces, start_frame, end_frame, segment_index):

        neighbor_window = self.num_samples - 1

        peak_traces = traces[self.margin // 2 : -self.margin // 2, :]
        peak_sample_index, peak_chan_ind = DetectPeakByChannel.detect_peaks(
            peak_traces, self.peak_sign, self.abs_threholds, self.exclude_sweep_size
        )
        from sklearn.feature_extraction.image import extract_patches_2d

        if self.jitter > 0:
            jittered_peaks = peak_sample_index[:, np.newaxis] + np.arange(-self.jitter, self.jitter)
            jittered_channels = peak_chan_ind[:, np.newaxis] + np.zeros(2 * self.jitter)
            mask = (jittered_peaks > 0) & (jittered_peaks < len(peak_traces))
            jittered_peaks = jittered_peaks[mask]
            jittered_channels = jittered_channels[mask]
            peak_sample_index, unique_idx = np.unique(jittered_peaks, return_index=True)
            peak_chan_ind = jittered_channels[unique_idx]
        else:
            peak_sample_index, unique_idx = np.unique(peak_sample_index, return_index=True)
            peak_chan_ind = peak_chan_ind[unique_idx]

        num_peaks = len(peak_sample_index)

        if self.sym_patch:
            snippets = extract_patches_2d(traces, self.patch_sizes)[peak_sample_index]
            peak_sample_index += self.margin // 2
        else:
            peak_sample_index += self.margin // 2
            snippet_window = np.arange(-self.nbefore, self.nafter)
            snippets = traces[peak_sample_index[:, np.newaxis] + snippet_window]

        if num_peaks > 0:
            snippets = snippets.reshape(num_peaks, -1)
            scalar_products = self.circus_templates.dot(snippets.T)
        else:
            scalar_products = np.zeros((self.num_templates, 0), dtype=np.float32)

        num_spikes = 0
        spikes = np.empty(scalar_products.size, dtype=spike_dtype)
        idx_lookup = np.arange(scalar_products.size).reshape(self.num_templates, -1)

        min_sps = (self.amplitudes[:, 0] * self.norms)[:, np.newaxis]
        max_sps = (self.amplitudes[:, 1] * self.norms)[:, np.newaxis]

        is_valid = (scalar_products > min_sps) & (scalar_products < max_sps)

        cached_overlaps = {}

        while np.any(is_valid):
            best_amplitude_ind = scalar_products[is_valid].argmax()
            best_cluster_ind, peak_index = np.unravel_index(idx_lookup[is_valid][best_amplitude_ind], idx_lookup.shape)

            best_amplitude = scalar_products[best_cluster_ind, peak_index]
            best_peak_sample_index = peak_sample_index[peak_index]
            best_peak_chan_ind = peak_chan_ind[peak_index]

            peak_data = peak_sample_index - peak_sample_index[peak_index]
            is_valid_nn = np.searchsorted(peak_data, [-neighbor_window, neighbor_window + 1])
            idx_neighbor = peak_data[is_valid_nn[0] : is_valid_nn[1]] + neighbor_window

            if not best_cluster_ind in cached_overlaps.keys():
                cached_overlaps[best_cluster_ind] = self.overlaps[best_cluster_ind].toarray()

            to_add = -best_amplitude * cached_overlaps[best_cluster_ind][:, idx_neighbor]

            scalar_products[:, is_valid_nn[0] : is_valid_nn[1]] += to_add
            scalar_products[best_cluster_ind, is_valid_nn[0] : is_valid_nn[1]] = -np.inf

            spikes["sample_index"][num_spikes] = best_peak_sample_index
            spikes["channel_index"][num_spikes] = best_peak_chan_ind
            spikes["cluster_index"][num_spikes] = best_cluster_ind
            spikes["amplitude"][num_spikes] = best_amplitude
            num_spikes += 1

            is_valid = (scalar_products > min_sps) & (scalar_products < max_sps)

        spikes["amplitude"][:num_spikes] /= self.norms[spikes["cluster_index"][:num_spikes]]

        spikes = spikes[:num_spikes]
        order = np.argsort(spikes["sample_index"])
        spikes = spikes[order]

        return spikes
