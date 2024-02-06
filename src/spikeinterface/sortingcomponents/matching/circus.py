"""Sorting components: template matching."""

from __future__ import annotations


import numpy as np
import warnings

import scipy.spatial

import scipy

try:
    import sklearn
    from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


from spikeinterface.core import get_noise_levels, get_random_data_chunks, compute_sparsity
from spikeinterface.sortingcomponents.peak_detection import DetectPeakByChannel

(potrs,) = scipy.linalg.get_lapack_funcs(("potrs",), dtype=np.float32)

(nrm2,) = scipy.linalg.get_blas_funcs(("nrm2",), dtype=np.float32)

spike_dtype = [
    ("sample_index", "int64"),
    ("channel_index", "int64"),
    ("cluster_index", "int64"),
    ("amplitude", "float64"),
    ("segment_index", "int64"),
]

from .main import BaseTemplateMatchingEngine


from scipy.fft._helper import _init_nd_shape_and_axes

try:
    from scipy.signal.signaltools import _init_freq_conv_axes, _apply_conv_mode
except Exception:
    from scipy.signal._signaltools import _init_freq_conv_axes, _apply_conv_mode
from scipy import linalg, fft as sp_fft


def get_scipy_shape(in1, in2, mode="full", axes=None, calc_fast_len=True):
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])

    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False)

    s1 = in1.shape
    s2 = in2.shape

    shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1 for i in range(in1.ndim)]

    if not len(axes):
        return in1 * in2

    complex_result = in1.dtype.kind == "c" or in2.dtype.kind == "c"

    if calc_fast_len:
        # Speed up FFT by padding to optimal size.
        fshape = [sp_fft.next_fast_len(shape[a], not complex_result) for a in axes]
    else:
        fshape = shape

    return fshape, axes


def fftconvolve_with_cache(in1, in2, cache, mode="full", axes=None):
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])

    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False)

    s1 = in1.shape
    s2 = in2.shape

    shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1 for i in range(in1.ndim)]

    ret = _freq_domain_conv(in1, in2, axes, shape, cache, calc_fast_len=True)

    return _apply_conv_mode(ret, s1, s2, mode, axes)


def _freq_domain_conv(in1, in2, axes, shape, cache, calc_fast_len=True):
    if not len(axes):
        return in1 * in2

    complex_result = in1.dtype.kind == "c" or in2.dtype.kind == "c"

    if calc_fast_len:
        # Speed up FFT by padding to optimal size.
        fshape = [sp_fft.next_fast_len(shape[a], not complex_result) for a in axes]
    else:
        fshape = shape

    if not complex_result:
        fft, ifft = sp_fft.rfftn, sp_fft.irfftn
    else:
        fft, ifft = sp_fft.fftn, sp_fft.ifftn

    sp1 = cache["full"][cache["mask"]]
    sp2 = cache["template"]

    # sp2 = fft(in2[cache['mask']], fshape, axes=axes)
    ret = ifft(sp1 * sp2, fshape, axes=axes)

    if calc_fast_len:
        fslice = tuple([slice(sz) for sz in shape])
        ret = ret[fslice]

    return ret


def compute_overlaps(templates, num_samples, num_channels, sparsities):
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


class CircusOMPPeeler(BaseTemplateMatchingEngine):
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
    amplitude: tuple
        (Minimal, Maximal) amplitudes allowed for every template
    omp_min_sps: float
        Stopping criteria of the OMP algorithm, in percentage of the norm
    noise_levels: array
        The noise levels, for every channels. If None, they will be automatically
        computed
    random_chunk_kwargs: dict
        Parameters for computing noise levels, if not provided (sub optimal)
    sparse_kwargs: dict
        Parameters to extract a sparsity mask from the waveform_extractor, if not
        already sparse.
    -----
    """

    _default_params = {
        "amplitudes": [0.6, 2],
        "omp_min_sps": 0.1,
        "waveform_extractor": None,
        "templates": None,
        "overlaps": None,
        "norms": None,
        "random_chunk_kwargs": {},
        "noise_levels": None,
        "sparse_kwargs": {"method": "ptp", "threshold": 1},
        "ignored_ids": [],
        "vicinity": 0,
    }

    @classmethod
    def _prepare_templates(cls, d):
        waveform_extractor = d["waveform_extractor"]
        num_templates = len(d["waveform_extractor"].sorting.unit_ids)

        if not waveform_extractor.is_sparse():
            sparsity = compute_sparsity(waveform_extractor, **d["sparse_kwargs"]).mask
        else:
            sparsity = waveform_extractor.sparsity.mask

        templates = waveform_extractor.get_all_templates(mode="median").copy()

        d["sparsities"] = {}
        d["templates"] = {}
        d["norms"] = np.zeros(num_templates, dtype=np.float32)

        for count, unit_id in enumerate(waveform_extractor.sorting.unit_ids):
            template = templates[count][:, sparsity[count]]
            (d["sparsities"][count],) = np.nonzero(sparsity[count])
            d["norms"][count] = np.linalg.norm(template)
            d["templates"][count] = template / d["norms"][count]

        return d

    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):
        d = cls._default_params.copy()
        d.update(kwargs)

        # assert isinstance(d['waveform_extractor'], WaveformExtractor)

        for v in ["omp_min_sps"]:
            assert (d[v] >= 0) and (d[v] <= 1), f"{v} should be in [0, 1]"

        d["num_channels"] = d["waveform_extractor"].recording.get_num_channels()
        d["num_samples"] = d["waveform_extractor"].nsamples
        d["nbefore"] = d["waveform_extractor"].nbefore
        d["nafter"] = d["waveform_extractor"].nafter
        d["sampling_frequency"] = d["waveform_extractor"].recording.get_sampling_frequency()
        d["vicinity"] *= d["num_samples"]

        if d["noise_levels"] is None:
            print("CircusOMPPeeler : noise should be computed outside")
            d["noise_levels"] = get_noise_levels(recording, **d["random_chunk_kwargs"], return_scaled=False)

        if d["templates"] is None:
            d = cls._prepare_templates(d)
        else:
            for key in ["norms", "sparsities"]:
                assert d[key] is not None, "If templates are provided, %d should also be there" % key

        d["num_templates"] = len(d["templates"])

        if d["overlaps"] is None:
            d["overlaps"] = compute_overlaps(d["templates"], d["num_samples"], d["num_channels"], d["sparsities"])

        d["ignored_ids"] = np.array(d["ignored_ids"])

        omp_min_sps = d["omp_min_sps"]
        # nb_active_channels = np.array([len(d['sparsities'][count]) for count in range(d['num_templates'])])
        d["stop_criteria"] = omp_min_sps * np.sqrt(d["noise_levels"].sum() * d["num_samples"])

        return d

    @classmethod
    def serialize_method_kwargs(cls, kwargs):
        kwargs = dict(kwargs)
        # remove waveform_extractor
        kwargs.pop("waveform_extractor")
        return kwargs

    @classmethod
    def unserialize_in_worker(cls, kwargs):
        return kwargs

    @classmethod
    def get_margin(cls, recording, kwargs):
        margin = 2 * max(kwargs["nbefore"], kwargs["nafter"])
        return margin

    @classmethod
    def main_function(cls, traces, d):
        templates = d["templates"]
        num_templates = d["num_templates"]
        num_channels = d["num_channels"]
        num_samples = d["num_samples"]
        overlaps = d["overlaps"]
        norms = d["norms"]
        nbefore = d["nbefore"]
        nafter = d["nafter"]
        omp_tol = np.finfo(np.float32).eps
        num_samples = d["nafter"] + d["nbefore"]
        neighbor_window = num_samples - 1
        min_amplitude, max_amplitude = d["amplitudes"]
        sparsities = d["sparsities"]
        ignored_ids = d["ignored_ids"]
        stop_criteria = d["stop_criteria"]
        vicinity = d["vicinity"]

        if "cached_fft_kernels" not in d:
            d["cached_fft_kernels"] = {"fshape": 0}

        cached_fft_kernels = d["cached_fft_kernels"]

        num_timesteps = len(traces)

        num_peaks = num_timesteps - num_samples + 1

        traces = traces.T

        dummy_filter = np.empty((num_channels, num_samples), dtype=np.float32)
        dummy_traces = np.empty((num_channels, num_timesteps), dtype=np.float32)

        fshape, axes = get_scipy_shape(dummy_filter, traces, axes=1)
        fft_cache = {"full": sp_fft.rfftn(traces, fshape, axes=axes)}

        scalar_products = np.empty((num_templates, num_peaks), dtype=np.float32)

        flagged_chunk = cached_fft_kernels["fshape"] != fshape[0]

        for i in range(num_templates):
            if i not in ignored_ids:
                if i not in cached_fft_kernels or flagged_chunk:
                    kernel_filter = np.ascontiguousarray(templates[i][::-1].T)
                    cached_fft_kernels.update({i: sp_fft.rfftn(kernel_filter, fshape, axes=axes)})
                    cached_fft_kernels["fshape"] = fshape[0]

                fft_cache.update({"mask": sparsities[i], "template": cached_fft_kernels[i]})

                convolution = fftconvolve_with_cache(dummy_filter, dummy_traces, fft_cache, axes=1, mode="valid")
                if len(convolution) > 0:
                    scalar_products[i] = convolution.sum(0)
                else:
                    scalar_products[i] = 0

        if len(ignored_ids) > 0:
            scalar_products[ignored_ids] = -np.inf

        num_spikes = 0

        spikes = np.empty(scalar_products.size, dtype=spike_dtype)
        idx_lookup = np.arange(scalar_products.size).reshape(num_templates, -1)

        M = np.zeros((100, 100), dtype=np.float32)

        all_selections = np.empty((2, scalar_products.size), dtype=np.int32)
        final_amplitudes = np.zeros(scalar_products.shape, dtype=np.float32)
        num_selection = 0

        full_sps = scalar_products.copy()

        neighbors = {}
        cached_overlaps = {}

        is_valid = scalar_products > stop_criteria
        all_amplitudes = np.zeros(0, dtype=np.float32)
        is_in_vicinity = np.zeros(0, dtype=np.int32)

        while np.any(is_valid):
            best_amplitude_ind = scalar_products[is_valid].argmax()
            best_cluster_ind, peak_index = np.unravel_index(idx_lookup[is_valid][best_amplitude_ind], idx_lookup.shape)

            if num_selection > 0:
                delta_t = selection[1] - peak_index
                idx = np.where((delta_t < neighbor_window) & (delta_t > -num_samples))[0]
                myline = num_samples + delta_t[idx]

                if not best_cluster_ind in cached_overlaps:
                    cached_overlaps[best_cluster_ind] = overlaps[best_cluster_ind].toarray()

                if num_selection == M.shape[0]:
                    Z = np.zeros((2 * num_selection, 2 * num_selection), dtype=np.float32)
                    Z[:num_selection, :num_selection] = M
                    M = Z

                M[num_selection, idx] = cached_overlaps[best_cluster_ind][selection[0, idx], myline]

                if vicinity == 0:
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
                    is_in_vicinity = np.where(np.abs(delta_t) < vicinity)[0]

                    if len(is_in_vicinity) > 0:
                        L = M[is_in_vicinity, :][:, is_in_vicinity]

                        M[num_selection, is_in_vicinity] = scipy.linalg.solve_triangular(
                            L, M[num_selection, is_in_vicinity], trans=0, lower=1, overwrite_b=True, check_finite=False
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

            if True:  # vicinity == 0:
                all_amplitudes, _ = potrs(M[:num_selection, :num_selection], res_sps, lower=True, overwrite_b=False)
                all_amplitudes /= norms[selection[0]]
            else:
                # This is not working, need to figure out why
                is_in_vicinity = np.append(is_in_vicinity, num_selection - 1)
                all_amplitudes = np.append(all_amplitudes, np.float32(1))
                L = M[is_in_vicinity, :][:, is_in_vicinity]
                all_amplitudes[is_in_vicinity], _ = potrs(L, res_sps[is_in_vicinity], lower=True, overwrite_b=False)
                all_amplitudes[is_in_vicinity] /= norms[selection[0][is_in_vicinity]]

            diff_amplitudes = all_amplitudes - final_amplitudes[selection[0], selection[1]]
            modified = np.where(np.abs(diff_amplitudes) > omp_tol)[0]
            final_amplitudes[selection[0], selection[1]] = all_amplitudes

            for i in modified:
                tmp_best, tmp_peak = selection[:, i]
                diff_amp = diff_amplitudes[i] * norms[tmp_best]

                if not tmp_best in cached_overlaps:
                    cached_overlaps[tmp_best] = overlaps[tmp_best].toarray()

                if not tmp_peak in neighbors.keys():
                    idx = [max(0, tmp_peak - num_samples), min(num_peaks, tmp_peak + neighbor_window)]
                    tdx = [num_samples + idx[0] - tmp_peak, num_samples + idx[1] - tmp_peak]
                    neighbors[tmp_peak] = {"idx": idx, "tdx": tdx}

                idx = neighbors[tmp_peak]["idx"]
                tdx = neighbors[tmp_peak]["tdx"]

                to_add = diff_amp * cached_overlaps[tmp_best][:, tdx[0] : tdx[1]]
                scalar_products[:, idx[0] : idx[1]] -= to_add

            is_valid = scalar_products > stop_criteria

        is_valid = (final_amplitudes > min_amplitude) * (final_amplitudes < max_amplitude)
        valid_indices = np.where(is_valid)

        num_spikes = len(valid_indices[0])
        spikes["sample_index"][:num_spikes] = valid_indices[1] + d["nbefore"]
        spikes["channel_index"][:num_spikes] = 0
        spikes["cluster_index"][:num_spikes] = valid_indices[0]
        spikes["amplitude"][:num_spikes] = final_amplitudes[valid_indices[0], valid_indices[1]]

        spikes = spikes[:num_spikes]
        order = np.argsort(spikes["sample_index"])
        spikes = spikes[order]

        return spikes


class CircusOMPSVDPeeler(BaseTemplateMatchingEngine):
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
    amplitude: tuple
        (Minimal, Maximal) amplitudes allowed for every template
    max_failures: int
        Stopping criteria of the OMP algorithm, as number of retry while updating amplitudes
    sparse_kwargs: dict
        Parameters to extract a sparsity mask from the waveform_extractor, if not
        already sparse.
    rank: int, default: 5
        Number of components used internally by the SVD
    vicinity: int
        Size of the area surrounding a spike to perform modification (expressed in terms
        of template temporal width)
    -----
    """

    _default_params = {
        "amplitudes": [0.6, 2],
        "stop_criteria": "max_failures",
        "max_failures": 20,
        "omp_min_sps": 0.1,
        "relative_error": 5e-5,
        "waveform_extractor": None,
        "rank": 5,
        "sparse_kwargs": {"method": "ptp", "threshold": 1},
        "ignored_ids": [],
        "vicinity": 0,
        "optimize_amplitudes": False,
    }

    @classmethod
    def _prepare_templates(cls, d):
        waveform_extractor = d["waveform_extractor"]
        num_templates = len(d["waveform_extractor"].sorting.unit_ids)

        assert d["stop_criteria"] in ["max_failures", "omp_min_sps", "relative_error"]

        if not waveform_extractor.is_sparse():
            sparsity = compute_sparsity(waveform_extractor, **d["sparse_kwargs"]).mask
        else:
            sparsity = waveform_extractor.sparsity.mask

        d["sparsity_mask"] = sparsity
        units_overlaps = np.sum(np.logical_and(sparsity[:, np.newaxis, :], sparsity[np.newaxis, :, :]), axis=2)
        d["units_overlaps"] = units_overlaps > 0
        d["unit_overlaps_indices"] = {}
        for i in range(num_templates):
            (d["unit_overlaps_indices"][i],) = np.nonzero(d["units_overlaps"][i])

        templates = waveform_extractor.get_all_templates(mode="median").copy()

        # First, we set masked channels to 0
        for count in range(num_templates):
            templates[count][:, ~d["sparsity_mask"][count]] = 0

        # Then we keep only the strongest components
        rank = d["rank"]
        temporal, singular, spatial = np.linalg.svd(templates, full_matrices=False)
        d["temporal"] = temporal[:, :, :rank]
        d["singular"] = singular[:, :rank]
        d["spatial"] = spatial[:, :rank, :]

        # We reconstruct the approximated templates
        templates = np.matmul(d["temporal"] * d["singular"][:, np.newaxis, :], d["spatial"])

        d["templates"] = np.zeros(templates.shape, dtype=np.float32)
        d["norms"] = np.zeros(num_templates, dtype=np.float32)

        # And get the norms, saving compressed templates for CC matrix
        for count in range(num_templates):
            template = templates[count][:, d["sparsity_mask"][count]]
            d["norms"][count] = np.linalg.norm(template)
            d["templates"][count][:, d["sparsity_mask"][count]] = template / d["norms"][count]

        if d["optimize_amplitudes"]:
            noise = np.random.randn(200, d["num_samples"] * d["num_channels"])
            r = d["templates"].reshape(num_templates, -1).dot(noise.reshape(len(noise), -1).T)
            s = r / d["norms"][:, np.newaxis]
            mad = np.median(np.abs(s - np.median(s, 1)[:, np.newaxis]), 1)
            a_min = np.median(s, 1) + 5 * mad

            means = np.zeros((num_templates, num_templates), dtype=np.float32)
            stds = np.zeros((num_templates, num_templates), dtype=np.float32)
            for count, unit_id in enumerate(waveform_extractor.unit_ids):
                w = waveform_extractor.get_waveforms(unit_id, force_dense=True)
                r = d["templates"].reshape(num_templates, -1).dot(w.reshape(len(w), -1).T)
                s = r / d["norms"][:, np.newaxis]
                means[count] = np.median(s, 1)
                stds[count] = np.median(np.abs(s - np.median(s, 1)[:, np.newaxis]), 1)

            _, a_max = d["amplitudes"]
            d["amplitudes"] = np.zeros((num_templates, 2), dtype=np.float32)

            for count in range(num_templates):
                indices = np.argsort(means[count])
                a = np.where(indices == count)[0][0]
                d["amplitudes"][count][1] = 1 + 5 * stds[count, indices[a]]
                d["amplitudes"][count][0] = max(a_min[count], 1 - 5 * stds[count, indices[a]])

        d["temporal"] /= d["norms"][:, np.newaxis, np.newaxis]
        d["temporal"] = np.flip(d["temporal"], axis=1)

        d["overlaps"] = []
        for i in range(num_templates):
            num_overlaps = np.sum(d["units_overlaps"][i])
            overlapping_units = np.where(d["units_overlaps"][i])[0]

            # Reconstruct unit template from SVD Matrices
            data = d["temporal"][i] * d["singular"][i][np.newaxis, :]
            template_i = np.matmul(data, d["spatial"][i, :, :])
            template_i = np.flipud(template_i)

            unit_overlaps = np.zeros([num_overlaps, 2 * d["num_samples"] - 1], dtype=np.float32)

            for count, j in enumerate(overlapping_units):
                overlapped_channels = d["sparsity_mask"][j]
                visible_i = template_i[:, overlapped_channels]

                spatial_filters = d["spatial"][j, :, overlapped_channels]
                spatially_filtered_template = np.matmul(visible_i, spatial_filters)
                visible_i = spatially_filtered_template * d["singular"][j]

                for rank in range(visible_i.shape[1]):
                    unit_overlaps[count, :] += np.convolve(visible_i[:, rank], d["temporal"][j][:, rank], mode="full")

            d["overlaps"].append(unit_overlaps)

        d["spatial"] = np.moveaxis(d["spatial"], [0, 1, 2], [1, 0, 2])
        d["temporal"] = np.moveaxis(d["temporal"], [0, 1, 2], [1, 2, 0])
        d["singular"] = d["singular"].T[:, :, np.newaxis]

        return d

    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):
        d = cls._default_params.copy()
        d.update(kwargs)

        d["num_channels"] = d["waveform_extractor"].recording.get_num_channels()
        d["num_samples"] = d["waveform_extractor"].nsamples
        d["nbefore"] = d["waveform_extractor"].nbefore
        d["nafter"] = d["waveform_extractor"].nafter
        d["sampling_frequency"] = d["waveform_extractor"].recording.get_sampling_frequency()
        d["vicinity"] *= d["num_samples"]

        if "templates" not in d:
            d = cls._prepare_templates(d)
        else:
            for key in [
                "norms",
                "temporal",
                "spatial",
                "singular",
                "units_overlaps",
                "sparsity_mask",
                "unit_overlaps_indices",
            ]:
                assert d[key] is not None, "If templates are provided, %d should also be there" % key

        d["num_templates"] = len(d["templates"])
        d["ignored_ids"] = np.array(d["ignored_ids"])

        d["unit_overlaps_tables"] = {}
        for i in range(d["num_templates"]):
            d["unit_overlaps_tables"][i] = np.zeros(d["num_templates"], dtype=int)
            d["unit_overlaps_tables"][i][d["unit_overlaps_indices"][i]] = np.arange(len(d["unit_overlaps_indices"][i]))

        return d

    @classmethod
    def serialize_method_kwargs(cls, kwargs):
        kwargs = dict(kwargs)
        # remove waveform_extractor
        kwargs.pop("waveform_extractor")
        return kwargs

    @classmethod
    def unserialize_in_worker(cls, kwargs):
        return kwargs

    @classmethod
    def get_margin(cls, recording, kwargs):
        margin = 2 * max(kwargs["nbefore"], kwargs["nafter"])
        return margin

    @classmethod
    def main_function(cls, traces, d):
        templates = d["templates"]
        num_templates = d["num_templates"]
        num_channels = d["num_channels"]
        num_samples = d["num_samples"]
        overlaps = d["overlaps"]
        norms = d["norms"]
        nbefore = d["nbefore"]
        nafter = d["nafter"]
        omp_tol = np.finfo(np.float32).eps
        num_samples = d["nafter"] + d["nbefore"]
        neighbor_window = num_samples - 1
        if d["optimize_amplitudes"]:
            min_amplitude, max_amplitude = d["amplitudes"][:, 0], d["amplitudes"][:, 1]
            min_amplitude = min_amplitude[:, np.newaxis]
            max_amplitude = max_amplitude[:, np.newaxis]
        else:
            min_amplitude, max_amplitude = d["amplitudes"]
        ignored_ids = d["ignored_ids"]
        vicinity = d["vicinity"]
        rank = d["rank"]

        num_timesteps = len(traces)

        num_peaks = num_timesteps - num_samples + 1
        conv_shape = (num_templates, num_peaks)
        scalar_products = np.zeros(conv_shape, dtype=np.float32)

        # Filter using overlap-and-add convolution
        if len(ignored_ids) > 0:
            not_ignored = ~np.isin(np.arange(num_templates), ignored_ids)
            spatially_filtered_data = np.matmul(d["spatial"][:, not_ignored, :], traces.T[np.newaxis, :, :])
            scaled_filtered_data = spatially_filtered_data * d["singular"][:, not_ignored, :]
            objective_by_rank = scipy.signal.oaconvolve(
                scaled_filtered_data, d["temporal"][:, not_ignored, :], axes=2, mode="valid"
            )
            scalar_products[not_ignored] += np.sum(objective_by_rank, axis=0)
            scalar_products[ignored_ids] = -np.inf
        else:
            spatially_filtered_data = np.matmul(d["spatial"], traces.T[np.newaxis, :, :])
            scaled_filtered_data = spatially_filtered_data * d["singular"]
            objective_by_rank = scipy.signal.oaconvolve(scaled_filtered_data, d["temporal"], axes=2, mode="valid")
            scalar_products += np.sum(objective_by_rank, axis=0)

        num_spikes = 0

        spikes = np.empty(scalar_products.size, dtype=spike_dtype)

        M = np.zeros((num_templates, num_templates), dtype=np.float32)

        all_selections = np.empty((2, scalar_products.size), dtype=np.int32)
        final_amplitudes = np.zeros(scalar_products.shape, dtype=np.float32)
        num_selection = 0

        full_sps = scalar_products.copy()

        neighbors = {}
        cached_overlaps = {}

        all_amplitudes = np.zeros(0, dtype=np.float32)
        is_in_vicinity = np.zeros(0, dtype=np.int32)

        if d["stop_criteria"] == "omp_min_sps":
            stop_criteria = d["omp_min_sps"] * np.maximum(d["norms"], np.sqrt(num_channels * num_samples))
        elif d["stop_criteria"] == "max_failures":
            nb_valids = 0
            nb_failures = d["max_failures"]
        elif d["stop_criteria"] == "relative_error":
            if len(ignored_ids) > 0:
                new_error = np.linalg.norm(scalar_products[not_ignored])
            else:
                new_error = np.linalg.norm(scalar_products)
            delta_error = np.inf

        do_loop = True

        while do_loop:
            best_amplitude_ind = scalar_products.argmax()
            best_cluster_ind, peak_index = np.unravel_index(best_amplitude_ind, scalar_products.shape)

            if num_selection > 0:
                delta_t = selection[1] - peak_index
                idx = np.where((delta_t < num_samples) & (delta_t > -num_samples))[0]
                myline = neighbor_window + delta_t[idx]
                myindices = selection[0, idx]

                local_overlaps = overlaps[best_cluster_ind]
                overlapping_templates = d["unit_overlaps_indices"][best_cluster_ind]
                table = d["unit_overlaps_tables"][best_cluster_ind]

                if num_selection == M.shape[0]:
                    Z = np.zeros((2 * num_selection, 2 * num_selection), dtype=np.float32)
                    Z[:num_selection, :num_selection] = M
                    M = Z

                mask = np.isin(myindices, overlapping_templates)
                a, b = myindices[mask], myline[mask]
                M[num_selection, idx[mask]] = local_overlaps[table[a], b]

                if vicinity == 0:
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
                    is_in_vicinity = np.where(np.abs(delta_t) < vicinity)[0]

                    if len(is_in_vicinity) > 0:
                        L = M[is_in_vicinity, :][:, is_in_vicinity]

                        M[num_selection, is_in_vicinity] = scipy.linalg.solve_triangular(
                            L, M[num_selection, is_in_vicinity], trans=0, lower=1, overwrite_b=True, check_finite=False
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

            if True:  # vicinity == 0:
                all_amplitudes, _ = potrs(M[:num_selection, :num_selection], res_sps, lower=True, overwrite_b=False)
                all_amplitudes /= norms[selection[0]]
            else:
                # This is not working, need to figure out why
                is_in_vicinity = np.append(is_in_vicinity, num_selection - 1)
                all_amplitudes = np.append(all_amplitudes, np.float32(1))
                L = M[is_in_vicinity, :][:, is_in_vicinity]
                all_amplitudes[is_in_vicinity], _ = potrs(L, res_sps[is_in_vicinity], lower=True, overwrite_b=False)
                all_amplitudes[is_in_vicinity] /= norms[selection[0][is_in_vicinity]]

            diff_amplitudes = all_amplitudes - final_amplitudes[selection[0], selection[1]]
            modified = np.where(np.abs(diff_amplitudes) > omp_tol)[0]
            final_amplitudes[selection[0], selection[1]] = all_amplitudes

            for i in modified:
                tmp_best, tmp_peak = selection[:, i]
                diff_amp = diff_amplitudes[i] * norms[tmp_best]

                local_overlaps = overlaps[tmp_best]
                overlapping_templates = d["units_overlaps"][tmp_best]

                if not tmp_peak in neighbors.keys():
                    idx = [max(0, tmp_peak - neighbor_window), min(num_peaks, tmp_peak + num_samples)]
                    tdx = [neighbor_window + idx[0] - tmp_peak, num_samples + idx[1] - tmp_peak - 1]
                    neighbors[tmp_peak] = {"idx": idx, "tdx": tdx}

                idx = neighbors[tmp_peak]["idx"]
                tdx = neighbors[tmp_peak]["tdx"]

                to_add = diff_amp * local_overlaps[:, tdx[0] : tdx[1]]
                scalar_products[overlapping_templates, idx[0] : idx[1]] -= to_add

            # We stop when updates do not modify the chosen spikes anymore
            if d["stop_criteria"] == "omp_min_sps":
                is_valid = scalar_products > stop_criteria[:, np.newaxis]
                do_loop = np.any(is_valid)
            elif d["stop_criteria"] == "max_failures":
                is_valid = (final_amplitudes > min_amplitude) * (final_amplitudes < max_amplitude)
                new_nb_valids = np.sum(is_valid)
                if (new_nb_valids - nb_valids) == 0:
                    nb_failures -= 1
                nb_valids = new_nb_valids
                do_loop = nb_failures > 0
            elif d["stop_criteria"] == "relative_error":
                previous_error = new_error
                if len(ignored_ids) > 0:
                    new_error = np.linalg.norm(scalar_products[not_ignored])
                else:
                    new_error = np.linalg.norm(scalar_products)
                delta_error = np.abs(new_error / previous_error - 1)
                do_loop = delta_error > d["relative_error"]

        is_valid = (final_amplitudes > min_amplitude) * (final_amplitudes < max_amplitude)
        valid_indices = np.where(is_valid)

        num_spikes = len(valid_indices[0])
        spikes["sample_index"][:num_spikes] = valid_indices[1] + d["nbefore"]
        spikes["channel_index"][:num_spikes] = 0
        spikes["cluster_index"][:num_spikes] = valid_indices[0]
        spikes["amplitude"][:num_spikes] = final_amplitudes[valid_indices[0], valid_indices[1]]

        spikes = spikes[:num_spikes]
        order = np.argsort(spikes["sample_index"])
        spikes = spikes[order]

        return spikes


class CircusPeeler(BaseTemplateMatchingEngine):
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
    peak_sign: str
        Sign of the peak (neg, pos, or both)
    exclude_sweep_ms: float
        The number of samples before/after to classify a peak (should be low)
    jitter: int
        The number of samples considered before/after every peak to search for
        matches
    detect_threshold: int
        The detection threshold
    noise_levels: array
        The noise levels, for every channels
    random_chunk_kwargs: dict
        Parameters for computing noise levels, if not provided (sub optimal)
    max_amplitude: float
        Maximal amplitude allowed for every template
    min_amplitude: float
        Minimal amplitude allowed for every template
    use_sparse_matrix_threshold: float
        If density of the templates is below a given threshold, sparse matrix
        are used (memory efficient)
    sparse_kwargs: dict
        Parameters to extract a sparsity mask from the waveform_extractor, if not
        already sparse.
    -----


    """

    _default_params = {
        "peak_sign": "neg",
        "exclude_sweep_ms": 0.1,
        "jitter_ms": 0.1,
        "detect_threshold": 5,
        "noise_levels": None,
        "random_chunk_kwargs": {},
        "max_amplitude": 1.5,
        "min_amplitude": 0.5,
        "use_sparse_matrix_threshold": 0.25,
        "waveform_extractor": None,
        "sparse_kwargs": {"method": "ptp", "threshold": 1},
    }

    @classmethod
    def _prepare_templates(cls, d):
        waveform_extractor = d["waveform_extractor"]
        num_samples = d["num_samples"]
        num_channels = d["num_channels"]
        num_templates = d["num_templates"]
        use_sparse_matrix_threshold = d["use_sparse_matrix_threshold"]

        d["norms"] = np.zeros(num_templates, dtype=np.float32)

        all_units = list(d["waveform_extractor"].sorting.unit_ids)

        if not waveform_extractor.is_sparse():
            sparsity = compute_sparsity(waveform_extractor, **d["sparse_kwargs"]).mask
        else:
            sparsity = waveform_extractor.sparsity.mask

        templates = waveform_extractor.get_all_templates(mode="median").copy()
        d["sparsities"] = {}
        d["circus_templates"] = {}

        for count, unit_id in enumerate(all_units):
            (d["sparsities"][count],) = np.nonzero(sparsity[count])
            templates[count][:, ~sparsity[count]] = 0
            d["norms"][count] = np.linalg.norm(templates[count])
            templates[count] /= d["norms"][count]
            d["circus_templates"][count] = templates[count][:, sparsity[count]]

        templates = templates.reshape(num_templates, -1)

        nnz = np.sum(templates != 0) / (num_templates * num_samples * num_channels)
        if nnz <= use_sparse_matrix_threshold:
            templates = scipy.sparse.csr_matrix(templates)
            print(f"Templates are automatically sparsified (sparsity level is {nnz})")
            d["is_dense"] = False
        else:
            d["is_dense"] = True

        d["templates"] = templates

        return d

    @classmethod
    def _mcc_error(cls, bounds, good, bad):
        fn = np.sum((good < bounds[0]) | (good > bounds[1]))
        fp = np.sum((bounds[0] <= bad) & (bad <= bounds[1]))
        tp = np.sum((bounds[0] <= good) & (good <= bounds[1]))
        tn = np.sum((bad < bounds[0]) | (bad > bounds[1]))
        denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if denom > 0:
            mcc = 1 - (tp * tn - fp * fn) / np.sqrt(denom)
        else:
            mcc = 1
        return mcc

    @classmethod
    def _cost_function_mcc(cls, bounds, good, bad, delta_amplitude, alpha):
        # We want a minimal error, with the larger bounds that are possible
        cost = alpha * cls._mcc_error(bounds, good, bad) + (1 - alpha) * np.abs(
            (1 - (bounds[1] - bounds[0]) / delta_amplitude)
        )
        return cost

    @classmethod
    def _optimize_amplitudes(cls, noise_snippets, d):
        parameters = d
        waveform_extractor = parameters["waveform_extractor"]
        templates = parameters["templates"]
        num_templates = parameters["num_templates"]
        max_amplitude = parameters["max_amplitude"]
        min_amplitude = parameters["min_amplitude"]
        alpha = 0.5
        norms = parameters["norms"]
        all_units = list(waveform_extractor.sorting.unit_ids)

        parameters["amplitudes"] = np.zeros((num_templates, 2), dtype=np.float32)
        noise = templates.dot(noise_snippets) / norms[:, np.newaxis]

        all_amps = {}
        for count, unit_id in enumerate(all_units):
            waveform = waveform_extractor.get_waveforms(unit_id, force_dense=True)
            snippets = waveform.reshape(waveform.shape[0], -1).T
            amps = templates.dot(snippets) / norms[:, np.newaxis]
            good = amps[count, :].flatten()

            sub_amps = amps[np.concatenate((np.arange(count), np.arange(count + 1, num_templates))), :]
            bad = sub_amps[sub_amps >= good]
            bad = np.concatenate((bad, noise[count]))
            cost_kwargs = [good, bad, max_amplitude - min_amplitude, alpha]
            cost_bounds = [(min_amplitude, 1), (1, max_amplitude)]
            res = scipy.optimize.differential_evolution(cls._cost_function_mcc, bounds=cost_bounds, args=cost_kwargs)
            parameters["amplitudes"][count] = res.x

        return d

    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):
        assert HAVE_SKLEARN, "CircusPeeler needs sklearn to work"
        default_parameters = cls._default_params.copy()
        default_parameters.update(kwargs)

        # assert isinstance(d['waveform_extractor'], WaveformExtractor)
        for v in ["use_sparse_matrix_threshold"]:
            assert (default_parameters[v] >= 0) and (default_parameters[v] <= 1), f"{v} should be in [0, 1]"

        default_parameters["num_channels"] = default_parameters["waveform_extractor"].recording.get_num_channels()
        default_parameters["num_samples"] = default_parameters["waveform_extractor"].nsamples
        default_parameters["num_templates"] = len(default_parameters["waveform_extractor"].sorting.unit_ids)

        if default_parameters["noise_levels"] is None:
            print("CircusPeeler : noise should be computed outside")
            default_parameters["noise_levels"] = get_noise_levels(
                recording, **default_parameters["random_chunk_kwargs"], return_scaled=False
            )

        default_parameters["abs_threholds"] = (
            default_parameters["noise_levels"] * default_parameters["detect_threshold"]
        )

        default_parameters = cls._prepare_templates(default_parameters)

        default_parameters["overlaps"] = compute_overlaps(
            default_parameters["circus_templates"],
            default_parameters["num_samples"],
            default_parameters["num_channels"],
            default_parameters["sparsities"],
        )

        default_parameters["exclude_sweep_size"] = int(
            default_parameters["exclude_sweep_ms"] * recording.get_sampling_frequency() / 1000.0
        )

        default_parameters["nbefore"] = default_parameters["waveform_extractor"].nbefore
        default_parameters["nafter"] = default_parameters["waveform_extractor"].nafter
        default_parameters["patch_sizes"] = (
            default_parameters["waveform_extractor"].nsamples,
            default_parameters["num_channels"],
        )
        default_parameters["sym_patch"] = default_parameters["nbefore"] == default_parameters["nafter"]
        default_parameters["jitter"] = int(
            default_parameters["jitter_ms"] * recording.get_sampling_frequency() / 1000.0
        )

        num_segments = recording.get_num_segments()
        if default_parameters["waveform_extractor"]._params["max_spikes_per_unit"] is None:
            num_snippets = 1000
        else:
            num_snippets = 2 * default_parameters["waveform_extractor"]._params["max_spikes_per_unit"]

        num_chunks = num_snippets // num_segments
        noise_snippets = get_random_data_chunks(
            recording, num_chunks_per_segment=num_chunks, chunk_size=default_parameters["num_samples"], seed=42
        )
        noise_snippets = (
            noise_snippets.reshape(num_chunks, default_parameters["num_samples"], default_parameters["num_channels"])
            .reshape(num_chunks, -1)
            .T
        )
        parameters = cls._optimize_amplitudes(noise_snippets, default_parameters)

        return parameters

    @classmethod
    def serialize_method_kwargs(cls, kwargs):
        kwargs = dict(kwargs)
        # remove waveform_extractor
        kwargs.pop("waveform_extractor")
        return kwargs

    @classmethod
    def unserialize_in_worker(cls, kwargs):
        return kwargs

    @classmethod
    def get_margin(cls, recording, kwargs):
        margin = 2 * max(kwargs["nbefore"], kwargs["nafter"])
        return margin

    @classmethod
    def main_function(cls, traces, d):
        peak_sign = d["peak_sign"]
        abs_threholds = d["abs_threholds"]
        exclude_sweep_size = d["exclude_sweep_size"]
        templates = d["templates"]
        num_templates = d["num_templates"]
        num_channels = d["num_channels"]
        overlaps = d["overlaps"]
        margin = d["margin"]
        norms = d["norms"]
        jitter = d["jitter"]
        patch_sizes = d["patch_sizes"]
        num_samples = d["nafter"] + d["nbefore"]
        neighbor_window = num_samples - 1
        amplitudes = d["amplitudes"]
        sym_patch = d["sym_patch"]

        peak_traces = traces[margin // 2 : -margin // 2, :]
        peak_sample_index, peak_chan_ind = DetectPeakByChannel.detect_peaks(
            peak_traces, peak_sign, abs_threholds, exclude_sweep_size
        )

        if jitter > 0:
            jittered_peaks = peak_sample_index[:, np.newaxis] + np.arange(-jitter, jitter)
            jittered_channels = peak_chan_ind[:, np.newaxis] + np.zeros(2 * jitter)
            mask = (jittered_peaks > 0) & (jittered_peaks < len(peak_traces))
            jittered_peaks = jittered_peaks[mask]
            jittered_channels = jittered_channels[mask]
            peak_sample_index, unique_idx = np.unique(jittered_peaks, return_index=True)
            peak_chan_ind = jittered_channels[unique_idx]
        else:
            peak_sample_index, unique_idx = np.unique(peak_sample_index, return_index=True)
            peak_chan_ind = peak_chan_ind[unique_idx]

        num_peaks = len(peak_sample_index)

        if sym_patch:
            snippets = extract_patches_2d(traces, patch_sizes)[peak_sample_index]
            peak_sample_index += margin // 2
        else:
            peak_sample_index += margin // 2
            snippet_window = np.arange(-d["nbefore"], d["nafter"])
            snippets = traces[peak_sample_index[:, np.newaxis] + snippet_window]

        if num_peaks > 0:
            snippets = snippets.reshape(num_peaks, -1)
            scalar_products = templates.dot(snippets.T)
        else:
            scalar_products = np.zeros((num_templates, 0), dtype=np.float32)

        num_spikes = 0
        spikes = np.empty(scalar_products.size, dtype=spike_dtype)
        idx_lookup = np.arange(scalar_products.size).reshape(num_templates, -1)

        min_sps = (amplitudes[:, 0] * norms)[:, np.newaxis]
        max_sps = (amplitudes[:, 1] * norms)[:, np.newaxis]

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
                cached_overlaps[best_cluster_ind] = overlaps[best_cluster_ind].toarray()

            to_add = -best_amplitude * cached_overlaps[best_cluster_ind][:, idx_neighbor]

            scalar_products[:, is_valid_nn[0] : is_valid_nn[1]] += to_add
            scalar_products[best_cluster_ind, is_valid_nn[0] : is_valid_nn[1]] = -np.inf

            spikes["sample_index"][num_spikes] = best_peak_sample_index
            spikes["channel_index"][num_spikes] = best_peak_chan_ind
            spikes["cluster_index"][num_spikes] = best_cluster_ind
            spikes["amplitude"][num_spikes] = best_amplitude
            num_spikes += 1

            is_valid = (scalar_products > min_sps) & (scalar_products < max_sps)

        spikes["amplitude"][:num_spikes] /= norms[spikes["cluster_index"][:num_spikes]]

        spikes = spikes[:num_spikes]
        order = np.argsort(spikes["sample_index"])
        spikes = spikes[order]

        return spikes
