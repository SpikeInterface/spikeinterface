"""Sorting components: template matching."""

from __future__ import annotations


import numpy as np
import warnings

import scipy.spatial

import scipy

try:
    import sklearn
    from sklearn.feature_extraction.image import extract_patches_2d

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


from spikeinterface.core import get_noise_levels
from spikeinterface.sortingcomponents.peak_detection import DetectPeakByChannel
from spikeinterface.core.template import Templates

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
        "templates": None,
        "rank": 5,
        "ignored_ids": [],
        "vicinity": 3,
    }

    @classmethod
    def _prepare_templates(cls, d):
        templates = d["templates"]
        num_templates = len(d["templates"].unit_ids)

        assert d["stop_criteria"] in ["max_failures", "omp_min_sps", "relative_error"]

        sparsity = templates.sparsity.mask

        units_overlaps = np.sum(np.logical_and(sparsity[:, np.newaxis, :], sparsity[np.newaxis, :, :]), axis=2)
        d["units_overlaps"] = units_overlaps > 0
        d["unit_overlaps_indices"] = {}
        for i in range(num_templates):
            (d["unit_overlaps_indices"][i],) = np.nonzero(d["units_overlaps"][i])

        templates_array = templates.get_dense_templates().copy()
        templates_array -= templates_array.mean(axis=(1, 2))[:, None, None]

        # Then we keep only the strongest components
        rank = d["rank"]
        temporal, singular, spatial = np.linalg.svd(templates_array, full_matrices=False)
        d["temporal"] = temporal[:, :, :rank]
        d["singular"] = singular[:, :rank]
        d["spatial"] = spatial[:, :rank, :]

        # We reconstruct the approximated templates
        templates_array = np.matmul(d["temporal"] * d["singular"][:, np.newaxis, :], d["spatial"])

        d["normed_templates"] = np.zeros(templates_array.shape, dtype=np.float32)
        d["norms"] = np.zeros(num_templates, dtype=np.float32)

        # And get the norms, saving compressed templates for CC matrix
        for count in range(num_templates):
            template = templates_array[count][:, sparsity[count]]
            d["norms"][count] = np.linalg.norm(template)
            d["normed_templates"][count][:, sparsity[count]] = template / d["norms"][count]

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
                overlapped_channels = sparsity[j]
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

        assert isinstance(d["templates"], Templates), (
            f"The templates supplied is of type {type(d['templates'])} " f"and must be a Templates"
        )

        d["num_channels"] = recording.get_num_channels()
        d["num_samples"] = d["templates"].num_samples
        d["nbefore"] = d["templates"].nbefore
        d["nafter"] = d["templates"].nafter
        d["sampling_frequency"] = recording.get_sampling_frequency()
        d["vicinity"] *= d["num_samples"]

        if "overlaps" not in d:
            d = cls._prepare_templates(d)
        else:
            for key in [
                "norms",
                "temporal",
                "spatial",
                "singular",
                "units_overlaps",
                "unit_overlaps_indices",
            ]:
                assert d[key] is not None, "If templates are provided, %d should also be there" % key

        d["num_templates"] = len(d["templates"].templates_array)
        d["ignored_ids"] = np.array(d["ignored_ids"])

        d["unit_overlaps_tables"] = {}
        for i in range(d["num_templates"]):
            d["unit_overlaps_tables"][i] = np.zeros(d["num_templates"], dtype=int)
            d["unit_overlaps_tables"][i][d["unit_overlaps_indices"][i]] = np.arange(len(d["unit_overlaps_indices"][i]))

        return d

    @classmethod
    def serialize_method_kwargs(cls, kwargs):
        kwargs = dict(kwargs)
        return kwargs

    @classmethod
    def unserialize_in_worker(cls, kwargs):
        return kwargs

    @classmethod
    def get_margin(cls, recording, kwargs):
        if kwargs["vicinity"] > 0:
            margin = kwargs["vicinity"]
        else:
            margin = 2 * kwargs["num_samples"]
        return margin

    @classmethod
    def main_function(cls, traces, d):
        num_templates = d["num_templates"]
        num_channels = d["num_channels"]
        num_samples = d["num_samples"]
        overlaps_array = d["overlaps"]
        norms = d["norms"]
        omp_tol = np.finfo(np.float32).eps
        num_samples = d["nafter"] + d["nbefore"]
        neighbor_window = num_samples - 1
        min_amplitude, max_amplitude = d["amplitudes"]
        ignored_ids = d["ignored_ids"]
        vicinity = d["vicinity"]

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

                local_overlaps = overlaps_array[best_cluster_ind]
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

            if vicinity == 0:
                all_amplitudes, _ = potrs(M[:num_selection, :num_selection], res_sps, lower=True, overwrite_b=False)
                all_amplitudes /= norms[selection[0]]
            else:
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

                local_overlaps = overlaps_array[tmp_best]
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
        "templates": None,
    }

    @classmethod
    def _prepare_templates(cls, d):
        templates = d["templates"]
        num_samples = d["num_samples"]
        num_channels = d["num_channels"]
        num_templates = d["num_templates"]
        use_sparse_matrix_threshold = d["use_sparse_matrix_threshold"]

        d["norms"] = np.zeros(num_templates, dtype=np.float32)

        all_units = d["templates"].unit_ids

        sparsity = templates.sparsity.mask

        templates_array = templates.get_dense_templates()
        d["sparsities"] = {}
        d["normed_templates"] = {}

        for count, unit_id in enumerate(all_units):
            (d["sparsities"][count],) = np.nonzero(sparsity[count])
            d["norms"][count] = np.linalg.norm(templates_array[count])
            templates_array[count] /= d["norms"][count]
            d["normed_templates"][count] = templates_array[count][:, sparsity[count]]

        templates_array = templates_array.reshape(num_templates, -1)

        nnz = np.sum(templates_array != 0) / (num_templates * num_samples * num_channels)
        if nnz <= use_sparse_matrix_threshold:
            templates_array = scipy.sparse.csr_matrix(templates_array)
            print(f"Templates are automatically sparsified (sparsity level is {nnz})")
            d["is_dense"] = False
        else:
            d["is_dense"] = True

        d["circus_templates"] = templates_array

        return d

    # @classmethod
    # def _mcc_error(cls, bounds, good, bad):
    #     fn = np.sum((good < bounds[0]) | (good > bounds[1]))
    #     fp = np.sum((bounds[0] <= bad) & (bad <= bounds[1]))
    #     tp = np.sum((bounds[0] <= good) & (good <= bounds[1]))
    #     tn = np.sum((bad < bounds[0]) | (bad > bounds[1]))
    #     denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    #     if denom > 0:
    #         mcc = 1 - (tp * tn - fp * fn) / np.sqrt(denom)
    #     else:
    #         mcc = 1
    #     return mcc

    # @classmethod
    # def _cost_function_mcc(cls, bounds, good, bad, delta_amplitude, alpha):
    #     # We want a minimal error, with the larger bounds that are possible
    #     cost = alpha * cls._mcc_error(bounds, good, bad) + (1 - alpha) * np.abs(
    #         (1 - (bounds[1] - bounds[0]) / delta_amplitude)
    #     )
    #     return cost

    # @classmethod
    # def _optimize_amplitudes(cls, noise_snippets, d):
    #     parameters = d
    #     waveform_extractor = parameters["waveform_extractor"]
    #     templates = parameters["templates"]
    #     num_templates = parameters["num_templates"]
    #     max_amplitude = parameters["max_amplitude"]
    #     min_amplitude = parameters["min_amplitude"]
    #     alpha = 0.5
    #     norms = parameters["norms"]
    #     all_units = list(waveform_extractor.sorting.unit_ids)

    #     parameters["amplitudes"] = np.zeros((num_templates, 2), dtype=np.float32)
    #     noise = templates.dot(noise_snippets) / norms[:, np.newaxis]

    #     all_amps = {}
    #     for count, unit_id in enumerate(all_units):
    #         waveform = waveform_extractor.get_waveforms(unit_id, force_dense=True)
    #         snippets = waveform.reshape(waveform.shape[0], -1).T
    #         amps = templates.dot(snippets) / norms[:, np.newaxis]
    #         good = amps[count, :].flatten()

    #         sub_amps = amps[np.concatenate((np.arange(count), np.arange(count + 1, num_templates))), :]
    #         bad = sub_amps[sub_amps >= good]
    #         bad = np.concatenate((bad, noise[count]))
    #         cost_kwargs = [good, bad, max_amplitude - min_amplitude, alpha]
    #         cost_bounds = [(min_amplitude, 1), (1, max_amplitude)]
    #         res = scipy.optimize.differential_evolution(cls._cost_function_mcc, bounds=cost_bounds, args=cost_kwargs)
    #         parameters["amplitudes"][count] = res.x

    #     return d

    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):
        assert HAVE_SKLEARN, "CircusPeeler needs sklearn to work"
        d = cls._default_params.copy()
        d.update(kwargs)

        # assert isinstance(d['waveform_extractor'], WaveformExtractor)
        for v in ["use_sparse_matrix_threshold"]:
            assert (d[v] >= 0) and (d[v] <= 1), f"{v} should be in [0, 1]"

        d["num_channels"] = recording.get_num_channels()
        d["num_samples"] = d["templates"].num_samples
        d["num_templates"] = len(d["templates"].unit_ids)

        if d["noise_levels"] is None:
            print("CircusPeeler : noise should be computed outside")
            d["noise_levels"] = get_noise_levels(recording, **d["random_chunk_kwargs"], return_scaled=False)

        d["abs_threholds"] = d["noise_levels"] * d["detect_threshold"]

        if "overlaps" not in d:
            d = cls._prepare_templates(d)
            d["overlaps"] = compute_overlaps(
                d["normed_templates"],
                d["num_samples"],
                d["num_channels"],
                d["sparsities"],
            )
        else:
            for key in ["circus_templates", "norms"]:
                assert d[key] is not None, "If templates are provided, %d should also be there" % key

        d["exclude_sweep_size"] = int(d["exclude_sweep_ms"] * recording.get_sampling_frequency() / 1000.0)

        d["nbefore"] = d["templates"].nbefore
        d["nafter"] = d["templates"].nafter
        d["patch_sizes"] = (
            d["templates"].num_samples,
            d["num_channels"],
        )
        d["sym_patch"] = d["nbefore"] == d["nafter"]
        d["jitter"] = int(d["jitter_ms"] * recording.get_sampling_frequency() / 1000.0)

        d["amplitudes"] = np.zeros((d["num_templates"], 2), dtype=np.float32)
        d["amplitudes"][:, 0] = d["min_amplitude"]
        d["amplitudes"][:, 1] = d["max_amplitude"]
        # num_segments = recording.get_num_segments()
        # if d["waveform_extractor"]._params["max_spikes_per_unit"] is None:
        #     num_snippets = 1000
        # else:
        #     num_snippets = 2 * d["waveform_extractor"]._params["max_spikes_per_unit"]

        # num_chunks = num_snippets // num_segments
        # noise_snippets = get_random_data_chunks(
        #     recording, num_chunks_per_segment=num_chunks, chunk_size=d["num_samples"], seed=42
        # )
        # noise_snippets = (
        #     noise_snippets.reshape(num_chunks, d["num_samples"], d["num_channels"])
        #     .reshape(num_chunks, -1)
        #     .T
        # )
        # parameters = cls._optimize_amplitudes(noise_snippets, d)

        return d

    @classmethod
    def serialize_method_kwargs(cls, kwargs):
        kwargs = dict(kwargs)
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
        templates = d["circus_templates"]
        num_templates = d["num_templates"]
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
