"""Sorting components: template matching."""

import numpy as np

import scipy.spatial

from tqdm import tqdm
import scipy

try:
    import sklearn
    from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

from spikeinterface.core import (WaveformExtractor, get_noise_levels, get_random_data_chunks, 
                                 get_chunk_with_margin, get_channel_distances)
from spikeinterface.core.job_tools import ChunkRecordingExecutor
from spikeinterface.postprocessing import (get_template_channel_sparsity, get_template_extremum_channel)

from spikeinterface.sortingcomponents.peak_detection import detect_peak_locally_exclusive, detect_peaks_by_channel

potrs, = scipy.linalg.get_lapack_funcs(('potrs',), dtype=np.float32)

nrm2, = scipy.linalg.get_blas_funcs(('nrm2', ), dtype=np.float32)

spike_dtype = [('sample_ind', 'int64'), ('channel_ind', 'int64'), ('cluster_ind', 'int64'),
               ('amplitude', 'float64'), ('segment_ind', 'int64')]

from .main import BaseTemplateMatchingEngine

#################
# Circus peeler #
#################

from scipy.fft._helper import _init_nd_shape_and_axes
try:
    from scipy.signal.signaltools import  _init_freq_conv_axes, _apply_conv_mode
except Exception:
    from scipy.signal._signaltools import  _init_freq_conv_axes, _apply_conv_mode
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

    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes,
                                          sorted_axes=False)

    s1 = in1.shape
    s2 = in2.shape
    
    shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
             for i in range(in1.ndim)]

    if not len(axes):
        return in1 * in2

    complex_result = (in1.dtype.kind == 'c' or in2.dtype.kind == 'c')

    if calc_fast_len:
        # Speed up FFT by padding to optimal size.
        fshape = [
            sp_fft.next_fast_len(shape[a], not complex_result) for a in axes]
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

    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes,
                                          sorted_axes=False)

    s1 = in1.shape
    s2 = in2.shape
    
    shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
             for i in range(in1.ndim)]

    ret = _freq_domain_conv(in1, in2, axes, shape, cache, calc_fast_len=True)

    return _apply_conv_mode(ret, s1, s2, mode, axes)


def _freq_domain_conv(in1, in2, axes, shape, cache, calc_fast_len=True):
    
    if not len(axes):
        return in1 * in2

    complex_result = (in1.dtype.kind == 'c' or in2.dtype.kind == 'c')

    if calc_fast_len:
        # Speed up FFT by padding to optimal size.
        fshape = [
            sp_fft.next_fast_len(shape[a], not complex_result) for a in axes]
    else:
        fshape = shape

    if not complex_result:
        fft, ifft = sp_fft.rfftn, sp_fft.irfftn
    else:
        fft, ifft = sp_fft.fftn, sp_fft.ifftn

    sp1 = cache['full'][cache['mask']]
    sp2 = cache['template']

    #sp2 = fft(in2[cache['mask']], fshape, axes=axes)
    ret = ifft(sp1 * sp2, fshape, axes=axes)

    if calc_fast_len:
          fslice = tuple([slice(sz) for sz in shape])
          ret = ret[fslice]

    return ret


class CircusOMPPeeler(BaseTemplateMatchingEngine):
    """
    Orthogonal Matching Pursuit inspired from Spyking Circus sorter

    https://elifesciences.org/articles/34518
    
    This is an Orthogonal Template Matching algorithm. For speed and 
    memory optimization, templates are automatically sparsified if the 
    density of the matrix falls below a given threshold. Signal is
    convolved with the templates, and as long as some scalar products
    are higher than a given threshold, we use a Cholesky decomposition
    to compute the optimal amplitudes needed to reconstruct the signal.

    IMPORTANT NOTE: small chunks are more efficient for such Peeler,
    consider using 100ms chunk

    Parameters
    ----------
    noise_levels: array
        The noise levels, for every channels
    random_chunk_kwargs: dict
        Parameters for computing noise levels, if not provided (sub optimal)
    amplitude: tuple
        (Minimal, Maximal) amplitudes allowed for every template
    omp_min_sps: float
        Stopping criteria of the OMP algorithm, in percentage of the norm
    sparsify_threshold: float
        Templates are sparsified in order to keep only the channels necessary
        to explain a given fraction of the total norm
    use_sparse_matrix_threshold: float
        If density of the templates is below a given threshold, sparse matrix
        are used (memory efficient)
    progress_bar_steps: bool
        In order to display or not steps from the algorithm
    -----
    """

    _default_params = {
        'sparsify_threshold': 0.99,
        'amplitudes' : [0.6, 1.4],
        'noise_levels': None,
        'random_chunk_kwargs': {},
        'omp_min_sps' : 0.5,
        'waveform_extractor': None,
        'templates' : None,
        'overlaps' : None,
        'norms' : None,
        'ignored_ids' : []
    }

    @classmethod
    def _sparsify_template(cls, template, sparsify_threshold, noise_levels):

        is_silent = template.std(0) < 0.1*noise_levels
        template[:, is_silent] = 0

        channel_norms = np.linalg.norm(template, axis=0)**2
        total_norm = np.linalg.norm(template)**2

        idx = np.argsort(channel_norms)[::-1]
        explained_norms = np.cumsum(channel_norms[idx]/total_norm)
        channel = np.searchsorted(explained_norms, sparsify_threshold)
        active_channels = np.sort(idx[:channel])
        template[:, idx[channel:]] = 0
        return template, active_channels

    @classmethod
    def _prepare_templates(cls, d):
        
        waveform_extractor = d['waveform_extractor']
        num_samples = d['num_samples']
        num_channels = d['num_channels']
        num_templates = d['num_templates']

        d['norms'] = np.zeros(num_templates, dtype=np.float32)

        all_units = list(d['waveform_extractor'].sorting.unit_ids)

        templates = waveform_extractor.get_all_templates(mode='median')

        d['sparsities'] = {}
        d['templates'] = {}

        for count, unit_id in enumerate(all_units):
                
            template, active_channels = cls._sparsify_template(templates[count], d['sparsify_threshold'], d['noise_levels'])
            d['sparsities'][count] = active_channels
            d['norms'][count] = np.linalg.norm(template)
            d['templates'][count] = template[:, active_channels]/d['norms'][count]

        return d

    @classmethod
    def _prepare_overlaps(cls, d):

        templates = d['templates']
        num_samples = d['num_samples']
        num_channels = d['num_channels']
        num_templates = d['num_templates']
        sparsities = d['sparsities']

        dense_templates = np.zeros((num_templates, num_samples, num_channels), dtype=np.float32)
        for i in range(num_templates):
            dense_templates[i, :, sparsities[i]] = templates[i].T

        size = 2 * num_samples - 1

        all_delays = list(range(num_samples))

        overlaps = {}
        
        for delay in all_delays:
            source = dense_templates[:, :delay, :].reshape(num_templates, -1)
            target = dense_templates[:, num_samples-delay:, :].reshape(num_templates, -1)

            if delay > 0:
                overlaps[delay] = scipy.sparse.csr_matrix(source.dot(target.T))
            else:
                overlaps[delay] = scipy.sparse.csr_matrix((num_templates, num_templates), dtype=np.float32)
            
            if delay < num_samples:
                overlaps[size - delay-1] = overlaps[delay].T.tocsr()

        new_overlaps = []
        for i in range(num_templates):
            data = [overlaps[j][i, :].T for j in range(size)]
            data = scipy.sparse.hstack(data)
            new_overlaps += [data]
        
        d['overlaps'] = new_overlaps

        return d

    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):

        d = cls._default_params.copy()
        d.update(kwargs)

        assert isinstance(d['waveform_extractor'], WaveformExtractor)

        for v in ['sparsify_threshold', 'omp_min_sps']:
            assert (d[v] >= 0) and (d[v] <= 1), f'{v} should be in [0, 1]'
        
        if d['noise_levels'] is None:
            print('CircusOMPPeeler : noise should be computed outside')
            d['noise_levels'] = get_noise_levels(recording, **d['random_chunk_kwargs'], return_scaled=False)

        d['num_channels'] = d['waveform_extractor'].recording.get_num_channels()
        d['num_samples'] = d['waveform_extractor'].nsamples
        d['num_templates'] = len(d['waveform_extractor'].sorting.unit_ids)
        d['nbefore'] = d['waveform_extractor'].nbefore
        d['nafter'] = d['waveform_extractor'].nafter

        if d['templates'] is None:
            d = cls._prepare_templates(d)
        else:
            for key in ['norms', 'sparsities']:
                assert d[key] is not None, "If templates are provided, %d should also be there" %key

        if d['overlaps'] is None: 
            d = cls._prepare_overlaps(d)

        d['ignored_ids'] = np.array(d['ignored_ids'])

        return d        

    @classmethod
    def serialize_method_kwargs(cls, kwargs):
        kwargs = dict(kwargs)
        # remove waveform_extractor
        kwargs.pop('waveform_extractor')
        return kwargs

    @classmethod
    def unserialize_in_worker(cls, kwargs):
        return kwargs

    @classmethod
    def get_margin(cls, recording, kwargs):
        margin = 2 * max(kwargs['nbefore'], kwargs['nafter'])
        return margin

    @classmethod
    def main_function(cls, traces, d):
        templates = d['templates']
        num_templates = d['num_templates']
        num_channels = d['num_channels']
        overlaps = d['overlaps']
        margin = d['margin']
        norms = d['norms']
        nbefore = d['nbefore']
        nafter = d['nafter']
        omp_tol = np.finfo(np.float32).eps
        omp_min_sps = d['omp_min_sps']
        num_samples = d['nafter'] + d['nbefore']
        neighbor_window = num_samples - 1
        min_amplitude, max_amplitude = d['amplitudes']
        sparsities = d['sparsities']
        ignored_ids = d['ignored_ids']

        if 'cached_fft_kernels' not in d:
            d['cached_fft_kernels'] = {'fshape' : 0}

        cached_fft_kernels = d['cached_fft_kernels']

        stop_criteria = omp_min_sps * norms[:, np.newaxis]

        num_timesteps = len(traces)
        num_peaks = num_timesteps - num_samples + 1

        traces = np.ascontiguousarray(traces.T)

        dummy_filter = np.empty((num_channels, num_samples), dtype=np.float32)
        dummy_traces = np.empty((num_channels, num_timesteps), dtype=np.float32)

        fshape, axes = get_scipy_shape(dummy_filter, traces, axes=1)
        fft_cache = {'full' : sp_fft.rfftn(traces, fshape, axes=axes)}

        scalar_products = np.empty((num_templates, num_peaks), dtype=np.float32)

        flagged_chunk = cached_fft_kernels['fshape'] != fshape[0]

        for i in range(num_templates):

            if i not in cached_fft_kernels or flagged_chunk:
                kernel_filter = np.ascontiguousarray(templates[i][::-1].T)
                cached_fft_kernels.update({i : sp_fft.rfftn(kernel_filter, fshape, axes=axes)})
                cached_fft_kernels['fshape'] = fshape[0]

            fft_cache.update({'mask' : sparsities[i], 'template' : cached_fft_kernels[i]})

            convolution = fftconvolve_with_cache(dummy_filter, dummy_traces, fft_cache, axes=1, mode='valid')
            if len(convolution) > 0:
                scalar_products[i] = convolution.sum(0)
            else:
                scalar_products[i] = 0

        if len(ignored_ids) > 0:
            scalar_products[ignored_ids] = -np.inf

        num_spikes = 0
        spikes = np.empty(scalar_products.size, dtype=spike_dtype)
        idx_lookup = np.arange(scalar_products.size).reshape(num_templates, -1)

        M = np.empty((num_peaks, num_peaks), dtype=np.float32)

        all_selections = np.empty((2, scalar_products.size), dtype=np.int32)
        final_amplitudes = np.zeros(scalar_products.shape, dtype=np.float32)
        num_selection = 0

        full_sps = scalar_products.copy()

        neighbors = {}
        cached_overlaps = {}

        is_valid = (scalar_products > stop_criteria)

        while np.any(is_valid):

            best_amplitude_ind = scalar_products[is_valid].argmax()
            best_cluster_ind, peak_index = np.unravel_index(idx_lookup[is_valid][best_amplitude_ind], idx_lookup.shape)
            
            all_selections[:, num_selection] = [best_cluster_ind, peak_index]
            num_selection += 1

            selection = all_selections[:, :num_selection]

            res_sps = full_sps[selection[0], selection[1]]

            mb_selection = num_selection - 1

            delta_t = selection[1] - peak_index
            idx = np.where(np.abs(delta_t) <= neighbor_window)[0]

            myline = neighbor_window + delta_t[idx]
            if best_cluster_ind not in cached_overlaps.keys():
                cached_overlaps[best_cluster_ind] = overlaps[best_cluster_ind].toarray()

            M[mb_selection, idx] = cached_overlaps[best_cluster_ind][selection[0, idx], myline]

            if num_selection >= (M.shape[0] - 1):
                Z = np.empty((2*M.shape[0], 2*M.shape[1]), dtype=np.float32)
                Z[:num_selection, :num_selection] = M[:num_selection, :num_selection]
                M = Z

            if mb_selection > 0:
                scipy.linalg.solve_triangular(M[:mb_selection, :mb_selection], M[mb_selection, :mb_selection], trans=0,
                 lower=1,
                 overwrite_b=True,
                 check_finite=False)

                v = nrm2(M[mb_selection, :mb_selection]) ** 2
                if 1 - v <= omp_tol:  # selected atoms are dependent
                    break
                M[mb_selection, mb_selection] = np.sqrt(1 - v)

            all_amplitudes, _ = potrs(M[:num_selection, :num_selection], res_sps,
                lower=True, overwrite_b=False)

            all_amplitudes /= norms[selection[0]]

            diff_amplitudes = (all_amplitudes - final_amplitudes[selection[0], selection[1]])
            modified = np.where(np.abs(diff_amplitudes) > omp_tol)[0]
            final_amplitudes[selection[0], selection[1]] = all_amplitudes

            for i in modified:

                tmp_best, tmp_peak = selection[:, i]
                diff_amp = diff_amplitudes[i]*norms[tmp_best]
                
                if not tmp_best in cached_overlaps.keys():
                    cached_overlaps[tmp_best] = overlaps[tmp_best].toarray()

                if not tmp_peak in neighbors.keys():
                    idx = [max(0, tmp_peak - neighbor_window), min(num_peaks, tmp_peak + neighbor_window + 1)]
                    offset = [neighbor_window + idx[0] - tmp_peak, neighbor_window + idx[1] - tmp_peak]
                    neighbors[tmp_peak] = {'idx' : idx, 'tdx' : offset}

                idx = neighbors[tmp_peak]['idx']
                tdx = neighbors[tmp_peak]['tdx']

                to_add = diff_amp * cached_overlaps[tmp_best][:, tdx[0]:tdx[1]]
                scalar_products[:, idx[0]:idx[1]] -= to_add

            scalar_products[best_cluster_ind, peak_index] = -np.inf
            
            is_valid = (scalar_products > stop_criteria)

        is_valid = (final_amplitudes > min_amplitude)*(final_amplitudes < max_amplitude)
        valid_indices = np.where(is_valid)

        num_spikes = len(valid_indices[0])
        spikes['sample_ind'][:num_spikes] = valid_indices[1] + d['nbefore']
        spikes['channel_ind'][:num_spikes] = 0
        spikes['cluster_ind'][:num_spikes] = valid_indices[0]
        spikes['amplitude'][:num_spikes] = final_amplitudes[valid_indices[0], valid_indices[1]]
        
        spikes = spikes[:num_spikes]
        order = np.argsort(spikes['sample_ind'])
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
    sparsify_threshold: float
        Templates are sparsified in order to keep only the channels necessary
        to explain a given fraction of the total norm
    use_sparse_matrix_threshold: float
        If density of the templates is below a given threshold, sparse matrix
        are used (memory efficient)
    progress_bar_steps: bool
        In order to display or not steps from the algorithm
    -----


    """

    _default_params = {
        'peak_sign': 'neg', 
        'exclude_sweep_ms': 0.1,
        'jitter' : 1, 
        'detect_threshold': 5, 
        'noise_levels': None, 
        'random_chunk_kwargs': {},
        'sparsify_threshold': 0.99,
        'max_amplitude' : 1.5,
        'min_amplitude' : 0.5,
        'use_sparse_matrix_threshold' : 0.25,
        'progess_bar_steps' : True,
        'waveform_extractor': None,
    }

    @classmethod
    def _sparsify_template(cls, template, sparsify_threshold, noise_levels):

        is_silent = template.std(0) < 0.1*noise_levels
        template[:, is_silent] = 0

        channel_norms = np.linalg.norm(template, axis=0)**2
        total_norm = np.linalg.norm(template)**2

        idx = np.argsort(channel_norms)[::-1]
        explained_norms = np.cumsum(channel_norms[idx]/total_norm)
        channel = np.searchsorted(explained_norms, sparsify_threshold)
        active_channels = np.sort(idx[:channel])
        template[:, idx[channel:]] = 0
        return template, active_channels

    @classmethod
    def _prepare_templates(cls, d):
        
        waveform_extractor = d['waveform_extractor']
        num_samples = d['num_samples']
        num_channels = d['num_channels']
        num_templates = d['num_templates']
        max_amplitude = d['max_amplitude']
        min_amplitude = d['min_amplitude']
        use_sparse_matrix_threshold = d['use_sparse_matrix_threshold']

        d['norms'] = np.zeros(num_templates, dtype=np.float32)

        all_units = list(d['waveform_extractor'].sorting.unit_ids)

        templates = waveform_extractor.get_all_templates(mode='median').copy()

        d['sparsities'] = {}
        
        for count, unit_id in enumerate(all_units):
                
            templates[count], active_channels = cls._sparsify_template(templates[count], d['sparsify_threshold'], d['noise_levels'])
            d['sparsities'][count] = active_channels
            
            d['norms'][count] = np.linalg.norm(templates[count])
            templates[count] /= d['norms'][count]

        templates = templates.reshape(num_templates, -1)

        nnz = np.sum(templates != 0)/(num_templates * num_samples * num_channels)
        if nnz <= use_sparse_matrix_threshold:
            templates = scipy.sparse.csr_matrix(templates)
            print(f'Templates are automatically sparsified (sparsity level is {nnz})')
            d['is_dense'] = False
        else:
            d['is_dense'] = True

        d['templates'] = templates

        return d

    @classmethod
    def _prepare_overlaps(cls, d):

        templates = d['templates']
        num_samples = d['num_samples']
        num_channels = d['num_channels']
        num_templates = d['num_templates']
        is_dense = d['is_dense']

        if not is_dense:
            dense_templates = templates.toarray()
        else:
            dense_templates = templates

        dense_templates = dense_templates.reshape(num_templates, num_samples, num_channels)

        size = 2 * num_samples - 1

        all_delays = list(range(num_samples))
        if d['progess_bar_steps']:
            all_delays = tqdm(all_delays, desc='[1] compute overlaps')

        overlaps = {}
        
        for delay in all_delays:
            source = dense_templates[:, :delay, :].reshape(num_templates, -1)
            target = dense_templates[:, num_samples-delay:, :].reshape(num_templates, -1)

            if delay > 0:
                overlaps[delay] = scipy.sparse.csr_matrix(source.dot(target.T))
            else:
                overlaps[delay] = scipy.sparse.csr_matrix((num_templates, num_templates), dtype=np.float32)
            
            if delay < num_samples:
                overlaps[size - delay-1] = overlaps[delay].T.tocsr()

        new_overlaps = []
        for i in range(num_templates):
            data = [overlaps[j][i, :].T for j in range(size)]
            data = scipy.sparse.hstack(data)
            new_overlaps += [data]
        
        d['overlaps'] = new_overlaps

        return d

    @classmethod
    def _mcc_error(cls, bounds, good, bad):
        fn = np.sum((good < bounds[0]) | (good > bounds[1]))
        fp = np.sum((bounds[0] <= bad) & (bad <= bounds[1]))
        tp = np.sum((bounds[0] <= good) & (good <= bounds[1]))
        tn = np.sum((bad < bounds[0]) | (bad > bounds[1]))
        denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
        if denom > 0:
            mcc = 1 - (tp*tn - fp*fn)/np.sqrt(denom)
        else:
            mcc = 1
        return mcc

    @classmethod
    def _cost_function_mcc(cls, bounds, good, bad, delta_amplitude, alpha):
        # We want a minimal error, with the larger bounds that are possible
        cost = alpha*cls._mcc_error(bounds, good, bad) + (1 - alpha)*np.abs((1 - (bounds[1] - bounds[0])/delta_amplitude))
        return cost

    @classmethod
    def _optimize_amplitudes(cls, noise_snippets, d):

        waveform_extractor = d['waveform_extractor']
        templates = d['templates']
        num_templates = d['num_templates']
        max_amplitude = d['max_amplitude']
        min_amplitude = d['min_amplitude']
        alpha = 0.5
        norms = d['norms']
        all_units = list(waveform_extractor.sorting.unit_ids)
        if d['progess_bar_steps']:
            all_units = tqdm(all_units, desc='[2] compute amplitudes')

        d['amplitudes'] = np.zeros((num_templates, 2), dtype=np.float32)
        noise = templates.dot(noise_snippets)/norms[:, np.newaxis]

        all_amps = {}
        for count, unit_id in enumerate(all_units):
            w = waveform_extractor.get_waveforms(unit_id)
            snippets = w.reshape(w.shape[0], -1).T
            amps = templates.dot(snippets)/norms[:, np.newaxis]
            good = amps[count, :].flatten()

            sub_amps = amps[np.concatenate((np.arange(count), np.arange(count+1, num_templates))), :]
            bad = sub_amps[sub_amps >= good]
            bad = np.concatenate((bad, noise[count]))
            cost_kwargs = [good, bad, max_amplitude - min_amplitude, alpha]
            cost_bounds = [(min_amplitude, 1), (1, max_amplitude)]
            res = scipy.optimize.differential_evolution(cls._cost_function_mcc, bounds=cost_bounds, args=cost_kwargs)
            d['amplitudes'][count] = res.x

            # import pylab as plt
            # plt.hist(good, 100, alpha=0.5)
            # plt.hist(bad, 100, alpha=0.5)
            # plt.hist(noise[count], 100, alpha=0.5)
            # ymin, ymax = plt.ylim()
            # plt.plot([res.x[0], res.x[0]], [ymin, ymax], 'k--')
            # plt.plot([res.x[1], res.x[1]], [ymin, ymax], 'k--')
            # plt.savefig('test_%d.png' %count)
            # plt.close()

        return d

    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):

        assert HAVE_SKLEARN, "CircusPeeler needs sklearn to work"
        d = cls._default_params.copy()
        d.update(kwargs)

        assert isinstance(d['waveform_extractor'], WaveformExtractor)

        for v in ['sparsify_threshold', 'use_sparse_matrix_threshold']:
            assert (d[v] >= 0) and (d[v] <= 1), f'{v} should be in [0, 1]'
        
        d['num_channels'] = d['waveform_extractor'].recording.get_num_channels()
        d['num_samples'] = d['waveform_extractor'].nsamples
        d['num_templates'] = len(d['waveform_extractor'].sorting.unit_ids)

        if d['noise_levels'] is None:
            print('CircusPeeler : noise should be computed outside')
            d['noise_levels'] = get_noise_levels(recording, **d['random_chunk_kwargs'], return_scaled=False)

        d['abs_threholds'] = d['noise_levels'] * d['detect_threshold']

        d = cls._prepare_templates(d)
        d = cls._prepare_overlaps(d)

        d['exclude_sweep_size'] = int(d['exclude_sweep_ms'] * recording.get_sampling_frequency() / 1000.)

        d['nbefore'] = d['waveform_extractor'].nbefore
        d['nafter'] = d['waveform_extractor'].nafter
        d['patch_sizes'] = (d['waveform_extractor'].nsamples, d['num_channels'])
        d['sym_patch'] = d['nbefore'] == d['nafter']
        #d['jitter'] = int(1e-3*d['jitter'] * recording.get_sampling_frequency())

        num_segments = recording.get_num_segments()
        if d['waveform_extractor']._params['max_spikes_per_unit'] is None:
            num_snippets = 1000
        else:
            num_snippets = 2*d['waveform_extractor']._params['max_spikes_per_unit']

        num_chunks = num_snippets // num_segments
        noise_snippets = get_random_data_chunks(recording, num_chunks_per_segment=num_chunks, chunk_size=d['num_samples'], seed=42)
        noise_snippets = noise_snippets.reshape(num_chunks, d['num_samples'], d['num_channels']).reshape(num_chunks, -1).T
        d = cls._optimize_amplitudes(noise_snippets, d)

        return d        

    @classmethod
    def serialize_method_kwargs(cls, kwargs):
        kwargs = dict(kwargs)
        # remove waveform_extractor
        kwargs.pop('waveform_extractor')
        return kwargs

    @classmethod
    def unserialize_in_worker(cls, kwargs):
        return kwargs

    @classmethod
    def get_margin(cls, recording, kwargs):
        margin = 2 * max(kwargs['nbefore'], kwargs['nafter'])
        return margin

    @classmethod
    def main_function(cls, traces, d):
        peak_sign = d['peak_sign']
        abs_threholds = d['abs_threholds']
        exclude_sweep_size = d['exclude_sweep_size']
        templates = d['templates']
        num_templates = d['num_templates']
        num_channels = d['num_channels']
        overlaps = d['overlaps']
        margin = d['margin']
        norms = d['norms']
        jitter = d['jitter']
        patch_sizes = d['patch_sizes']
        num_samples = d['nafter'] + d['nbefore']
        neighbor_window = num_samples - 1
        amplitudes = d['amplitudes']
        sym_patch = d['sym_patch']
        sparsities = d['sparsities']
        is_dense = d['is_dense']
        
        peak_traces = traces[margin // 2:-margin // 2, :]
        peak_sample_ind, peak_chan_ind = detect_peaks_by_channel(peak_traces, peak_sign, abs_threholds, exclude_sweep_size)

        if jitter > 0:
            jittered_peaks = peak_sample_ind[:, np.newaxis] + np.arange(-jitter, jitter)
            jittered_channels = peak_chan_ind[:, np.newaxis] + np.zeros(2*jitter)
            mask = (jittered_peaks > 0) & (jittered_peaks < len(peak_traces))
            jittered_peaks = jittered_peaks[mask]
            jittered_channels = jittered_channels[mask]
            peak_sample_ind, unique_idx = np.unique(jittered_peaks, return_index=True)
            peak_chan_ind = jittered_channels[unique_idx]
        else:
            peak_sample_ind, unique_idx = np.unique(peak_sample_ind, return_index=True)
            peak_chan_ind = peak_chan_ind[unique_idx]

        num_peaks = len(peak_sample_ind)

        if sym_patch:
            snippets = extract_patches_2d(traces, patch_sizes)[peak_sample_ind]
            peak_sample_ind += margin // 2
        else:
            peak_sample_ind += margin // 2
            snippet_window = np.arange(-d['nbefore'], d['nafter'])
            snippets = traces[peak_sample_ind[:, np.newaxis] + snippet_window]

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
            best_peak_sample_ind = peak_sample_ind[peak_index]
            best_peak_chan_ind = peak_chan_ind[peak_index]

            peak_data = peak_sample_ind - peak_sample_ind[peak_index]
            is_valid = np.searchsorted(peak_data, [-neighbor_window, neighbor_window + 1])
            idx_neighbor = peak_data[is_valid[0]:is_valid[1]] + neighbor_window

            if not best_cluster_ind in cached_overlaps.keys():
                cached_overlaps[best_cluster_ind] = overlaps[best_cluster_ind].toarray()

            to_add = -best_amplitude * cached_overlaps[best_cluster_ind][:, idx_neighbor]

            scalar_products[:, is_valid[0]:is_valid[1]] += to_add
            scalar_products[best_cluster_ind, is_valid[0]:is_valid[1]] = -np.inf

            spikes['sample_ind'][num_spikes] = best_peak_sample_ind
            spikes['channel_ind'][num_spikes] = best_peak_chan_ind
            spikes['cluster_ind'][num_spikes] = best_cluster_ind
            spikes['amplitude'][num_spikes] = best_amplitude
            num_spikes += 1

            is_valid = (scalar_products > min_sps) & (scalar_products < max_sps)

        spikes['amplitude'][:num_spikes] /= norms[spikes['cluster_ind'][:num_spikes]]
        
        spikes = spikes[:num_spikes]
        order = np.argsort(spikes['sample_ind'])
        spikes = spikes[order]

        return spikes
