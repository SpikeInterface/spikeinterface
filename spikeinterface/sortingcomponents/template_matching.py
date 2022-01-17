"""Sorting components: template matching."""

import numpy as np

import scipy.spatial

from tqdm import tqdm
import sklearn, scipy

from threadpoolctl import threadpool_limits

try:
    import numba
    from numba import jit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


from spikeinterface.core import WaveformExtractor
from spikeinterface.core.job_tools import ChunkRecordingExecutor
from spikeinterface.toolkit import (get_noise_levels, get_template_channel_sparsity,
    get_channel_distances, get_chunk_with_margin, get_template_extremum_channel)

from spikeinterface.sortingcomponents.peak_detection import detect_peak_locally_exclusive, detect_peaks_by_channel

spike_dtype = [('sample_ind', 'int64'), ('channel_ind', 'int64'), ('cluster_ind', 'int64'),
               ('amplitude', 'float64'), ('segment_ind', 'int64')]


def find_spikes_from_templates(recording, method='naive', method_kwargs={}, extra_ouputs=False,
                              **job_kwargs):
    """Find spike from a recording from given templates.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    waveform_extractor: WaveformExtractor
        The waveform extractor
    method: str 
        Which method to use ('naive' | 'tridesclous' | 'circus')
    method_kwargs: dict, optional
        Keyword arguments for the chosen method
    extra_ouputs: bool
        If True then method_kwargs is also return
    job_kwargs: dict
        Parameters for ChunkRecordingExecutor

    Returns
    -------
    spikes: ndarray
        Spikes found from templates.
    method_kwargs: 
        Optionaly returns for debug purpose.
    Notes
    -----
    Templates are represented as WaveformExtractor so statistics can be extracted.
    """

    assert method in template_matching_methods
    
    method_class = template_matching_methods[method]
    
    # initialize
    method_kwargs = method_class.initialize_and_check_kwargs(recording, method_kwargs)
    
    # add 
    method_kwargs['margin'] = method_class.get_margin(recording, method_kwargs)
    
    # serialiaze for worker
    method_kwargs_seralized = method_class.serialize_method_kwargs(method_kwargs)
    
    # and run
    func = _find_spikes_chunk
    init_func = _init_worker_find_spikes
    init_args = (recording.to_dict(), method, method_kwargs_seralized)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args,
                                       handle_returns=True, job_name=f'find spikes ({method})', **job_kwargs)
    spikes = processor.run()

    spikes = np.concatenate(spikes)
    
    if extra_ouputs:
        return spikes, method_kwargs
    else:
        return spikes


def _init_worker_find_spikes(recording, method, method_kwargs):
    """Initialize worker for finding spikes."""

    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)

    method_class = template_matching_methods[method]
    method_kwargs = method_class.unserialize_in_worker(method_kwargs)


    # create a local dict per worker
    worker_ctx = {}
    worker_ctx['recording'] = recording
    worker_ctx['method'] = method
    worker_ctx['method_kwargs'] = method_kwargs
    worker_ctx['function'] = method_class.main_function
    

    return worker_ctx


def _find_spikes_chunk(segment_index, start_frame, end_frame, worker_ctx):
    """Find spikes from a chunk of data."""

    # recover variables of the worker
    recording = worker_ctx['recording']
    method = worker_ctx['method']
    method_kwargs = worker_ctx['method_kwargs']
    margin = method_kwargs['margin']
    
    # load trace in memory given some margin
    recording_segment = recording._recording_segments[segment_index]
    traces, left_margin, right_margin = get_chunk_with_margin(recording_segment,
                start_frame, end_frame, None, margin, add_zeros=True)

    
    function = worker_ctx['function']
    
    with threadpool_limits(limits=1):
        spikes = function(traces, method_kwargs)
    
    # remove spikes in margin
    if margin > 0:
        keep = (spikes['sample_ind']  >= margin) & (spikes['sample_ind']  < (traces.shape[0] - margin))
        spikes = spikes[keep]

    spikes['sample_ind'] += (start_frame - margin)
    spikes['segment_ind'] = segment_index
    return spikes


# generic class for template engine
class BaseTemplateMatchingEngine:
    default_params = {}
    
    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):
        """This function runs before loops"""
        # need to be implemented in subclass
        raise NotImplementedError

    @classmethod
    def serialize_method_kwargs(cls, kwargs):
        """This function serializes kwargs to distribute them to workers"""
        # need to be implemented in subclass
        raise NotImplementedError

    @classmethod
    def unserialize_in_worker(cls, recording, kwargs):
        """This function unserializes kwargs in workers"""
        # need to be implemented in subclass
        raise NotImplementedError

    @classmethod
    def get_margin(cls, recording, kwargs):
        # need to be implemented in subclass
        raise NotImplementedError

    @classmethod
    def main_function(cls, traces, method_kwargs):
        """This function returns the number of samples for the chunk margins"""
        # need to be implemented in subclass
        raise NotImplementedError

    

##################
# naive matching #
##################


class NaiveMatching(BaseTemplateMatchingEngine):
    """
    This is a naive template matching that does not resolve collision
    and does not take in account sparsity.
    It just minimizes the distance to templates for detected peaks.

    It is implemented for benchmarking against this low quality template matching.
    And also as an example how to deal with methods_kwargs, margin, intit, func, ...
    """
    default_params = {
        'waveform_extractor': None,
        'peak_sign': 'neg',
        'n_shifts': 10,
        'detect_threshold': 5,
        'noise_levels': None,
        'local_radius_um': 100,
        'random_chunk_kwargs': {},
    }
    

    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):
        d = cls.default_params.copy()
        d.update(kwargs)
        
        assert d['waveform_extractor'] is not None
        
        we = d['waveform_extractor']

        if d['noise_levels'] is None:
            d['noise_levels'] = get_noise_levels(recording, **d['random_chunk_kwargs'])

        d['abs_threholds'] = d['noise_levels'] * d['detect_threshold']

        channel_distance = get_channel_distances(recording)
        d['neighbours_mask'] = channel_distance < d['local_radius_um']

        d['nbefore'] = we.nbefore
        d['nafter'] = we.nafter        

        return d
    
    @classmethod
    def get_margin(cls, recording, kwargs):
        margin = max(kwargs['nbefore'], kwargs['nafter'])
        return margin

    @classmethod
    def serialize_method_kwargs(cls, kwargs):
        kwargs = dict(kwargs)
        
        waveform_extractor = kwargs['waveform_extractor']
        kwargs['waveform_extractor'] = str(waveform_extractor.folder)
        
        return kwargs

    @classmethod
    def unserialize_in_worker(cls, kwargs):
        
        we = kwargs['waveform_extractor']
        if  isinstance(we, str):
            we = WaveformExtractor.load_from_folder(we)
            kwargs['waveform_extractor'] = we
        
        templates = we.get_all_templates(mode='average')
        
        kwargs['templates'] = templates
        
        return kwargs

    @classmethod
    def main_function(cls, traces, method_kwargs):
        
        peak_sign = method_kwargs['peak_sign']
        abs_threholds = method_kwargs['abs_threholds']
        n_shifts = method_kwargs['n_shifts']
        neighbours_mask = method_kwargs['neighbours_mask']
        templates = method_kwargs['templates']
        
        nbefore = method_kwargs['nbefore']
        nafter = method_kwargs['nafter']
        
        margin = method_kwargs['margin']
        
        if margin > 0:
            peak_traces = traces[margin:-margin, :]
        else:
            peak_traces = traces
        peak_sample_ind, peak_chan_ind = detect_peak_locally_exclusive(peak_traces, peak_sign, abs_threholds, n_shifts, neighbours_mask)
        peak_sample_ind += margin


        spikes = np.zeros(peak_sample_ind.size, dtype=spike_dtype)
        spikes['sample_ind'] = peak_sample_ind
        spikes['channel_ind'] = peak_chan_ind  # TODO need to put the channel from template
        
        # naively take the closest template
        for i in range(peak_sample_ind.size):
            i0 = peak_sample_ind[i] - nbefore
            i1 = peak_sample_ind[i] + nafter
            
            wf = traces[i0:i1, :]
            dist = np.sum(np.sum((templates - wf[None, : , :])**2, axis=1), axis=1)
            cluster_ind = np.argmin(dist)

            spikes['cluster_ind'][i] = cluster_ind
            spikes['amplitude'][i] = 0.

        return spikes


######################
# tridesclous peeler #
######################


class TridesclousPeeler(BaseTemplateMatchingEngine):
    """
    Template-matching ported from Tridesclous sorter.
    
    @Sam add short description
    """    
    default_params = {
        'waveform_extractor': None,
        'peak_sign': 'neg',
        'peak_shift_ms':  0.2,
        'detect_threshold': 5,
        'noise_levels': None,
        'local_radius_um': 100,
        'num_closest' : 5,
        'shift' : 4,
    }
    
    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):
        
        assert HAVE_NUMBA
        
        d = cls.default_params.copy()
        d.update(kwargs)

        assert isinstance(d['waveform_extractor'], WaveformExtractor)
        
        we = d['waveform_extractor']
        unit_ids = we.sorting.unit_ids
        channel_ids = we.recording.channel_ids
        
        sr = we.recording.get_sampling_frequency()


        # TODO load as sharedmem
        templates = we.get_all_templates(mode='average')
        d['templates'] = templates

        
        d['peak_shift'] = int(d['peak_shift_ms'] / 1000 * sr)

        if d['noise_levels'] is None:
            print('TridesclousPeeler : noise should be computed outside')
            d['noise_levels'] = get_noise_levels(recording)

        d['abs_threholds'] = d['noise_levels'] * d['detect_threshold']
    
        channel_distance = get_channel_distances(recording)
        d['neighbours_mask'] = channel_distance < d['local_radius_um']
        
        #
        template_sparsity_inds = get_template_channel_sparsity(we, method='radius',
                                  peak_sign=d['peak_sign'], outputs='index', radius_um=d['local_radius_um'])
        template_sparsity = np.zeros((unit_ids.size, channel_ids.size), dtype='bool')
        for unit_index, unit_id in enumerate(unit_ids):
            chan_inds = template_sparsity_inds[unit_id]
            template_sparsity[unit_index, chan_inds]  = True
        
        d['template_sparsity'] = template_sparsity
        
        extremum_channel = get_template_extremum_channel(we, peak_sign=d['peak_sign'], outputs='index')
        # as numpy vector
        extremum_channel = np.array([extremum_channel[unit_id] for unit_id in unit_ids], dtype='int64')
        d['extremum_channel'] = extremum_channel
        
        channel_locations = we.recording.get_channel_locations()
        
        # TODO try it with real locaion
        unit_locations = channel_locations[extremum_channel]
        #~ print(unit_locations)
        
        # distance between units
        unit_distances = scipy.spatial.distance.cdist(unit_locations, unit_locations, metric='euclidean')
        
        # seach for closet units and unitary discriminant vector
        closest_units = []
        for unit_ind, unit_id in enumerate(unit_ids):
            order = np.argsort(unit_distances[unit_ind, :])
            closest_u = np.arange(unit_ids.size)[order].tolist()
            closest_u.remove(unit_ind)
            closest_u = np.array(closest_u[:d['num_closest']])

            # compute unitary discriminent vector
            chans, = np.nonzero(d['template_sparsity'][unit_ind, :])
            template_sparse = templates[unit_ind, :, :][:, chans]
            closest_vec = []
            # against N closets
            for u in closest_u:
                vec = (templates[u, :, :][:, chans] - template_sparse)
                vec /= np.sum(vec ** 2)
                closest_vec.append((u, vec))
            # against noise
            closest_vec.append((None, - template_sparse / np.sum(template_sparse ** 2)))
            
            closest_units.append(closest_vec)

        d['closest_units'] = closest_units
        
        # distance channel from unit
        distances = scipy.spatial.distance.cdist(channel_locations, unit_locations, metric='euclidean')
        near_cluster_mask = distances < d['local_radius_um']

        # nearby cluster for each channel
        possible_clusters_by_channel = []
        for channel_ind in range(distances.shape[0]):
            cluster_inds, = np.nonzero(near_cluster_mask[channel_ind, :])
            possible_clusters_by_channel.append(cluster_inds)
        
        d['possible_clusters_by_channel'] = possible_clusters_by_channel


        d['nbefore'] = we.nbefore
        d['nafter'] = we.nafter
        
        d['possible_shifts'] = np.arange(-d['shift'], d['shift'] +1, dtype='int64')

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
        margin = 2 * (kwargs['nbefore'] + kwargs['nafter'])
        return margin

    @classmethod
    def main_function(cls, traces, d):
        
        traces = traces.copy()
        
        all_spikes = []
        level = 0
        while True:
            
            # find spikes
            spikes = _tdc_find_spikes(traces, d, level=level)
            keep = (spikes['cluster_ind'] >= 0)
            
            if not np.any(keep):
                break
            all_spikes.append(spikes[keep])
            
            #~ _tdc_remove_spikes(traces, spikes, d)
            
            level += 1
            
            if level == 2:
                break

        all_spikes = np.concatenate(all_spikes)
        order = np.argsort(all_spikes['sample_ind'])
        all_spikes = all_spikes[order]
        
        return all_spikes


#~ def _tdc_remove_spikes(traces, spikes, d):
    #~ nbefore, nafter = d['nbefore'], d['nafter']
    #~ for spike in spikes:
        #~ if spike['cluster_ind'] < 0:
            #~ continue
        #~ template = d['templates'][spike['cluster_ind'], :, :]
        #~ s0 = spike['sample_ind'] - d['nbefore']
        #~ s1 = spike['sample_ind'] + d['nafter']
        #~ traces[s0:s1, :] -= template * spike['amplitude']
    

def _tdc_find_spikes(traces, d, level=0):
        peak_sign = d['peak_sign']
        templates = d['templates']
        margin = d['margin']
        possible_clusters_by_channel = d['possible_clusters_by_channel']
        
        
        peak_traces = traces[margin // 2:-margin // 2, :]
        peak_sample_ind, peak_chan_ind = detect_peak_locally_exclusive(peak_traces, peak_sign,
                                    d['abs_threholds'], d['peak_shift'], d['neighbours_mask'])
        peak_sample_ind += margin // 2


        peak_amplitude = traces[peak_sample_ind, peak_chan_ind]
        order = np.argsort(np.abs(peak_amplitude))[::-1]
        peak_sample_ind = peak_sample_ind[order]
        peak_chan_ind = peak_chan_ind[order]

        spikes = np.zeros(peak_sample_ind.size, dtype=spike_dtype)
        spikes['sample_ind'] = peak_sample_ind
        spikes['channel_ind'] = peak_chan_ind  # TODO need to put the channel from template


        
        # naively take the closest template
        for i in range(peak_sample_ind.size):
            sample_ind = peak_sample_ind[i]
            
            i0 = sample_ind - d['nbefore']
            i1 = sample_ind + d['nafter']

            chan_ind = peak_chan_ind[i]
            possible_clusters = possible_clusters_by_channel[chan_ind]
            
            if possible_clusters.size > 0:
                wf = traces[i0:i1, :]
                
                ## pure numpy with cluster spasity
                # distances = np.sum(np.sum((templates[possible_clusters, :, :] - wf[None, : , :])**2, axis=1), axis=1)

                ## pure numpy with cluster+channel spasity
                union_channels, = np.nonzero(np.any(d['template_sparsity'][possible_clusters, :], axis=0))
                distances = np.sum(np.sum((templates[possible_clusters][:, :, union_channels] - wf[: , union_channels][None, : :])**2, axis=1), axis=1)
                
                ## numba with cluster+channel spasity
                # union_channels = np.any(d['template_sparsity'][possible_clusters, :], axis=0)
                # distances = numba_sparse_dist(wf, templates, union_channels, possible_clusters)

                ind = np.argmin(distances)
                cluster_ind = possible_clusters[ind]
                #~ print(scalar_product[ind])

                chan_sparsity = d['template_sparsity'][cluster_ind, :]
                template_sparse = templates[cluster_ind, :, :][:, chan_sparsity]
                
                # find best shift
                distance_shifts = np.zeros(d['possible_shifts'].size)
                for s, shift in enumerate(d['possible_shifts']):
                    i0 = sample_ind - d['nbefore'] + shift
                    i1 = sample_ind + d['nafter'] + shift
                    distance_shifts[s] =  np.sum((traces[i0:i1,chan_sparsity] - template_sparse)**2)
                ind_shift = np.argmin(distance_shifts)
                shift = d['possible_shifts'][ind_shift]
                
                sample_ind += shift
                i0 = sample_ind - d['nbefore']
                i1 = sample_ind + d['nafter']
                wf_sparse = traces[i0:i1, chan_sparsity]
                
                # accept or not
                centered = wf_sparse - template_sparse
                accepted = True
                for other_ind, other_vector in d['closest_units'][cluster_ind]:
                    v = np.sum(centered * other_vector)
                    if np.abs(v) >0.5:
                        accepted = False
                        break
                
                if accepted:
                    amplitude = 1.
                    
                    # remove template
                    template = templates[cluster_ind, :, :]
                    s0 = sample_ind - d['nbefore']
                    s1 = sample_ind + d['nafter']
                    traces[s0:s1, :] -= template * amplitude
                    
                else:
                    cluster_ind = -1
                    amplitude = 0.
                
            else:
                cluster_ind = -1
                amplitude = 0.
            
            spikes['cluster_ind'][i] = cluster_ind
            spikes['amplitude'][i] =amplitude
            

        return spikes    


'''
if HAVE_NUMBA:
    #@jit(parallel=True)
    @jit(nopython=True)
    def numba_sparse_dist(wf, templates, union_channels, possible_clusters):
        """
        numba implementation that compute distance from template
        """
        total_cluster, width, num_chan = templates.shape

        num_cluster = possible_clusters.shape[0]
        distances = np.zeros((num_cluster,), dtype=np.float32)
        #~ scalar_product = np.zeros((num_cluster,), dtype=np.float32)
        # for i in prange(num_cluster):
        for i in prange(num_cluster):
            cluster_ind = possible_clusters[i]
            sum_dist = 0.
            #~ sum_sp = 0.
            #~ sum_norm = 0.
            for chan_ind in range(num_chan):
                if union_channels[chan_ind]:
                    for s in range(width):
                        v = wf[s, chan_ind]
                        t = templates[cluster_ind, s, chan_ind]
                        sum_dist += (v - t) ** 2
                        #~ sum_sp += v * t
                        #~ sum_norm += t * t
            distances[i] = sum_dist
            #~ scalar_product[i] = sum_sp / sum_norm
        return distances
        #~ return scalar_product, distances
'''


#################
# Circus peeler #
#################

class CircusPeeler(BaseTemplateMatchingEngine):

    _default_params = {
        'peak_sign': 'neg',
        'n_shifts': 1,
        'spread' : 5, 
        'detect_threshold': 5,
        'noise_levels': None,
        'random_chunk_kwargs': {},
        'overlaps' : None,
        'templates' : None,
        'amplitudes' : None,
        'sparsify_threshold': 0.2 ,
        'max_amplitude' : 3,
        'min_amplitude' : 0.5,
        'use_sparse_matrix_threshold' : 0.2,
        'mcc_amplitudes': True,
        'omp' : True,
        'omp_min_sps' : 0.5,
        'omp_tol' : 1e-5,
        'progess_bar_steps' : False,
    }

    @classmethod
    def _sparsify_template(cls, template, sparse_thresholds):
        stds = np.std(template, axis=0)
        idx = np.where(stds < sparse_thresholds)[0]
        template[:, idx] = 0
        return template

    @classmethod
    def _prepare_templates(cls, d):
        
        waveform_extractor = d['waveform_extractor']
        nb_samples = d['nb_samples']
        nb_channels = d['nb_channels']
        nb_templates = d['nb_templates']
        spread = d['spread']
        max_amplitude = d['max_amplitude']
        min_amplitude = d['min_amplitude']
        use_sparse_matrix_threshold = d['use_sparse_matrix_threshold']
        sparse_thresholds = d['noise_levels'] * d['sparsify_threshold']

        norms = np.zeros(nb_templates, dtype=np.float32)
        amplitudes = np.zeros((nb_templates, 2), dtype=np.float32)

        all_units = list(d['waveform_extractor'].sorting.unit_ids)
        if d['progess_bar_steps']:
            all_units = tqdm(all_units, desc='[1] - prepare templates')

        templates = np.zeros((nb_templates,  nb_samples * nb_channels), dtype=np.float32)

        for count, unit_id in enumerate(all_units):
            w = waveform_extractor.get_waveforms(unit_id)
            snippets = w.reshape(w.shape[0], -1).T

            template = np.median(w, axis=0)
            template = cls._sparsify_template(template, sparse_thresholds)
            template = cls._denoise_template(template, snippets, d)

            norms[count] = np.linalg.norm(template)
            template /= norms[count]
            template = template.flatten()

            amps = template.dot(snippets)/norms[count]
            median_amps = np.median(amps)
            mads_amps = np.median(np.abs(amps - median_amps))
            amplitudes[count] = [max(min_amplitude, median_amps - spread*mads_amps), min(max_amplitude, median_amps + spread*mads_amps)]

            templates[count] = template

        nnz = np.sum(templates != 0)/(nb_templates * nb_samples * nb_channels)
        if nnz <= use_sparse_matrix_threshold:
            import scipy
            templates = scipy.sparse.csr_matrix(templates)

        return templates, norms, amplitudes

    @classmethod
    def _prepare_overlaps(cls, templates, d):

        nb_samples = d['nb_samples']
        nb_channels = d['nb_channels']
        nb_templates = d['nb_templates']

        is_dense = isinstance(templates, np.ndarray)

        if not is_dense:
            dense_templates = templates.toarray()
        else:
            dense_templates = templates

        dense_templates = dense_templates.reshape(nb_templates, nb_samples, nb_channels)

        size = 2 * nb_samples - 1

        all_delays = list(range(nb_samples))
        if d['progess_bar_steps']:
            all_delays = tqdm(all_delays, desc='[2] - compute overlaps')

        overlaps = {}
        
        for delay in all_delays:
            source = dense_templates[:, :delay, :].reshape(nb_templates, -1)
            target = dense_templates[:, nb_samples-delay:, :].reshape(nb_templates, -1)

            overlaps[delay] = scipy.sparse.csr_matrix(sklearn.metrics.pairwise.distance.cdist(source, target, lambda u,v :u.dot(v)))
            
            if delay < nb_samples:
                overlaps[size - delay-1] = overlaps[delay].T.tocsr()

        new_overlaps = []
        for i in range(nb_templates):
            data = [overlaps[j][i, :].T for j in range(size)]
            new_overlaps += [scipy.sparse.hstack(data).tocsc()]
        overlaps = new_overlaps

        return overlaps

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
    def _optimize_amplitudes(cls, d):

        waveform_extractor = d['waveform_extractor']
        templates = d['templates']
        nb_templates = d['nb_templates']
        max_amplitude = d['max_amplitude']
        min_amplitude = d['min_amplitude']
        alpha = 0.5
        norms = d['norms']
        all_units = list(waveform_extractor.sorting.unit_ids)
        if d['progess_bar_steps']:
            all_units = tqdm(all_units, desc='[3] - optimize amplitudes')

        amplitudes = np.zeros((nb_templates, 2), dtype=np.float32)

        all_amps = {}
        for count, unit_id in enumerate(all_units):
            w = waveform_extractor.get_waveforms(unit_id)
            snippets = w.reshape(w.shape[0], -1).T
            amps = templates.dot(snippets)/norms[:, np.newaxis]
            good = amps[count, :].flatten()

            sub_amps = amps[np.concatenate((np.arange(count), np.arange(count+1, nb_templates))), :]
            bad = sub_amps[sub_amps >= good]
            median_amps = np.median(good)
            cost_kwargs = [good, bad, max_amplitude - min_amplitude, alpha]
            cost_bounds = [(min_amplitude, median_amps), (median_amps, max_amplitude)]
            res = scipy.optimize.differential_evolution(cls._cost_function_mcc, bounds=cost_bounds, args=cost_kwargs)
            amplitudes[count] = res.x

            # import pylab as plt
            # plt.hist(good, 100, alpha=0.5)
            # plt.hist(bad, 100, alpha=0.5)
            # ymin, ymax = plt.ylim()
            # plt.plot([res.x[0], res.x[0]], [ymin, ymax], 'k--')
            # plt.plot([res.x[1], res.x[1]], [ymin, ymax], 'k--')
            # plt.savefig('test_%d.png' %count)
            # plt.close()

        return amplitudes

    @classmethod
    def _savgol_filter(cls, n, template):
        if n > 3:
            filtered_template = scipy.signal.savgol_filter(template, n, 3, axis=0)
        else:
            filtered_template = template.copy()
        return filtered_template

    @classmethod
    def _wiener_filter(cls, n, template):
        filtered_template = scipy.signal.wiener(template, (n, 1))
        return filtered_template

    @classmethod
    def _spline_filter(cls, n, template, d):
        xdata = np.arange(d['nb_samples'])
        ydata = np.arange(d['nb_channels'])
        size = len(xdata)*len(ydata)
        try:
            f = scipy.interpolate.RectBivariateSpline(xdata, ydata, template, kx=3, ky=1, s=n*size)
        except Exception:
            f = scipy.interpolate.RectBivariateSpline(xdata, ydata, template, kx=3, ky=1, s=0)
        filtered_template = f(xdata, ydata).astype(np.float32)
        return filtered_template.copy()

    @classmethod
    def _hanning_filter(cls, template, filtered_template, d):
        before = np.hanning(2*d['waveform_extractor'].nbefore)[:d['waveform_extractor'].nbefore]
        after = np.hanning(2*d['waveform_extractor'].nafter)[d['waveform_extractor'].nafter:]
        hanning = np.concatenate((before, after))[:, np.newaxis]
        return (1 - hanning)*filtered_template + hanning*template

    @classmethod
    def _cost_function_denoise(cls, n, template, snippets, d, mode='savgol', hanning=True):
        
        if mode == 'savgol':
            filtered_template = cls._savgol_filter(n, template)
        elif mode == 'spline':
            filtered_template = cls._spline_filter(n, template, d)
        elif mode == 'wiener':
            filtered_template = cls._wiener_filter(n, template)

        if hanning:
            filtered_template = cls._hanning_filter(template, filtered_template, d)

        amps = filtered_template.flatten().dot(snippets)/(np.linalg.norm(filtered_template)**2)
        median_amps = np.median(amps)
        mads_amps = np.median(np.abs(amps - np.median(amps)))
        cost = np.abs(1 - median_amps) + mads_amps
        return cost

    @classmethod
    def _denoise_template(cls, template, snippets, d, mode='savgol', hanning=True):
        nb_samples = d['nb_samples']
        nb_channels = d['nb_channels']

        if mode == 'savgol':
            indices = np.arange(3, 50, 2)
            costs = [cls._cost_function_denoise(n, template, snippets, d, 'savgol', hanning) for n in indices]
            best_idx = np.argmin(costs)
            #print(best_idx)
            filtered_template = cls._savgol_filter(indices[best_idx], template)
        elif mode == 'wiener':
            indices = np.arange(1, d['nb_samples']//2)
            costs = [cls._cost_function_denoise(n, template, snippets, d, 'wiener', hanning) for n in indices]
            best_idx = np.argmin(costs)
            #print(best_idx)
            filtered_template = cls._wiener_filter(indices[best_idx], template)
        elif mode == 'spline':
            cost_kwargs = [template, snippets, d, 'spline', hanning]
            res = scipy.optimize.fminbound(cls._cost_function_denoise, 0, 2, args=cost_kwargs)
            filtered_template = cls._spline_filter(res, template, d)

        if hanning:
            filtered_template = cls._hanning_filter(template, filtered_template, d)

        return filtered_template

    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):

        d = cls._default_params.copy()
        d.update(kwargs)

        assert isinstance(d['waveform_extractor'], WaveformExtractor)
        
        d['nb_channels'] = d['waveform_extractor'].recording.get_num_channels()
        d['nb_samples'] = d['waveform_extractor'].nsamples
        d['nb_templates'] = len(d['waveform_extractor'].sorting.unit_ids)

        if d['noise_levels'] is None:
            print('CircusPeeler : noise should be computed outside')
            d['noise_levels'] = get_noise_levels(recording)

        d['abs_threholds'] = d['noise_levels'] * d['detect_threshold']

        if d['templates'] is None:
            d['templates'], d['norms'], d['amplitudes'] = cls._prepare_templates(d)
            d['overlaps'] = cls._prepare_overlaps(d['templates'], d)


        d['nbefore'] = d['waveform_extractor'].nbefore
        d['nafter'] = d['waveform_extractor'].nafter
        d['snippet_window'] = np.arange(-d['nbefore'], d['nafter'])
        d['snippet_size'] = d['nb_channels'] * len(d['snippet_window'])

        if d['mcc_amplitudes']:
            d['amplitudes'] = cls._optimize_amplitudes(d)

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
        n_shifts = d['n_shifts']
        templates = d['templates']
        overlaps = d['overlaps']
        snippet_window = d['snippet_window']
        snippet_size = d['snippet_size']
        margin = d['margin']
        norms = d['norms']
        omp = d['omp']
        omp_tol = d['omp_tol']
        omp_min_sps = d['omp_min_sps']

        neighbor_window = len(snippet_window) - 1
        amplitudes = d['amplitudes']

        peak_traces = traces[margin // 2:-margin // 2, :]
        peak_sample_ind, peak_chan_ind = detect_peaks_by_channel(peak_traces, peak_sign, abs_threholds, n_shifts)

        peak_sample_ind, unique_idx = np.unique(peak_sample_ind, return_index=True)
        peak_sample_ind += margin // 2

        peak_chan_ind = peak_chan_ind[unique_idx]

        nb_peaks = len(peak_sample_ind)
        nb_spikes = 0

        if nb_peaks > 0:

            snippets = traces[peak_sample_ind[:, None] + snippet_window]
            snippets = snippets.reshape(nb_peaks, -1).T

            #snippets_norms = np.linalg.norm(snippets, axis=0)
            scalar_products = templates.dot(snippets)

            peaks_times = peak_sample_ind - peak_sample_ind[:, np.newaxis]

            spikes = np.empty(scalar_products.size, dtype=spike_dtype)

            if not omp:

                min_sps = (amplitudes[:, 0] * norms)[:, np.newaxis]
                max_sps = (amplitudes[:, 1] * norms)[:, np.newaxis]
                
                while True:

                    is_valid = (scalar_products > min_sps) & (scalar_products < max_sps)
                    valid_indices = np.where(is_valid)

                    if len(valid_indices[0]) == 0:
                        break

                    best_amplitude_ind = scalar_products[is_valid].argmax()
                    best_cluster_ind, peak_index = valid_indices[0][best_amplitude_ind], valid_indices[1][best_amplitude_ind]
                    best_amplitude = scalar_products[best_cluster_ind, peak_index]
                    best_peak_sample_ind = peak_sample_ind[peak_index]
                    best_peak_chan_ind = peak_chan_ind[peak_index]

                    peak_data = peaks_times[peak_index]
                    is_valid = np.searchsorted(peak_data, [-neighbor_window, neighbor_window])
                    is_neighbor = np.arange(is_valid[0], is_valid[1])
                    idx_neighbor = peak_data[is_neighbor] + neighbor_window

                    to_add = -best_amplitude * overlaps[best_cluster_ind].toarray()[:, idx_neighbor]
                    scalar_products[:, is_neighbor] += to_add
                    scalar_products[best_cluster_ind, peak_index] = -np.inf

                    spikes['sample_ind'][nb_spikes] = best_peak_sample_ind
                    spikes['channel_ind'][nb_spikes] = best_peak_chan_ind
                    spikes['cluster_ind'][nb_spikes] = best_cluster_ind
                    spikes['amplitude'][nb_spikes] = best_amplitude
                    nb_spikes += 1

                spikes['amplitude'][:nb_spikes] /= norms[spikes['cluster_ind'][:nb_spikes]]

            else:

                min_sps = amplitudes[:, 0][:, np.newaxis]
                max_sps = amplitudes[:, 1][:, np.newaxis]

                M = np.zeros((5*nb_peaks, 5*nb_peaks), dtype=np.float32)
                stop_criteria = omp_min_sps * norms[:, np.newaxis]

                all_selections = np.empty((2, scalar_products.size), dtype=np.int32, order='F')
                res_sps = np.zeros(0, dtype=np.float32)
                amplitudes = np.zeros(scalar_products.shape, dtype=np.float32)
                nb_selection = 0

                full_sps = scalar_products.copy()

                all_neighbors = np.abs(peaks_times) <= neighbor_window
                neighbors = {}
                for i in range(len(all_neighbors)):
                    idx = np.where(all_neighbors[i])[0]
                    if len(idx) > 0:
                        neighbors[i] = {'idx' : idx, 'tdx' : peaks_times[i][idx] + neighbor_window }

                while True:

                    is_valid = scalar_products > stop_criteria
                    valid_indices = np.where(is_valid)

                    if len(valid_indices[0]) == 0:
                        break

                    best_amplitude_ind = scalar_products[is_valid].argmax()
                    best_cluster_ind, peak_index = valid_indices[0][best_amplitude_ind], valid_indices[1][best_amplitude_ind]
                
                    all_selections[:, nb_selection] = [best_cluster_ind, peak_index]
                    nb_selection += 1
                    selection = all_selections[:, :nb_selection]
        
                    res_sps = full_sps[selection[0], selection[1]]
                    scalar_products[best_cluster_ind, peak_index] = -np.inf

                    delta_t = peak_sample_ind[selection[1]] - peak_sample_ind[selection[1, -1]]
                    idx = np.where(np.abs(delta_t) <= neighbor_window)[0]

                    myline = neighbor_window + delta_t[idx]
                    line_1 = overlaps[selection[0, -1]].toarray()[selection[0, idx], myline]
                    M[nb_selection-1, idx] = line_1

                    if nb_selection >= (M.shape[0] - 1):
                        Z = np.zeros((2*M.shape[0], 2*M.shape[1]), dtype=np.float32)
                        Z[:nb_selection, :nb_selection] = M[:nb_selection, :nb_selection]
                        M = Z

                    all_amplitudes = scipy.linalg.solve(M[:nb_selection, :nb_selection], res_sps, assume_a='sym', check_finite=False, lower=True, overwrite_b=True)/norms[selection[0]]
                    diff_amplitudes = (all_amplitudes - amplitudes[selection[0], selection[1]])
                    modified = np.where(np.abs(diff_amplitudes) > omp_tol)[0]
                    amplitudes[selection[0], selection[1]] = all_amplitudes

                    for i in modified:

                        tmp_best, tmp_peak = selection[:, i]
                        
                        if tmp_best in neighbors:
                            diff_amp = diff_amplitudes[i]*norms[tmp_best]
                            idx = neighbors[tmp_peak]['idx']
                            tdx = neighbors[tmp_peak]['tdx']
                            scalar_products[:, idx] -= diff_amp * overlaps[tmp_best].toarray()[:, tdx]

                is_valid = (amplitudes > min_sps)*(amplitudes < max_sps)
                valid_indices = np.where(is_valid)

                nb_spikes = len(valid_indices[0])
                spikes['sample_ind'][:nb_spikes] = peak_sample_ind[valid_indices[1]]
                spikes['channel_ind'][:nb_spikes] = peak_chan_ind[valid_indices[1]]
                spikes['cluster_ind'][:nb_spikes] = valid_indices[0]
                spikes['amplitude'][:nb_spikes] = amplitudes[is_valid]

            spikes = spikes[:nb_spikes]

            order = np.argsort(spikes['sample_ind'])
            spikes = spikes[order]

        else:
            spikes = np.zeros(0, dtype=spike_dtype)

        return spikes

template_matching_methods = {
    'naive' : NaiveMatching,
    'tridesclous' : TridesclousPeeler,
    'circus' : CircusPeeler
}
