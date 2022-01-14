"""Sorting components: template matching."""

import numpy as np

import scipy.spatial

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

from spikeinterface.sortingcomponents.peak_detection import detect_peak_locally_exclusive

spike_dtype = [('sample_ind', 'int64'), ('channel_ind', 'int64'), ('cluster_ind', 'int64'),
               ('amplitude', 'float64'), ('segment_ind', 'int64')]



def find_spikes_from_templates(recording, method='simple', method_kwargs={}, extra_ouputs=False,
                              **job_kwargs):
    """Find spike from a recording from given templates.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object.
    waveform_extractor: WaveformExtractor
        The waveform extractor.
    method: {'simple'}
        Which method to use.
    method_kwargs: dict, optional
        Keyword arguments for the chosen method.
    extra_ouputs: bool
        If True then method_kwargs is also return
    job_kwargs: dict
        Parameters for ChunkRecordingExecutor.

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
    func = _find_spike_chunk
    init_func = _init_worker_find_spike
    init_args = (recording.to_dict(), method, method_kwargs_seralized)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args,
                                       handle_returns=True, job_name=f'find spikes {method}', **job_kwargs)
    spikes = processor.run()

    spikes = np.concatenate(spikes)
    
    if extra_ouputs:
        return spikes, method_kwargs
    else:
        return spikes


def _init_worker_find_spike(recording, method, method_kwargs):
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


def _find_spike_chunk(segment_index, start_frame, end_frame, worker_ctx):
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
        # need to be overwrite in subclass
        raise NotImplementedError
        # this function before loops

    @classmethod
    def serialize_method_kwargs(cls, kwargs):
        # need to be overwrite in subclass
        raise NotImplementedError
        # this serializa params to distribute it to workers

    @classmethod
    def unserialize_in_worker(cls, recording, kwargs):
        # need to be overwrite in subclass
        raise NotImplementedError
        # this in worker at init to unserialize some wkargs if necessary

    @classmethod
    def get_margin(cls, recording, kwargs):
        # need to be overwrite in subclass
        raise NotImplementedError
        # this must return number of sample for margin


    @classmethod
    def main_function(cls, traces, method_kwargs):
        # need to be overwrite in subclass
        raise NotImplementedError
        # this is the main function to detect and label spikes
        # this return spikes in traces chunk


    

##########
# naive mathing
##########


class NaiveMatching(BaseTemplateMatchingEngine):
    """
    This is a naive template matching that do not resolve collision
    and do not take in account sparsity.
    It just minimize the dist to templates for detected peaks.

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


##########
# tridesclous peeler
##########


class TridesclousPeeler(BaseTemplateMatchingEngine):
    default_params = {
        'waveform_extractor': None,
        'peak_sign': 'neg',
        'peak_shift_ms':  0.2,
        'detect_threshold': 5,
        'noise_levels': None,
        'local_radius_um': 100,
        'num_closest' : 3,
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
            for u in closest_u:
                vec = (templates[u, :, :][:, chans] - template_sparse)
                vec /= np.sum(vec ** 2)
                closest_vec.append((u, vec))
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
            
            _tdc_remove_spikes(traces, spikes, d)
            
            level += 1
            
            if level == 2:
                break

        all_spikes = np.concatenate(all_spikes)
        order = np.argsort(all_spikes['sample_ind'])
        all_spikes = all_spikes[order]
        
        return all_spikes


def _tdc_remove_spikes(traces, spikes, d):
    nbefore, nafter = d['nbefore'], d['nafter']
    for spike in spikes:
        if spike['cluster_ind'] < 0:
            continue
        template = d['templates'][spike['cluster_ind'], :, :]
        s0 = spike['sample_ind'] - d['nbefore']
        s1 = spike['sample_ind'] + d['nafter']
        traces[s0:s1, :] -= template * spike['amplitude']
    

def _tdc_find_spikes(traces, d, level=0):
        peak_sign = d['peak_sign']
        templates = d['templates']
        margin = d['margin']
        possible_clusters_by_channel = d['possible_clusters_by_channel']
        
        
        peak_traces = traces[margin // 2:-margin // 2, :]
        peak_sample_ind, peak_chan_ind = detect_peak_locally_exclusive(peak_traces, peak_sign,
                                    d['abs_threholds'], d['peak_shift'], d['neighbours_mask'])
        peak_sample_ind += margin // 2


        spikes = np.zeros(peak_sample_ind.size, dtype=spike_dtype)
        spikes['sample_ind'] = peak_sample_ind
        spikes['channel_ind'] = peak_chan_ind  # TODO need to put the channel from template
        
        # naively take the closest template
        for i in range(peak_sample_ind.size):
            i0 = peak_sample_ind[i] - d['nbefore']
            i1 = peak_sample_ind[i] + d['nafter']

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
                #~ union_channels = np.any(d['template_sparsity'][possible_clusters, :], axis=0)
                #~ scalar_product, distances = numba_sparse_dist(wf, templates, union_channels, possible_clusters)
                #~ distances = numba_sparse_dist(wf, templates, union_channels, possible_clusters)
                #~ print(scalar_product)
                
                ind = np.argmin(distances)
                cluster_ind = possible_clusters[ind]
                #~ print(scalar_product[ind])
                
                # accept or not
                chan_sparsity = d['template_sparsity'][cluster_ind, :]
                template_sparse = templates[cluster_ind, :, :][:, chan_sparsity]
                wf_sparse = wf[:, chan_sparsity]
                centered = wf_sparse - template_sparse
                
                accepted = True
                for other_ind, other_vector in d['closest_units'][cluster_ind]:
                    v = np.sum(centered * other_vector)
                    if np.abs(v) >0.5:
                        accepted = False
                        break
                
                if accepted:
                #~ if 1.:
                    amplitude = 1.
                else:
                    cluster_ind = -1
                    amplitude = 0.
                
            else:
                cluster_ind = -1
                amplitude = 0.
            
            spikes['cluster_ind'][i] = cluster_ind
            spikes['amplitude'][i] =amplitude
            

        return spikes    



if HAVE_NUMBA:
    @jit(parallel=True)
    def numba_sparse_dist(wf, templates, union_channels, possible_clusters):
        """
        numba implementation that compute distance from template
        """
        total_cluster, width, num_chan = templates.shape
        num_cluster = possible_clusters.shape[0]
        distances = np.zeros((num_cluster,), dtype=np.float32)
        #~ scalar_product = np.zeros((num_cluster,), dtype=np.float32)
        for i in prange(num_cluster):
        #~ for i in prange(num_cluster):
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
        
    



template_matching_methods = {
    'naive' : NaiveMatching,
    'tridesclous' : TridesclousPeeler,
}

