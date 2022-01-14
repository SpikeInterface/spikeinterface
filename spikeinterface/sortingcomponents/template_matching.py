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

from spikeinterface.sortingcomponents.peak_detection import detect_peak_locally_exclusive, detect_peaks_by_channel

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
        
    


##########
# Circus peeler
##########


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
        'sparsify_threshold': 0.2,
        'max_amplitude' : 3,
        'use_sparse_matrix_threshold' : 0.2,
        'mcc_amplitudes': True,
        'omp' : True
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
        use_sparse_matrix_threshold = d['use_sparse_matrix_threshold']
        sparse_thresholds = d['noise_levels'] * d['sparsify_threshold']

        norms = np.zeros(nb_templates, dtype=np.float32)
        amplitudes = np.zeros((nb_templates, 2), dtype=np.float32)

        from tqdm import tqdm

        all_units = tqdm(d['waveform_extractor'].sorting.unit_ids, desc='prepare templates')

        templates = np.zeros((nb_templates,  nb_samples * nb_channels), dtype=np.float32)

        for count, unit_id in enumerate(all_units):
            w = waveform_extractor.get_waveforms(unit_id)

            template = np.median(w, axis=0)
            template = cls._sparsify_template(template, sparse_thresholds)
            norms[count] = np.linalg.norm(template)
            template /= norms[count]
            template = template.flatten()

            amps = template.dot(w.reshape(w.shape[0], -1).T)/norms[count]
            median_amps = np.median(amps)
            mads_amps = np.median(np.abs(amps - np.median(amps)))
            amplitudes[count] = [max(0.5, median_amps - spread*mads_amps), min(max_amplitude, median_amps+spread*mads_amps)]

            # Necessary to clear cache of the waveform_extractor
            waveform_extractor._waveforms = {}

            templates[count] = template

        nnz = np.sum(templates != 0)/(nb_templates * nb_samples * nb_channels)
        if nnz <= use_sparse_matrix_threshold:
            import scipy
            templates = scipy.sparse.csr_matrix(templates)

        return templates, norms, amplitudes

    @classmethod
    def _prepare_overlaps(cls, templates, d):

        import sklearn, scipy
        from tqdm import tqdm

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

        all_delays = tqdm(range(nb_samples), desc='compute overlaps')

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

    def _mcc_error(good_values, bad_values, bounds):
        fn = np.sum((good_values < bounds[0]) | (good_values > bounds[1]))
        fp = np.sum((bounds[0] <= bad_values) & (bad_values <= bounds[1]))
        tp = np.sum((bounds[0] <= good_values) & (good_values <= bounds[1]))
        tn = np.sum((bad_values < bounds[0]) | (bad_values > bounds[1]))
        denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
        if denom > 0:
            mcc = 1 - (tp*tn - fp*fn)/np.sqrt(denom)
        else:
            mcc = 1
        return mcc

    # def _cost_function(x, good_values, bad_values, alpha=1e-1, max_amplitude=3):
    #     # We want a minimal error, with the larger bounds that are possible
    #     cost = _mcc_error(good_values, bad_values, x) + alpha*np.abs((max_amplitude - (x[1] - x[0])))
    #     return cost


    # def _optimize_amplitudes(self):
    #     import scipy
    #     from tqdm import tqdm
    #     all_units = tqdm(self.waveform_extractor.sorting.unit_ids, desc='optimize amplitudes')
    #     for count, unit_id in enumerate(all_units):
    #         w = self.waveform_extractor.get_waveforms(unit_id)
    #         if self.sparse:
    #             self.waveform_extractor._waveforms = {}
    #         amps = self.templates.dot(w.reshape(w.shape[0], -1).T)/self.norms[:, np.newaxis]
    #         good_values = amps[count, :].flatten()
    #         bad_values = amps[np.concatenate((np.arange(count), np.arange(count+1, self.nb_templates))), :].flatten()
    #         res = scipy.optimize.differential_evolution(self._cost_function, bounds=[(0,1), (1, self.max_amplitude)], args=(good_values, bad_values))
    #         self.amplitudes[count] = res.x


    
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

        #if mcc_amplitudes:
        #    self.optimize_amplitudes()

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
            snippets = snippets.reshape(nb_peaks, -1)

            scalar_products = templates.dot(snippets.T)

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
                    best_amplitude_ = best_amplitude / norms[best_cluster_ind]
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
                    spikes['amplitude'][nb_spikes] = best_amplitude_
                    nb_spikes += 1

            else:

                min_sps = amplitudes[:, 0][:, np.newaxis]
                max_sps = amplitudes[:, 1][:, np.newaxis]

                import scipy

                M = np.zeros((10*nb_peaks, 10*nb_peaks), dtype=np.float32)
                stop_criteria = 0.5 * norms[:, np.newaxis]
                error_tol = 1e-5

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

                    all_amplitudes = scipy.linalg.solve(M[:nb_selection, :nb_selection], res_sps, assume_a='sym', check_finite=False, lower=True)/norms[selection[0]]
                    diff_amplitudes = (all_amplitudes - amplitudes[selection[0], selection[1]])
                    modified = np.where(np.abs(diff_amplitudes) > error_tol)[0]
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

