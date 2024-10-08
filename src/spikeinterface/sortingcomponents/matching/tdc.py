from __future__ import annotations

import numpy as np
from spikeinterface.core import (
    get_noise_levels,
    get_channel_distances,
    compute_sparsity,
    get_template_extremum_channel,
)

from spikeinterface.sortingcomponents.peak_detection import DetectPeakLocallyExclusive, DetectPeakMatchedFiltering
from spikeinterface.core.template import Templates

from .base import BaseTemplateMatching, _base_matching_dtype


try:
    import numba
    from numba import jit, prange

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


class TridesclousPeeler(BaseTemplateMatching):
    """
    Template-matching ported from Tridesclous sorter.

    The idea of this peeler is pretty simple.
    1. Find peaks
    2. order by best amplitues
    3. find nearest template
    4. remove it from traces.
    5. in the residual find peaks again

    This method is quite fast but don't give exelent results to resolve
    spike collision when templates have high similarity.
    """

    def __init__(
        self,
        recording,
        return_output=True,
        parents=None,
        templates=None,
        peak_sign="neg",
        peak_shift_ms=0.2,
        detect_threshold=5,
        noise_levels=None,
        radius_um=100.0,
        num_closest=5,
        sample_shift=3,
        ms_before=0.8,
        ms_after=1.2,
        num_peeler_loop=2,
        num_template_try=1,
    ):

        BaseTemplateMatching.__init__(self, recording, templates, return_output=True, parents=None)

        # maybe in base?
        self.templates_array = templates.get_dense_templates()

        unit_ids = templates.unit_ids
        channel_ids = recording.channel_ids

        sr = recording.sampling_frequency

        self.nbefore = templates.nbefore
        self.nafter = templates.nafter

        self.peak_sign = peak_sign

        nbefore_short = int(ms_before * sr / 1000.0)
        nafter_short = int(ms_after * sr / 1000.0)
        assert nbefore_short <= templates.nbefore
        assert nafter_short <= templates.nafter
        self.nbefore_short = nbefore_short
        self.nafter_short = nafter_short
        s0 = templates.nbefore - nbefore_short
        s1 = -(templates.nafter - nafter_short)
        if s1 == 0:
            s1 = None
        # TODO check with out copy
        self.templates_short = self.templates_array[:, slice(s0, s1), :].copy()

        self.peak_shift = int(peak_shift_ms / 1000 * sr)

        assert noise_levels is not None, "TridesclousPeeler : noise should be computed outside"

        self.abs_thresholds = noise_levels * detect_threshold

        channel_distance = get_channel_distances(recording)
        self.neighbours_mask = channel_distance < radius_um

        if templates.sparsity is not None:
            self.template_sparsity = templates.sparsity.mask
        else:
            self.template_sparsity = np.ones((unit_ids.size, channel_ids.size), dtype=bool)

        extremum_chan = get_template_extremum_channel(templates, peak_sign=peak_sign, outputs="index")
        # as numpy vector
        self.extremum_channel = np.array([extremum_chan[unit_id] for unit_id in unit_ids], dtype="int64")

        channel_locations = templates.probe.contact_positions
        unit_locations = channel_locations[self.extremum_channel]

        # distance between units
        import scipy

        unit_distances = scipy.spatial.distance.cdist(unit_locations, unit_locations, metric="euclidean")

        # seach for closet units and unitary discriminant vector
        closest_units = []
        for unit_ind, unit_id in enumerate(unit_ids):
            order = np.argsort(unit_distances[unit_ind, :])
            closest_u = np.arange(unit_ids.size)[order].tolist()
            closest_u.remove(unit_ind)
            closest_u = np.array(closest_u[:num_closest])

            # compute unitary discriminent vector
            (chans,) = np.nonzero(self.template_sparsity[unit_ind, :])
            template_sparse = self.templates_array[unit_ind, :, :][:, chans]
            closest_vec = []
            # against N closets
            for u in closest_u:
                vec = self.templates_array[u, :, :][:, chans] - template_sparse
                vec /= np.sum(vec**2)
                closest_vec.append((u, vec))
            # against noise
            closest_vec.append((None, -template_sparse / np.sum(template_sparse**2)))

            closest_units.append(closest_vec)

        self.closest_units = closest_units

        # distance channel from unit
        import scipy

        distances = scipy.spatial.distance.cdist(channel_locations, unit_locations, metric="euclidean")
        near_cluster_mask = distances < radius_um

        # nearby cluster for each channel
        self.possible_clusters_by_channel = []
        for channel_index in range(distances.shape[0]):
            (cluster_inds,) = np.nonzero(near_cluster_mask[channel_index, :])
            self.possible_clusters_by_channel.append(cluster_inds)

        self.possible_shifts = np.arange(-sample_shift, sample_shift + 1, dtype="int64")

        self.num_peeler_loop = num_peeler_loop
        self.num_template_try = num_template_try

        self.margin = max(self.nbefore, self.nafter) * 2

    def get_trace_margin(self):
        return self.margin

    def compute_matching(self, traces, start_frame, end_frame, segment_index):
        traces = traces.copy()

        all_spikes = []
        level = 0
        while True:
            # spikes = _tdc_find_spikes(traces, d, level=level)
            spikes = self._find_spikes_one_level(traces, level=level)
            keep = spikes["cluster_index"] >= 0

            if not np.any(keep):
                break
            all_spikes.append(spikes[keep])

            level += 1

            if level == self.num_peeler_loop:
                break

        if len(all_spikes) > 0:
            all_spikes = np.concatenate(all_spikes)
            order = np.argsort(all_spikes["sample_index"])
            all_spikes = all_spikes[order]
        else:
            all_spikes = np.zeros(0, dtype=_base_matching_dtype)

        return all_spikes

    def _find_spikes_one_level(self, traces, level=0):

        peak_traces = traces[self.margin // 2 : -self.margin // 2, :]
        peak_sample_ind, peak_chan_ind = DetectPeakLocallyExclusive.detect_peaks(
            peak_traces, self.peak_sign, self.abs_thresholds, self.peak_shift, self.neighbours_mask
        )
        peak_sample_ind += self.margin // 2

        peak_amplitude = traces[peak_sample_ind, peak_chan_ind]
        order = np.argsort(np.abs(peak_amplitude))[::-1]
        peak_sample_ind = peak_sample_ind[order]
        peak_chan_ind = peak_chan_ind[order]

        spikes = np.zeros(peak_sample_ind.size, dtype=_base_matching_dtype)
        spikes["sample_index"] = peak_sample_ind
        spikes["channel_index"] = peak_chan_ind  # TODO need to put the channel from template

        possible_shifts = self.possible_shifts
        distances_shift = np.zeros(possible_shifts.size)

        for i in range(peak_sample_ind.size):
            sample_index = peak_sample_ind[i]

            chan_ind = peak_chan_ind[i]
            possible_clusters = self.possible_clusters_by_channel[chan_ind]

            if possible_clusters.size > 0:
                # ~ s0 = sample_index - d['nbefore']
                # ~ s1 = sample_index + d['nafter']

                # ~ wf = traces[s0:s1, :]

                s0 = sample_index - self.nbefore_short
                s1 = sample_index + self.nafter_short
                wf_short = traces[s0:s1, :]

                ## pure numpy with cluster spasity
                # distances = np.sum(np.sum((templates[possible_clusters, :, :] - wf[None, : , :])**2, axis=1), axis=1)

                ## pure numpy with cluster+channel spasity
                # union_channels, = np.nonzero(np.any(d['template_sparsity'][possible_clusters, :], axis=0))
                # distances = np.sum(np.sum((templates[possible_clusters][:, :, union_channels] - wf[: , union_channels][None, : :])**2, axis=1), axis=1)

                ## numba with cluster+channel spasity
                union_channels = np.any(self.template_sparsity[possible_clusters, :], axis=0)
                # distances = numba_sparse_dist(wf, templates, union_channels, possible_clusters)
                distances = numba_sparse_dist(wf_short, self.templates_short, union_channels, possible_clusters)

                # DEBUG
                # ~ ind = np.argmin(distances)
                # ~ cluster_index = possible_clusters[ind]

                for ind in np.argsort(distances)[: self.num_template_try]:
                    cluster_index = possible_clusters[ind]

                    chan_sparsity = self.template_sparsity[cluster_index, :]
                    template_sparse = self.templates_array[cluster_index, :, :][:, chan_sparsity]

                    # find best shift

                    ## pure numpy version
                    # for s, shift in enumerate(possible_shifts):
                    #     wf_shift = traces[s0 + shift: s1 + shift, chan_sparsity]
                    #     distances_shift[s] = np.sum((template_sparse - wf_shift)**2)
                    # ind_shift = np.argmin(distances_shift)
                    # shift = possible_shifts[ind_shift]

                    ## numba version
                    numba_best_shift(
                        traces,
                        self.templates_array[cluster_index, :, :],
                        sample_index,
                        self.nbefore,
                        possible_shifts,
                        distances_shift,
                        chan_sparsity,
                    )
                    ind_shift = np.argmin(distances_shift)
                    shift = possible_shifts[ind_shift]

                    sample_index = sample_index + shift
                    s0 = sample_index - self.nbefore
                    s1 = sample_index + self.nafter
                    wf_sparse = traces[s0:s1, chan_sparsity]

                    # accept or not

                    centered = wf_sparse - template_sparse
                    accepted = True
                    for other_ind, other_vector in self.closest_units[cluster_index]:
                        v = np.sum(centered * other_vector)
                        if np.abs(v) > 0.5:
                            accepted = False
                            break

                    if accepted:
                        # ~ if ind != np.argsort(distances)[0]:
                        # ~ print('not first one', np.argsort(distances), ind)
                        break

                if accepted:
                    amplitude = 1.0

                    # remove template
                    template = self.templates_array[cluster_index, :, :]
                    s0 = sample_index - self.nbefore
                    s1 = sample_index + self.nafter
                    traces[s0:s1, :] -= template * amplitude

                else:
                    cluster_index = -1
                    amplitude = 0.0

            else:
                cluster_index = -1
                amplitude = 0.0

            spikes["cluster_index"][i] = cluster_index
            spikes["amplitude"][i] = amplitude

        return spikes


if HAVE_NUMBA:

    @jit(nopython=True)
    def numba_sparse_dist(wf, templates, union_channels, possible_clusters):
        """
        numba implementation that compute distance from template with sparsity
        handle by two separate vectors
        """
        total_cluster, width, num_chan = templates.shape
        num_cluster = possible_clusters.shape[0]
        distances = np.zeros((num_cluster,), dtype=np.float32)
        for i in prange(num_cluster):
            cluster_index = possible_clusters[i]
            sum_dist = 0.0
            for chan_ind in range(num_chan):
                if union_channels[chan_ind]:
                    for s in range(width):
                        v = wf[s, chan_ind]
                        t = templates[cluster_index, s, chan_ind]
                        sum_dist += (v - t) ** 2
            distances[i] = sum_dist
        return distances

    @jit(nopython=True)
    def numba_best_shift(traces, template, sample_index, nbefore, possible_shifts, distances_shift, chan_sparsity):
        """
        numba implementation to compute several sample shift before template substraction
        """
        width, num_chan = template.shape
        n_shift = possible_shifts.size
        for i in range(n_shift):
            shift = possible_shifts[i]
            sum_dist = 0.0
            for chan_ind in range(num_chan):
                if chan_sparsity[chan_ind]:
                    for s in range(width):
                        v = traces[sample_index - nbefore + s + shift, chan_ind]
                        t = template[s, chan_ind]
                        sum_dist += (v - t) ** 2
            distances_shift[i] = sum_dist

        return distances_shift




class TridesclousPeeler2(BaseTemplateMatching):
    """
    Template-matching used by Tridesclous sorter.

    """
    def __init__(self, recording, return_output=True, parents=None,
        templates=None,
        peak_sign="neg",
        exclude_sweep_ms=0.5,
        peak_shift_ms=0.2,
        detect_threshold=5,
        noise_levels=None,
        # TODO optimize theses radius
        detection_radius_um=100.,
        cluster_radius_um=150.,
        amplitude_radius_um=200.,

        sample_shift=2,
        ms_before=0.5,
        ms_after=0.8,
        max_peeler_loop=3,
        amplitude_limits=(0.7, 1.4),
        ):

        BaseTemplateMatching.__init__(self, recording, templates, return_output=True, parents=None)

        unit_ids = templates.unit_ids
        channel_ids = recording.channel_ids

        num_templates = unit_ids.size

        sr = recording.sampling_frequency

        self.nbefore = templates.nbefore
        self.nafter = templates.nafter
        
        self.peak_sign = peak_sign

        nbefore_short = int(ms_before * sr / 1000.0)
        nafter_short = int(ms_after * sr / 1000.0)
        assert nbefore_short <= templates.nbefore
        assert nafter_short <= templates.nafter
        self.nbefore_short = nbefore_short
        self.nafter_short = nafter_short
        s0 = templates.nbefore - nbefore_short
        s1 = -(templates.nafter - nafter_short)
        if s1 == 0:
            s1 = None

        # TODO check with out copy
        # self.dense_templates_array = templates.get_dense_templates()
        # self.dense_templates_array_short = self.dense_templates_array[:, slice(s0, s1), :].copy()
        self.sparse_templates_array_short = templates.templates_array[:, slice(s0, s1), :].copy()

        self.peak_shift = int(peak_shift_ms / 1000 * sr)

        assert noise_levels is not None, "TridesclousPeeler : noise should be computed outside"

        self.abs_thresholds = noise_levels * detect_threshold

        channel_distance = get_channel_distances(recording)
        self.neighbours_mask = channel_distance <= detection_radius_um

        if templates.sparsity is not None:
            self.sparsity_mask = templates.sparsity.mask
        else:
            self.sparsity_mask = np.ones((unit_ids.size, channel_ids.size), dtype=bool)

        extremum_chan = get_template_extremum_channel(templates, peak_sign=peak_sign, outputs="index")
        # as numpy vector
        self.extremum_channel = np.array([extremum_chan[unit_id] for unit_id in unit_ids], dtype="int64")

        channel_locations = templates.probe.contact_positions
        unit_locations = channel_locations[self.extremum_channel]

        # distance between units
        import scipy

        # nearby cluster for each channel
        distances = scipy.spatial.distance.cdist(channel_locations, unit_locations, metric="euclidean")
        near_cluster_mask = distances <= cluster_radius_um
        self.possible_clusters_by_channel = []
        for channel_index in range(distances.shape[0]):
            (cluster_inds,) = np.nonzero(near_cluster_mask[channel_index, :])
            self.possible_clusters_by_channel.append(cluster_inds)


        self.template_norms = np.zeros(num_templates, dtype="float32")
        for i in range(unit_ids.size):
            chan_mask = self.sparsity_mask[i, :]
            n = np.sum(chan_mask)
            template = templates.templates_array[i, :, :n]
            self.template_norms[i] = np.sum(template ** 2)

        # template = sparse_templates_array[cluster_index, :, :num_chans]
        # wf = traces[start: stop, :][:, chan_sparsity_mask]
        # # TODO precompute template norms
        # amplitude = np.sum(template.flatten() * wf.flatten()) / np.sum(template.flatten()**2)
    

        # 
        distances = scipy.spatial.distance.cdist(channel_locations, channel_locations, metric="euclidean")
        self.near_chan_mask = distances <= amplitude_radius_um

        self.possible_shifts = np.arange(-sample_shift, sample_shift + 1, dtype="int64")

        self.max_peeler_loop = max_peeler_loop
        self.amplitude_limits = amplitude_limits

        


        self.peak_detector_level0 = DetectPeakLocallyExclusive(
            recording=recording,
            peak_sign=peak_sign,
            detect_threshold=detect_threshold,
            exclude_sweep_ms=exclude_sweep_ms,
            radius_um=detection_radius_um,
            noise_levels=noise_levels,
        )
        
        ##get prototype from best channel of each template
        prototype = np.zeros(self.nbefore+self.nafter, dtype='float32')
        for i in range(num_templates):
            template = templates.templates_array[i, :, :]
            chan_ind = np.argmax(np.abs(template[self.nbefore, :]))
            if template[self.nbefore, chan_ind] != 0:
                prototype += template[:, chan_ind] / np.abs(template[self.nbefore, chan_ind])
        prototype /= np.abs(prototype[self.nbefore])

        # import matplotlib.pyplot as plt
        # fig,ax = plt.subplots()
        # ax.plot(prototype)    
        # plt.show()
        
        self.peak_detector_level1 = DetectPeakMatchedFiltering(
            recording=recording,
            prototype=prototype,
            ms_before=templates.nbefore / sr * 1000.,
            peak_sign="neg",
            detect_threshold=detect_threshold,
            exclude_sweep_ms=exclude_sweep_ms,
            radius_um=detection_radius_um,
            rank=1,
            noise_levels=noise_levels,
        )
        
        # TODO max maargin detector
        self.detector_margin0 = self.peak_detector_level0.get_trace_margin()
        self.detector_margin1 = self.peak_detector_level1.get_trace_margin()
        self.peeler_margin = max(self.nbefore, self.nafter) * 2
        self.margin = max(self.peeler_margin,  self.detector_margin0, self.detector_margin1)



    def get_trace_margin(self):
        return self.margin

    def compute_matching(self, traces, start_frame, end_frame, segment_index):
        
        # TODO check if this is usefull
        traces = traces.copy()

        all_spikes = []
        level = 0
        spikes_prev_loop = np.zeros(0, dtype=_base_matching_dtype)
        while True:
            # print('level', level)
            spikes = self._find_spikes_one_level(traces, spikes_prev_loop, level=level)
            if not np.any(spikes.size):
                break
            all_spikes.append(spikes)

            level += 1

            if level == self.max_peeler_loop:
                break
        
            spikes_prev_loop = spikes

        if len(all_spikes) > 0:
            all_spikes = np.concatenate(all_spikes)
            order = np.argsort(all_spikes["sample_index"])
            all_spikes = all_spikes[order]
        else:
            all_spikes = np.zeros(0, dtype=_base_matching_dtype)

        return all_spikes

    def _find_spikes_one_level(self, traces, spikes_prev_loop, level=0):

        # TODO change the threhold dynaically depending the level
        # peak_traces = traces[self.detector_margin : -self.detector_margin, :]
        
        # peak_sample_ind, peak_chan_ind = DetectPeakLocallyExclusive.detect_peaks(
        #     peak_traces, self.peak_sign, self.abs_thresholds, self.peak_shift, self.neighbours_mask
        # )

        
        if level == 0:
            peak_detector = self.peak_detector_level0
        else:
            peak_detector = self.peak_detector_level1

        detector_margin = peak_detector.get_trace_margin()
        if self.peeler_margin > detector_margin:
            margin_shift = self.peeler_margin - detector_margin
            sl = slice(margin_shift, -margin_shift)
        else:
            sl = slice(None)
            margin_shift = 0
        peak_traces = traces[sl, :]
        peaks, = peak_detector.compute(peak_traces, None, None, 0, self.margin)
        peak_sample_ind = peaks["sample_index"]
        peak_chan_ind = peaks["channel_index"]
        peak_sample_ind += margin_shift





        peak_amplitude = traces[peak_sample_ind, peak_chan_ind]
        order = np.argsort(np.abs(peak_amplitude))[::-1]
        peak_sample_ind = peak_sample_ind[order]
        peak_chan_ind = peak_chan_ind[order]

        spikes = np.zeros(peak_sample_ind.size, dtype=_base_matching_dtype)
        spikes["sample_index"] = peak_sample_ind
        spikes["channel_index"] = peak_chan_ind

        distances_shift = np.zeros(self.possible_shifts.size)

        delta_sample = max(self.nbefore, self.nafter) #  TODO check this maybe add margin
        # neighbors_spikes_inds = get_neighbors_spikes(spikes["sample_index"], spikes["channel_index"], delta_sample, self.near_chan_mask)

        # neighbors in actual and previous level
        neighbors_spikes_inds = get_neighbors_spikes(
            np.concatenate([spikes["sample_index"], spikes_prev_loop["sample_index"]]),
            np.concatenate([spikes["channel_index"], spikes_prev_loop["channel_index"]]),
            delta_sample, self.near_chan_mask)


        for i in range(spikes.size):
            sample_index = peak_sample_ind[i]

            chan_ind = peak_chan_ind[i]
            possible_clusters = self.possible_clusters_by_channel[chan_ind]

            if possible_clusters.size > 0:
                cluster_index = get_most_probable_cluster(traces, self.sparse_templates_array_short, possible_clusters,
                                          sample_index, chan_ind, self.nbefore_short, self.nafter_short, self.sparsity_mask)
                

                chan_sparsity_mask = self.sparsity_mask[cluster_index, :]

                # find best shift
                numba_best_shift_sparse(
                    traces,
                    self.sparse_templates_array_short[cluster_index, :, :],
                    sample_index,
                    self.nbefore_short,
                    self.possible_shifts,
                    distances_shift,
                    chan_sparsity_mask,
                )
                
                ind_shift = np.argmin(distances_shift)
                shift = self.possible_shifts[ind_shift]

                # TODO DEBUG shift later
                spikes["sample_index"][i] += shift

                spikes["cluster_index"][i] = cluster_index


                # check that the the same cluster is not already detected at same place
                # this can happen for small template the substract forvever the traces
                outer_neighbors_inds = [ ind for ind in neighbors_spikes_inds[i] if ind>i and ind >= spikes.size]
                is_valid = True
                for b in outer_neighbors_inds:
                    b = b - spikes.size
                    if (spikes[i]["sample_index"] == spikes_prev_loop[b]["sample_index"]) and \
                        (spikes[i]["cluster_index"] == spikes_prev_loop[b]["cluster_index"]):
                        is_valid = False

                if is_valid:
                    # temporary assign a cluster to neighbors if not done yet
                    inner_neighbors_inds = [ ind for ind in neighbors_spikes_inds[i] if (ind>i and ind < spikes.size)]
                    for b in inner_neighbors_inds:
                        spikes["cluster_index"][b] = get_most_probable_cluster(
                            traces, self.sparse_templates_array_short, possible_clusters,
                            spikes["sample_index"][b], spikes["channel_index"][b], self.nbefore_short,
                            self.nafter_short, self.sparsity_mask
                        )

                    amp = fit_one_amplitude_with_neighbors(spikes[i], spikes[inner_neighbors_inds],  traces, 
                                                    self.sparsity_mask, self.templates.templates_array,
                                                    self.template_norms,
                                                    self.nbefore, self.nafter)
                    
                    low_lim, up_lim = self.amplitude_limits
                    if ( low_lim <= amp <= up_lim):
                        spikes["amplitude"][i] = amp
                        wanted_channel_mask = np.ones(traces.shape[1], dtype=bool) # TODO move this before the loop
                        construct_prediction_sparse(spikes[i:i+1], traces, self.templates.templates_array,
                                                    self.sparsity_mask, wanted_channel_mask,
                                                    self.nbefore, additive=False)
                    elif low_lim > amp:
                        # print("bad amp", amp)
                        spikes["cluster_index"][i] = -1
                    else:
                        # amp > up_lim
                        # TODO should try other cluster for the fit!!
                        # spikes["cluster_index"][i] = -1

                        # force amplitude to be one and need a fiting at next level
                        spikes["amplitude"][i] = 1

                        # print(amp)
                        # import matplotlib.pyplot as plt
                        # fig, ax = plt.subplots()
                        # sample_ind = spikes["sample_index"][i]
                        # wf = traces[sample_ind - self.nbefore : sample_ind + self.nafter][:, chan_sparsity_mask]
                        # template = self.dense_templates_array[cluster_index, :, :][:, chan_sparsity_mask]
                        # ax.plot(wf.T.flatten())
                        # ax.plot(template.T.flatten())
                        # ax.set_title(f"amp{amp}")
                        # plt.show()
                else:
                    # not valid because already detected
                    spikes["cluster_index"][i] = -1

            else:
                spikes["cluster_index"][i] = -1
            
        

        # delta_sample = self.nbefore + self.nafter
        # # TODO benchmark this and make this faster
        # neighbors_spikes_inds = get_neighbors_spikes(spikes["sample_index"], spikes["channel_index"], delta_sample, self.near_chan_mask)
        # for i in range(spikes.size):
        #     amp = fit_one_amplitude_with_neighbors(spikes[i], spikes[neighbors_spikes_inds[i]],  traces, 
        #                                      self.sparsity_mask, self.templates.templates_array, self.nbefore, self.nafter)
        #     spikes["amplitude"][i] = amp

        keep = spikes["cluster_index"] >= 0
        spikes = spikes[keep]

        # keep = (spikes["amplitude"] >= 0.7) & (spikes["amplitude"] <= 1.4)
        # spikes = spikes[keep]

        # sparse_templates_array = self.templates.templates_array
        # wanted_channel_mask = np.ones(traces.shape[1], dtype=bool)
        # assert np.sum(wanted_channel_mask) == traces.shape[1] # TODO remove this DEBUG later
        # construct_prediction_sparse(spikes, traces, sparse_templates_array, self.sparsity_mask, wanted_channel_mask, self.nbefore, additive=False)


        return spikes



def get_most_probable_cluster(traces, sparse_templates_array, possible_clusters,
                              sample_index, chan_ind, nbefore_short, nafter_short, template_sparsity_mask):
    s0 = sample_index - nbefore_short
    s1 = sample_index + nafter_short
    wf_short = traces[s0:s1, :]

    ## numba with cluster+channel spasity
    union_channels = np.any(template_sparsity_mask[possible_clusters, :], axis=0)
    distances = numba_sparse_distance(wf_short,
                                      sparse_templates_array, template_sparsity_mask,
                                      union_channels, possible_clusters)

    ind = np.argmin(distances)
    cluster_index = possible_clusters[ind]

    return cluster_index


def get_neighbors_spikes(sample_inds, chan_inds, delta_sample, near_chan_mask):

    neighbors_spikes_inds = []
    for i in range(sample_inds.size):

        inds = np.flatnonzero(np.abs(sample_inds - sample_inds[i]) < delta_sample)
        neighb = []
        for ind in inds:
            if near_chan_mask[chan_inds[i], chan_inds[ind]] and i != ind:
                neighb.append(ind)
        neighbors_spikes_inds.append(neighb)

    return neighbors_spikes_inds


def fit_one_amplitude_with_neighbors(spike, neighbors_spikes,  traces, 
                                     template_sparsity_mask, sparse_templates_array,
                                     template_norms,
                                     nbefore, nafter):
    """
    Fit amplitude one spike of one spike with/without neighbors
    
    """


    import scipy.linalg

    cluster_index = spike["cluster_index"]
    sample_index = spike["sample_index"]
    chan_sparsity_mask = template_sparsity_mask[cluster_index, :]
    num_chans = np.sum(chan_sparsity_mask)
    if num_chans == 0:
        # protect against empty template because too sparse
        return 0.
    start, stop = sample_index - nbefore, sample_index + nafter
    if neighbors_spikes is None or (neighbors_spikes.size == 0):
        template = sparse_templates_array[cluster_index, :, :num_chans]
        wf = traces[start: stop, :][:, chan_sparsity_mask]
        # TODO precompute template norms
        amplitude = np.sum(template.flatten() * wf.flatten()) / template_norms[cluster_index]
    else:
        

        lim0 = min(start, np.min(neighbors_spikes["sample_index"]) - nbefore)
        lim1 = max(stop, np.max(neighbors_spikes["sample_index"]) + nafter)

        local_traces = traces[lim0:lim1, :][:, chan_sparsity_mask]
        mask_not_fitted = (neighbors_spikes["amplitude"] == 0.) & (neighbors_spikes["cluster_index"] >= 0)
        local_spike = spike.copy()
        local_spike["sample_index"] -= lim0
        local_spike["amplitude"] = 1.0

        local_neighbors_spikes = neighbors_spikes.copy()
        local_neighbors_spikes["sample_index"] -= lim0
        local_neighbors_spikes["amplitude"][:] = 1.0

        num_spikes_to_fit = 1 + np.sum(mask_not_fitted)
        x = np.zeros((lim1 - lim0, num_chans, num_spikes_to_fit), dtype="float32")
        wanted_channel_mask = chan_sparsity_mask
        construct_prediction_sparse(np.array([local_spike]), x[:, :, 0], sparse_templates_array,
                                    template_sparsity_mask, chan_sparsity_mask, nbefore, True)

        j = 1
        for i in range(neighbors_spikes.size):
            if mask_not_fitted[i]:
                # add to one regressor
                construct_prediction_sparse(local_neighbors_spikes[i:i+1], x[:, :, j], sparse_templates_array, template_sparsity_mask, chan_sparsity_mask, nbefore, True)
                j += 1
            elif local_neighbors_spikes[neighbors_spikes[i]]["sample_index"] >= 0:
                # remove from traces
                construct_prediction_sparse(local_neighbors_spikes[i:i+1], local_traces, sparse_templates_array, template_sparsity_mask, chan_sparsity_mask, nbefore, False)
            # else:
            #     pass
        
        x = x.reshape(-1, num_spikes_to_fit)
        y = local_traces.flatten()
        
        res = scipy.linalg.lstsq(x, y, cond=None, lapack_driver="gelsd")
        amplitudes = res[0]
        amplitude = amplitudes[0]


        # import matplotlib.pyplot as plt
        # x_plot = x.reshape((lim1 - lim0, num_chans, num_spikes_to_fit)).swapaxes(0, 1).reshape(-1, num_spikes_to_fit)
        # pred = x @ amplitudes
        # pred_plot = pred.reshape(-1, num_chans).T.flatten()
        # y_plot = y.reshape(-1, num_chans).T.flatten()
        # fig, ax = plt.subplots()
        # ax.plot(x_plot, color='b')
        # print(x_plot.shape, y_plot.shape)
        # ax.plot(y_plot, color='g')
        # ax.plot(pred_plot , color='r')
        # ax.set_title(f"{amplitudes}")
        # # ax.set_title(f"{amplitudes} {amp_dot}")
        # plt.show()

    return amplitude



if HAVE_NUMBA:
    @jit(nopython=True)
    def construct_prediction_sparse(spikes, traces, sparse_templates_array, template_sparsity_mask, wanted_channel_mask, nbefore, additive):
        #  must have np.sum(wanted_channel_mask) == traces.shape[0]
        total_chans = wanted_channel_mask.shape[0]
        for spike in spikes:
            ind0 = spike["sample_index"] - nbefore
            ind1 = ind0 + sparse_templates_array.shape[1]
            cluster_index = spike["cluster_index"]
            amplitude = spike["amplitude"]
            chan_in_template = 0
            chan_in_trace = 0
            for chan in range(total_chans):
                if wanted_channel_mask[chan]:
                    if template_sparsity_mask[cluster_index, chan]:
                        if additive:
                            traces[ind0:ind1, chan_in_trace] += sparse_templates_array[cluster_index, :, chan_in_template] * amplitude
                        else:
                            traces[ind0:ind1, chan_in_trace] -= sparse_templates_array[cluster_index, :, chan_in_template] * amplitude
                        chan_in_template += 1
                    chan_in_trace += 1
                else:
                    if template_sparsity_mask[cluster_index, chan]:
                        chan_in_template += 1


    @jit(nopython=True)
    def numba_sparse_distance(wf, sparse_templates_array, template_sparsity_mask, wanted_channel_mask, possible_clusters):
        """
        numba implementation that compute distance from template with sparsity

        wf is dense
        sparse_templates_array is sparse with the template_sparsity_mask
        """
        width, total_chans = wf.shape
        num_cluster = possible_clusters.shape[0]
        distances = np.zeros((num_cluster,), dtype=np.float32)
        for i in prange(num_cluster):
            cluster_index = possible_clusters[i]
            sum_dist = 0.0
            chan_in_template = 0
            for chan in range(total_chans):
                if wanted_channel_mask[chan]:
                    if template_sparsity_mask[cluster_index, chan]:
                        for s in range(width):
                            v = wf[s, chan]
                            t = sparse_templates_array[cluster_index, s, chan_in_template]
                            sum_dist += (v - t) ** 2
                        chan_in_template += 1
                    else:
                        for s in range(width):
                            v = wf[s, chan]
                            t = 0
                            sum_dist += (v - t) ** 2
                else:
                    if template_sparsity_mask[cluster_index, chan]:
                        chan_in_template += 1
            distances[i] = sum_dist
        return distances


    @jit(nopython=True)
    def numba_best_shift_sparse(traces, sparse_template, 
                         sample_index, nbefore, possible_shifts, distances_shift, chan_sparsity):
        """
        numba implementation to compute several sample shift before template substraction
        """
        width = sparse_template.shape[0]
        total_chans = traces.shape[1]
        n_shift = possible_shifts.size
        for i in range(n_shift):
            shift = possible_shifts[i]
            sum_dist = 0.0
            chan_in_template = 0
            for chan in range(total_chans):
                if chan_sparsity[chan]:
                    for s in range(width):
                        v = traces[sample_index - nbefore + s + shift, chan]
                        t = sparse_template[s, chan_in_template]
                        sum_dist += (v - t) ** 2
                    chan_in_template += 1
            distances_shift[i] = sum_dist

        return distances_shift


