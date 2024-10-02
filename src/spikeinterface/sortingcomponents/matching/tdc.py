from __future__ import annotations

import numpy as np
from spikeinterface.core import (
    get_noise_levels,
    get_channel_distances,
    compute_sparsity,
    get_template_extremum_channel,
)

from spikeinterface.sortingcomponents.peak_detection import DetectPeakLocallyExclusive
from spikeinterface.core.template import Templates

from .base import BaseTemplateMatching, _base_matching_dtype

# spike_dtype = [
#     ("sample_index", "int64"),
#     ("channel_index", "int64"),
#     ("cluster_index", "int64"),
#     ("amplitude", "float64"),
#     ("segment_index", "int64"),
# ]

# from .main import BaseTemplateMatchingEngine

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
    def __init__(self, recording, return_output=True, parents=None,
        templates=None,
        peak_sign="neg",
        peak_shift_ms=0.2,
        detect_threshold=5,
        noise_levels=None,
        radius_um=100.,
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
            closest_u = np.array(closest_u[: num_closest])

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





# class TridesclousPeeler(BaseTemplateMatchingEngine):
#     """
#     Template-matching ported from Tridesclous sorter.

#     The idea of this peeler is pretty simple.
#     1. Find peaks
#     2. order by best amplitues
#     3. find nearest template
#     4. remove it from traces.
#     5. in the residual find peaks again

#     This method is quite fast but don't give exelent results to resolve
#     spike collision when templates have high similarity.
#     """

#     default_params = {
#         "templates": None,
#         "peak_sign": "neg",
#         "peak_shift_ms": 0.2,
#         "detect_threshold": 5,
#         "noise_levels": None,
#         "radius_um": 100,
#         "num_closest": 5,
#         "sample_shift": 3,
#         "ms_before": 0.8,
#         "ms_after": 1.2,
#         "num_peeler_loop": 2,
#         "num_template_try": 1,
#     }

#     @classmethod
#     def initialize_and_check_kwargs(cls, recording, kwargs):
#         assert HAVE_NUMBA, "TridesclousPeeler needs numba to be installed"

#         d = cls.default_params.copy()
#         d.update(kwargs)

#         assert isinstance(d["templates"], Templates), (
#             f"The templates supplied is of type {type(d['templates'])} " f"and must be a Templates"
#         )

#         templates = d["templates"]
#         unit_ids = templates.unit_ids
#         channel_ids = templates.channel_ids

#         sr = templates.sampling_frequency

#         d["nbefore"] = templates.nbefore
#         d["nafter"] = templates.nafter
#         templates_array = templates.get_dense_templates()

#         nbefore_short = int(d["ms_before"] * sr / 1000.0)
#         nafter_short = int(d["ms_before"] * sr / 1000.0)
#         assert nbefore_short <= templates.nbefore
#         assert nafter_short <= templates.nafter
#         d["nbefore_short"] = nbefore_short
#         d["nafter_short"] = nafter_short
#         s0 = templates.nbefore - nbefore_short
#         s1 = -(templates.nafter - nafter_short)
#         if s1 == 0:
#             s1 = None
#         templates_short = templates_array[:, slice(s0, s1), :].copy()
#         d["templates_short"] = templates_short

#         d["peak_shift"] = int(d["peak_shift_ms"] / 1000 * sr)

#         if d["noise_levels"] is None:
#             print("TridesclousPeeler : noise should be computed outside")
#             d["noise_levels"] = get_noise_levels(recording)

#         d["abs_thresholds"] = d["noise_levels"] * d["detect_threshold"]

#         channel_distance = get_channel_distances(recording)
#         d["neighbours_mask"] = channel_distance < d["radius_um"]

#         sparsity = compute_sparsity(
#             templates, method="best_channels"
#         )  # , peak_sign=d["peak_sign"], threshold=d["detect_threshold"])
#         template_sparsity_inds = sparsity.unit_id_to_channel_indices
#         template_sparsity = np.zeros((unit_ids.size, channel_ids.size), dtype="bool")
#         for unit_index, unit_id in enumerate(unit_ids):
#             chan_inds = template_sparsity_inds[unit_id]
#             template_sparsity[unit_index, chan_inds] = True

#         d["template_sparsity"] = template_sparsity

#         extremum_channel = get_template_extremum_channel(templates, peak_sign=d["peak_sign"], outputs="index")
#         # as numpy vector
#         extremum_channel = np.array([extremum_channel[unit_id] for unit_id in unit_ids], dtype="int64")
#         d["extremum_channel"] = extremum_channel

#         channel_locations = templates.probe.contact_positions

#         # TODO try it with real locaion
#         unit_locations = channel_locations[extremum_channel]
#         # ~ print(unit_locations)

#         # distance between units
#         import scipy

#         unit_distances = scipy.spatial.distance.cdist(unit_locations, unit_locations, metric="euclidean")

#         # seach for closet units and unitary discriminant vector
#         closest_units = []
#         for unit_ind, unit_id in enumerate(unit_ids):
#             order = np.argsort(unit_distances[unit_ind, :])
#             closest_u = np.arange(unit_ids.size)[order].tolist()
#             closest_u.remove(unit_ind)
#             closest_u = np.array(closest_u[: d["num_closest"]])

#             # compute unitary discriminent vector
#             (chans,) = np.nonzero(d["template_sparsity"][unit_ind, :])
#             template_sparse = templates_array[unit_ind, :, :][:, chans]
#             closest_vec = []
#             # against N closets
#             for u in closest_u:
#                 vec = templates_array[u, :, :][:, chans] - template_sparse
#                 vec /= np.sum(vec**2)
#                 closest_vec.append((u, vec))
#             # against noise
#             closest_vec.append((None, -template_sparse / np.sum(template_sparse**2)))

#             closest_units.append(closest_vec)

#         d["closest_units"] = closest_units

#         # distance channel from unit
#         import scipy

#         distances = scipy.spatial.distance.cdist(channel_locations, unit_locations, metric="euclidean")
#         near_cluster_mask = distances < d["radius_um"]

#         # nearby cluster for each channel
#         possible_clusters_by_channel = []
#         for channel_index in range(distances.shape[0]):
#             (cluster_inds,) = np.nonzero(near_cluster_mask[channel_index, :])
#             possible_clusters_by_channel.append(cluster_inds)

#         d["possible_clusters_by_channel"] = possible_clusters_by_channel
#         d["possible_shifts"] = np.arange(-d["sample_shift"], d["sample_shift"] + 1, dtype="int64")

#         return d

#     @classmethod
#     def serialize_method_kwargs(cls, kwargs):
#         kwargs = dict(kwargs)
#         return kwargs

#     @classmethod
#     def unserialize_in_worker(cls, kwargs):
#         return kwargs

#     @classmethod
#     def get_margin(cls, recording, kwargs):
#         margin = 2 * (kwargs["nbefore"] + kwargs["nafter"])
#         return margin

#     @classmethod
#     def main_function(cls, traces, d):
#         traces = traces.copy()

#         all_spikes = []
#         level = 0
#         while True:
#             spikes = _tdc_find_spikes(traces, d, level=level)
#             keep = spikes["cluster_index"] >= 0

#             if not np.any(keep):
#                 break
#             all_spikes.append(spikes[keep])

#             level += 1

#             if level == d["num_peeler_loop"]:
#                 break

#         if len(all_spikes) > 0:
#             all_spikes = np.concatenate(all_spikes)
#             order = np.argsort(all_spikes["sample_index"])
#             all_spikes = all_spikes[order]
#         else:
#             all_spikes = np.zeros(0, dtype=spike_dtype)

#         return all_spikes


# def _tdc_find_spikes(traces, d, level=0):
#     peak_sign = d["peak_sign"]
#     templates = d["templates"]
#     templates_short = d["templates_short"]
#     templates_array = templates.get_dense_templates()

#     margin = d["margin"]
#     possible_clusters_by_channel = d["possible_clusters_by_channel"]

#     peak_traces = traces[margin // 2 : -margin // 2, :]
#     peak_sample_ind, peak_chan_ind = DetectPeakLocallyExclusive.detect_peaks(
#         peak_traces, peak_sign, d["abs_thresholds"], d["peak_shift"], d["neighbours_mask"]
#     )
#     peak_sample_ind += margin // 2

#     peak_amplitude = traces[peak_sample_ind, peak_chan_ind]
#     order = np.argsort(np.abs(peak_amplitude))[::-1]
#     peak_sample_ind = peak_sample_ind[order]
#     peak_chan_ind = peak_chan_ind[order]

#     spikes = np.zeros(peak_sample_ind.size, dtype=spike_dtype)
#     spikes["sample_index"] = peak_sample_ind
#     spikes["channel_index"] = peak_chan_ind  # TODO need to put the channel from template

#     possible_shifts = d["possible_shifts"]
#     distances_shift = np.zeros(possible_shifts.size)

#     for i in range(peak_sample_ind.size):
#         sample_index = peak_sample_ind[i]

#         chan_ind = peak_chan_ind[i]
#         possible_clusters = possible_clusters_by_channel[chan_ind]

#         if possible_clusters.size > 0:
#             # ~ s0 = sample_index - d['nbefore']
#             # ~ s1 = sample_index + d['nafter']

#             # ~ wf = traces[s0:s1, :]

#             s0 = sample_index - d["nbefore_short"]
#             s1 = sample_index + d["nafter_short"]
#             wf_short = traces[s0:s1, :]

#             ## pure numpy with cluster spasity
#             # distances = np.sum(np.sum((templates[possible_clusters, :, :] - wf[None, : , :])**2, axis=1), axis=1)

#             ## pure numpy with cluster+channel spasity
#             # union_channels, = np.nonzero(np.any(d['template_sparsity'][possible_clusters, :], axis=0))
#             # distances = np.sum(np.sum((templates[possible_clusters][:, :, union_channels] - wf[: , union_channels][None, : :])**2, axis=1), axis=1)

#             ## numba with cluster+channel spasity
#             union_channels = np.any(d["template_sparsity"][possible_clusters, :], axis=0)
#             # distances = numba_sparse_dist(wf, templates, union_channels, possible_clusters)
#             distances = numba_sparse_dist(wf_short, templates_short, union_channels, possible_clusters)

#             # DEBUG
#             # ~ ind = np.argmin(distances)
#             # ~ cluster_index = possible_clusters[ind]

#             for ind in np.argsort(distances)[: d["num_template_try"]]:
#                 cluster_index = possible_clusters[ind]

#                 chan_sparsity = d["template_sparsity"][cluster_index, :]
#                 template_sparse = templates_array[cluster_index, :, :][:, chan_sparsity]

#                 # find best shift

#                 ## pure numpy version
#                 # for s, shift in enumerate(possible_shifts):
#                 #     wf_shift = traces[s0 + shift: s1 + shift, chan_sparsity]
#                 #     distances_shift[s] = np.sum((template_sparse - wf_shift)**2)
#                 # ind_shift = np.argmin(distances_shift)
#                 # shift = possible_shifts[ind_shift]

#                 ## numba version
#                 numba_best_shift(
#                     traces,
#                     templates_array[cluster_index, :, :],
#                     sample_index,
#                     d["nbefore"],
#                     possible_shifts,
#                     distances_shift,
#                     chan_sparsity,
#                 )
#                 ind_shift = np.argmin(distances_shift)
#                 shift = possible_shifts[ind_shift]

#                 sample_index = sample_index + shift
#                 s0 = sample_index - d["nbefore"]
#                 s1 = sample_index + d["nafter"]
#                 wf_sparse = traces[s0:s1, chan_sparsity]

#                 # accept or not

#                 centered = wf_sparse - template_sparse
#                 accepted = True
#                 for other_ind, other_vector in d["closest_units"][cluster_index]:
#                     v = np.sum(centered * other_vector)
#                     if np.abs(v) > 0.5:
#                         accepted = False
#                         break

#                 if accepted:
#                     # ~ if ind != np.argsort(distances)[0]:
#                     # ~ print('not first one', np.argsort(distances), ind)
#                     break

#             if accepted:
#                 amplitude = 1.0

#                 # remove template
#                 template = templates_array[cluster_index, :, :]
#                 s0 = sample_index - d["nbefore"]
#                 s1 = sample_index + d["nafter"]
#                 traces[s0:s1, :] -= template * amplitude

#             else:
#                 cluster_index = -1
#                 amplitude = 0.0

#         else:
#             cluster_index = -1
#             amplitude = 0.0

#         spikes["cluster_index"][i] = cluster_index
#         spikes["amplitude"][i] = amplitude

#     return spikes


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
        peak_shift_ms=0.2,
        detect_threshold=5,
        noise_levels=None,
        radius_um=100.,
        num_closest=5,
        sample_shift=3,
        ms_before=0.8,
        ms_after=1.2,
        num_peeler_loop=2,
        num_template_try=1,
        ):

        BaseTemplateMatching.__init__(self, recording, templates, return_output=True, parents=None)

        # self.dense_templates_array = templates.get_dense_templates()

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
        self.dense_templates_array = templates.get_dense_templates()
        self.dense_templates_array_short = self.dense_templates_array[:, slice(s0, s1), :].copy()

        self.peak_shift = int(peak_shift_ms / 1000 * sr)

        assert noise_levels is not None, "TridesclousPeeler : noise should be computed outside"

        self.abs_thresholds = noise_levels * detect_threshold

        channel_distance = get_channel_distances(recording)
        self.neighbours_mask = channel_distance <= radius_um

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
        unit_distances = scipy.spatial.distance.cdist(unit_locations, unit_locations, metric="euclidean")

        # # seach for closet units and unitary discriminant vector
        # closest_units = []
        # for unit_ind, unit_id in enumerate(unit_ids):
        #     order = np.argsort(unit_distances[unit_ind, :])
        #     closest_u = np.arange(unit_ids.size)[order].tolist()
        #     closest_u.remove(unit_ind)
        #     closest_u = np.array(closest_u[: num_closest])

        #     # compute unitary discriminent vector
        #     (chans,) = np.nonzero(self.sparsity_mask[unit_ind, :])
        #     template_sparse = self.templates_array[unit_ind, :, :][:, chans]
        #     closest_vec = []
        #     # against N closets
        #     for u in closest_u:
        #         vec = self.templates_array[u, :, :][:, chans] - template_sparse
        #         vec /= np.sum(vec**2)
        #         closest_vec.append((u, vec))
        #     # against noise
        #     closest_vec.append((None, -template_sparse / np.sum(template_sparse**2)))

        #     closest_units.append(closest_vec)

        # self.closest_units = closest_units

        # distance channel from unit

        # nearby cluster for each channel
        distances = scipy.spatial.distance.cdist(channel_locations, unit_locations, metric="euclidean")
        near_cluster_mask = distances <= radius_um
        self.possible_clusters_by_channel = []
        for channel_index in range(distances.shape[0]):
            (cluster_inds,) = np.nonzero(near_cluster_mask[channel_index, :])
            self.possible_clusters_by_channel.append(cluster_inds)

        # 
        distances = scipy.spatial.distance.cdist(channel_locations, channel_locations, metric="euclidean")
        self.near_chan_mask = distances <= radius_um


        self.possible_shifts = np.arange(-sample_shift, sample_shift + 1, dtype="int64")

        self.num_peeler_loop = num_peeler_loop
        self.num_template_try = num_template_try

        self.margin = max(self.nbefore, self.nafter) * 2

    def get_trace_margin(self):
        return self.margin

    def compute_matching(self, traces, start_frame, end_frame, segment_index):
        
        # TODO check if this is usefull
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
        spikes["channel_index"] = peak_chan_ind

        # possible_shifts = self.possible_shifts
        distances_shift = np.zeros(self.possible_shifts.size)

        for i in range(peak_sample_ind.size):
            sample_index = peak_sample_ind[i]

            chan_ind = peak_chan_ind[i]
            possible_clusters = self.possible_clusters_by_channel[chan_ind]

            if possible_clusters.size > 0:
                s0 = sample_index - self.nbefore_short
                s1 = sample_index + self.nafter_short
                wf_short = traces[s0:s1, :]

                ## numba with cluster+channel spasity
                union_channels = np.any(self.sparsity_mask[possible_clusters, :], axis=0)
                distances = numba_sparse_dist(wf_short, self.dense_templates_array_short, union_channels, possible_clusters)

                ind = np.argmin(distances)
                cluster_index = possible_clusters[ind]
                
                chan_sparsity_mask = self.sparsity_mask[cluster_index, :]

                # find best shift
                numba_best_shift(
                    traces,
                    self.dense_templates_array_short[cluster_index, :, :],
                    sample_index,
                    self.nbefore_short,
                    self.possible_shifts,
                    distances_shift,
                    chan_sparsity_mask,
                )
                ind_shift = np.argmin(distances_shift)
                shift = self.possible_shifts[ind_shift]

                
                template_sparse = self.dense_templates_array[cluster_index, :, :][:, chan_sparsity_mask]

                spikes["sample_index"][i] += shift
                spikes["cluster_index"][i] = cluster_index
            else:
                spikes["cluster_index"][i] = -1
            
        
        keep = spikes["cluster_index"] >= 0
        spikes = spikes[keep]

        delta_sample = self.nbefore + self.nafter
        # TODO benchmark this
        # delta_sample = self.nbefore_short + self.nafter_short
        neighbors_spikes = get_neighbors_spikes(spikes["sample_index"], spikes["channel_index"], delta_sample, self.near_chan_mask)
        for i in range(spikes.size):
            if len(neighbors_spikes[i]) == 0:
                # TODO find someting better
                spikes["amplitude"][i] = 1.0
            else:
                local_inds = [i] + neighbors_spikes[i]
                local_spikes = spikes[local_inds]

                # TODO make the clip shorter
                start, stop = np.min(spikes["sample_index"]) - self.nbefore, np.max(spikes["sample_index"]) + self.nafter
                sparse_templates_array = self.templates.templates_array
                cluster_index = spikes["cluster_index"][i]
                chan_sparsity_mask = self.sparsity_mask[cluster_index, :]
                local_amplitudes = fit_sevral_amplitudes(local_spikes, traces, start, stop, self.sparsity_mask,
                                                         sparse_templates_array, self.nbefore, chan_sparsity_mask)
                amp0 = local_amplitudes[0]
                spikes["amplitude"][i] = amp0

        keep = (spikes["amplitude"] >= 0.7) & (spikes["amplitude"] <= 1.4)
        spikes = spikes[keep]

        sparse_templates_array = self.templates.templates_array
        construct_prediction_sparse(spikes, traces, sparse_templates_array, self.sparsity_mask, self.nbefore, additive=False)


        return spikes


def get_neighbors_spikes(sample_inds, chan_inds, delta_sample, near_chan_mask):

    neighbors_spikes = []
    for i in range(sample_inds.size):

        inds = np.flatnonzero(np.abs(sample_inds - sample_inds[i]) < delta_sample)
        neighb = []
        for ind in inds:
            if near_chan_mask[chan_inds[i], chan_inds[ind]] and i != ind:
                neighb.append(ind)
        neighbors_spikes.append(neighb)

    return neighbors_spikes

def fit_sevral_amplitudes(spikes, traces, start, stop, sparsity_mask, sparse_templates_array, nbefore, chan_sparsity_mask):
    import scipy.linalg

    # import time

    # t0 = time.perf_counter()
    local_traces = traces[start:stop, :][:, chan_sparsity_mask]

    local_spikes = spikes.copy()
    local_spikes["sample_index"] -= start
    local_spikes["amplitude"][:] = 1.0

    num_spikes = spikes.size
    local_chans = np.flatnonzero(chan_sparsity_mask)
    local_sparsity_mask = sparsity_mask[:, local_chans]
    x = np.zeros((stop - start, local_chans.size, num_spikes), dtype="float32")
    for i in range(num_spikes):
        construct_prediction_sparse(local_spikes[i:i+1], x[:, :, i], sparse_templates_array, local_sparsity_mask, nbefore, True)
    
    x = x.reshape(-1, num_spikes)
    y = local_traces.flatten()
    
    # t1 = time.perf_counter()
    res = scipy.linalg.lstsq(x, y, cond=None, lapack_driver="gelsd")
    amplitudes = res[0]
    # t2 = time.perf_counter()
    # print(t1-t0, t2-t1)

    # import matplotlib.pyplot as plt
    # num_chans = local_chans.size
    # x_plot = x.reshape((stop - start, num_chans, num_spikes)).swapaxes(0, 1).reshape(-1, num_spikes)
    # pred = x @ amplitudes
    # pred_plot = pred.reshape(-1, num_chans).T.flatten()
    # y_plot = y.reshape(-1, num_chans).T.flatten()
    # fig, ax = plt.subplots()
    # ax.plot(x_plot, color='b')
    # print(x_plot.shape, y_plot.shape)
    # ax.plot(y_plot, color='g')
    # ax.plot(pred_plot , color='r')
    # ax.set_title(f"{amplitudes}")
    # plt.show()

    return amplitudes


if HAVE_NUMBA:
    @jit(nopython=True)
    def construct_prediction_sparse(spikes, traces, sparse_templates_array, sparsity_mask, nbefore, additive):
        for spike in spikes:
            ind0 = spike["sample_index"] - nbefore
            ind1 = ind0 + sparse_templates_array.shape[1]
            unit_index = spike["cluster_index"]
            i = 0
            for chan in range(traces.shape[1]):
                if sparsity_mask[unit_index, chan]:
                    if additive:
                        traces[ind0:ind1, chan] += sparse_templates_array[spike["cluster_index"], :, i] * spike["amplitude"]
                    else:
                        traces[ind0:ind1, chan] -= sparse_templates_array[spike["cluster_index"], :, i] * spike["amplitude"]
                    i += 1