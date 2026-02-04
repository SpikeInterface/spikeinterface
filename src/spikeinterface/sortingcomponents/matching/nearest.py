"""Sorting components: template matching."""

from __future__ import annotations


import numpy as np
from spikeinterface.core import get_noise_levels, get_channel_distances


from .base import BaseTemplateMatching, _base_matching_dtype


class NearestTemplatesPeeler(BaseTemplateMatching):

    name = "nearest"
    need_noise_levels = True
    # this is because numba
    need_first_call_before_pipeline = True

    params_doc = """
    peak_sign : 'neg' | 'pos' | 'both'
        The peak sign to use for detection
    exclude_sweep_ms : float
        The exclusion window (in ms) around a detected peak to exclude other peaks on neighboring channels
    detect_threshold : float
        The threshold for peak detection in term of k x MAD
    noise_levels : None | array
        If None the noise levels are estimated using random chunks of the recording. If array it should be an array of size (num_channels,) with the noise level of each channel
    radius_um : float
        The radius to define the neighborhood between channels in micrometers while detecting the peaks
    """

    def __init__(
        self,
        recording,
        templates,
        return_output=True,
        peak_sign="neg",
        exclude_sweep_ms=0.8,
        detect_threshold=5,
        noise_levels=None,
        detection_radius_um=100.0,
        neighborhood_radius_um=50.0,
        sparsity_radius_um=100.0,
    ):

        BaseTemplateMatching.__init__(self, recording, templates, return_output=return_output)

        self.noise_levels = noise_levels
        self.abs_threholds = self.noise_levels * detect_threshold
        self.peak_sign = peak_sign
        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance <= detection_radius_um

        num_templates = len(self.templates.unit_ids)
        num_channels = recording.get_num_channels()

        if neighborhood_radius_um is not None:
            from spikeinterface.core.template_tools import get_template_extremum_channel

            best_channels = get_template_extremum_channel(self.templates, peak_sign=self.peak_sign, outputs="index")
            best_channels = np.array([best_channels[i] for i in templates.unit_ids])
            channel_locations = recording.get_channel_locations()
            template_distances = np.linalg.norm(
                channel_locations[:, None] - channel_locations[best_channels][np.newaxis, :], axis=2
            )
            self.neighborhood_mask = template_distances <= neighborhood_radius_um
        else:
            self.neighborhood_mask = np.ones((num_channels, num_templates), dtype=bool)

        if sparsity_radius_um is not None:
            if not templates.are_templates_sparse():
                from spikeinterface.core.sparsity import compute_sparsity

                sparsity = compute_sparsity(
                    templates, method="radius", radius_um=sparsity_radius_um, peak_sign=self.peak_sign
                )
            else:
                sparsity = templates.sparsity

            self.sparsity_mask = np.zeros((num_channels, num_channels), dtype=bool)
            for channel_index in np.arange(num_channels):
                mask = self.neighborhood_mask[channel_index]
                self.sparsity_mask[channel_index] = np.sum(sparsity.mask[mask], axis=0) > 0
        else:
            self.sparsity_mask = np.ones((num_channels, num_channels), dtype=bool)

        self.templates_array = self.templates.get_dense_templates()
        self.exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.0)
        self.nbefore = self.templates.nbefore
        self.nafter = self.templates.nafter
        self.margin = max(self.nbefore, self.nafter)
        self.lookup_tables = {}
        self.lookup_tables["templates"] = {}
        self.lookup_tables["channels"] = {}
        for i in range(num_channels):
            self.lookup_tables["templates"][i] = np.flatnonzero(self.neighborhood_mask[i])
            self.lookup_tables["channels"][i] = np.flatnonzero(self.sparsity_mask[i])

    def get_trace_margin(self):
        return self.margin

    def compute_matching(self, traces, start_frame, end_frame, segment_index):
        from spikeinterface.sortingcomponents.peak_detection.locally_exclusive import (
            detect_peaks_numba_locally_exclusive_on_chunk,
        )
        from scipy.spatial.distance import cdist

        if self.margin > 0:
            peak_traces = traces[self.margin : -self.margin, :]
        else:
            peak_traces = traces
        peak_sample_ind, peak_chan_ind = detect_peaks_numba_locally_exclusive_on_chunk(
            peak_traces, self.peak_sign, self.abs_threholds, self.exclude_sweep_size, self.neighbours_mask
        )
        peak_sample_ind += self.margin

        spikes = np.empty(peak_sample_ind.size, dtype=_base_matching_dtype)
        spikes["sample_index"] = peak_sample_ind
        spikes["channel_index"] = peak_chan_ind
        spikes["amplitude"] = 1.0

        # naively take the closest template
        for main_chan in np.unique(spikes["channel_index"]):
            (idx,) = np.nonzero(spikes["channel_index"] == main_chan)

            unit_inds = self.lookup_tables["templates"][main_chan]
            templates = self.templates_array[unit_inds]
            num_templates = templates.shape[0]
            if num_templates > 0:
                waveforms = traces[spikes["sample_index"][idx][:, None] + np.arange(-self.nbefore, self.nafter)]
                chan_inds = self.lookup_tables["channels"][main_chan]
                XA = templates[:, :, chan_inds].reshape(num_templates, -1)
                XB = waveforms[:, :, chan_inds].reshape(len(idx), -1)

                dist = cdist(XA, XB, "euclidean")
                cluster_index = np.argmin(dist, 0)
                spikes["cluster_index"][idx] = unit_inds[cluster_index]
            else:
                spikes["cluster_index"][idx] = -1  # no template for this channel

        return spikes


class NearestTemplatesSVDPeeler(NearestTemplatesPeeler):

    name = "nearest-svd"
    need_noise_levels = True
    params_doc = NearestTemplatesPeeler.params_doc + """
    svd_model : The svd model used to project the waveforms
        The radius to use to select neighbour channels for locally exclusive detection.
    svd_radius_um : float
        The radius in um of the local neighboorhood used, centered on every detected peaks, to compute
        the distances with all the templates in the SVD space
    """

    def __init__(
        self,
        recording,
        templates,
        svd_model,
        return_output=True,
        peak_sign="neg",
        exclude_sweep_ms=0.8,
        detect_threshold=5,
        noise_levels=None,
        detection_radius_um=100.0,
        neighborhood_radius_um=50.0,
        sparsity_radius_um=100.0,
    ):

        NearestTemplatesPeeler.__init__(
            self,
            recording,
            templates,
            return_output=return_output,
            peak_sign=peak_sign,
            exclude_sweep_ms=exclude_sweep_ms,
            detect_threshold=detect_threshold,
            noise_levels=noise_levels,
            detection_radius_um=detection_radius_um,
            neighborhood_radius_um=neighborhood_radius_um,
            sparsity_radius_um=sparsity_radius_um,
        )

        from spikeinterface.sortingcomponents.waveforms.waveform_utils import (
            to_temporal_representation,
            from_temporal_representation,
        )

        self.num_channels = self.recording.get_num_channels()
        self.svd_model = svd_model
        temporal_templates = to_temporal_representation(self.templates_array)
        projected_temporal_templates = self.svd_model.transform(temporal_templates)
        self.svd_templates = from_temporal_representation(projected_temporal_templates, self.num_channels)

    def get_trace_margin(self):
        return self.margin

    def compute_matching(self, traces, start_frame, end_frame, segment_index):
        from spikeinterface.sortingcomponents.peak_detection.locally_exclusive import (
            detect_peaks_numba_locally_exclusive_on_chunk,
        )

        from scipy.spatial.distance import cdist
        from spikeinterface.sortingcomponents.waveforms.waveform_utils import (
            to_temporal_representation,
            from_temporal_representation,
        )

        if self.margin > 0:
            peak_traces = traces[self.margin : -self.margin, :]
        else:
            peak_traces = traces
        peak_sample_ind, peak_chan_ind = detect_peaks_numba_locally_exclusive_on_chunk(
            peak_traces, self.peak_sign, self.abs_threholds, self.exclude_sweep_size, self.neighbours_mask
        )
        peak_sample_ind += self.margin

        spikes = np.empty(peak_sample_ind.size, dtype=_base_matching_dtype)
        spikes["sample_index"] = peak_sample_ind
        spikes["channel_index"] = peak_chan_ind
        spikes["amplitude"] = 1.0

        # naively take the closest template
        for main_chan in np.unique(spikes["channel_index"]):
            (idx,) = np.nonzero(spikes["channel_index"] == main_chan)

            unit_inds = self.lookup_tables["templates"][main_chan]
            templates = self.svd_templates[unit_inds]
            num_templates = templates.shape[0]

            if num_templates > 0:
                chan_inds = self.lookup_tables["channels"][main_chan]
                waveforms = traces[spikes["sample_index"][idx][:, None] + np.arange(-self.nbefore, self.nafter)]
                temporal_waveforms = to_temporal_representation(waveforms)
                projected_temporal_waveforms = self.svd_model.transform(temporal_waveforms)
                projected_waveforms = from_temporal_representation(projected_temporal_waveforms, self.num_channels)

                XA = templates[:, :, chan_inds].reshape(num_templates, -1)
                XB = projected_waveforms[:, :, chan_inds].reshape(len(idx), -1)

                dist = cdist(XA, XB, "euclidean")
                cluster_index = np.argmin(dist, 0)
                spikes["cluster_index"][idx] = unit_inds[cluster_index]
            else:
                spikes["cluster_index"][idx] = -1  # no template for this channel

        return spikes
