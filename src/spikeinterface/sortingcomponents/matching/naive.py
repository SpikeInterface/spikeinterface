"""Sorting components: template matching."""

from __future__ import annotations


import numpy as np
from spikeinterface.core import get_noise_levels, get_channel_distances
from spikeinterface.sortingcomponents.peak_detection import DetectPeakLocallyExclusive


from .base import BaseTemplateMatching, _base_matching_dtype


class NaiveMatching(BaseTemplateMatching):
    def __init__(
        self,
        recording,
        return_output=True,
        parents=None,
        templates=None,
        peak_sign="neg",
        exclude_sweep_ms=0.1,
        detect_threshold=5,
        noise_levels=None,
        radius_um=100.0,
        random_chunk_kwargs={},
    ):

        BaseTemplateMatching.__init__(self, recording, templates, return_output=True, parents=None)

        self.templates_array = self.templates.get_dense_templates()

        if noise_levels is None:
            noise_levels = get_noise_levels(recording, **random_chunk_kwargs, return_scaled=False)
        self.abs_threholds = noise_levels * detect_threshold
        self.peak_sign = peak_sign
        channel_distance = get_channel_distances(recording)
        self.neighbours_mask = channel_distance < radius_um
        self.exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.0)
        self.nbefore = self.templates.nbefore
        self.nafter = self.templates.nafter
        self.margin = max(self.nbefore, self.nafter)

    def get_trace_margin(self):
        return self.margin

    def compute_matching(self, traces, start_frame, end_frame, segment_index):

        if self.margin > 0:
            peak_traces = traces[self.margin : -self.margin, :]
        else:
            peak_traces = traces
        peak_sample_ind, peak_chan_ind = DetectPeakLocallyExclusive.detect_peaks(
            peak_traces, self.peak_sign, self.abs_threholds, self.exclude_sweep_size, self.neighbours_mask
        )
        peak_sample_ind += self.margin

        spikes = np.zeros(peak_sample_ind.size, dtype=_base_matching_dtype)
        spikes["sample_index"] = peak_sample_ind
        spikes["channel_index"] = peak_chan_ind

        # naively take the closest template
        for i in range(peak_sample_ind.size):
            i0 = peak_sample_ind[i] - self.nbefore
            i1 = peak_sample_ind[i] + self.nafter

            waveforms = traces[i0:i1, :]
            dist = np.sum(np.sum((self.templates_array - waveforms[None, :, :]) ** 2, axis=1), axis=1)
            cluster_index = np.argmin(dist)

            spikes["cluster_index"][i] = cluster_index
            spikes["amplitude"][i] = 0.0

        return spikes
