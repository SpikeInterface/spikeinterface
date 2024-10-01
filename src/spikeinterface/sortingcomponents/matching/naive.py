"""Sorting components: template matching."""

from __future__ import annotations


import numpy as np
from spikeinterface.core import get_noise_levels, get_channel_distances
from spikeinterface.sortingcomponents.peak_detection import DetectPeakLocallyExclusive
# from spikeinterface.core.template import Templates

# spike_dtype = [
#     ("sample_index", "int64"),
#     ("channel_index", "int64"),
#     ("cluster_index", "int64"),
#     ("amplitude", "float64"),
#     ("segment_index", "int64"),
# ]


from .base import BaseTemplateMatching, _base_matching_dtype

class NaiveMatching(BaseTemplateMatching):
    def __init__(self, recording, return_output=True, parents=None,
        templates=None,
        peak_sign="neg",
        exclude_sweep_ms=0.1,
        detect_threshold=5,
        noise_levels=None,
        radius_um=100.,
        random_chunk_kwargs={},
        ):

        BaseTemplateMatching.__init__(self, recording, templates, return_output=True, parents=None)

        # TODO put this in base ????
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
            peak_traces = traces[self.margin:-self.margin, :]
        else:
            peak_traces = traces
        peak_sample_ind, peak_chan_ind = DetectPeakLocallyExclusive.detect_peaks(
            peak_traces, self.peak_sign, self.abs_threholds, self.exclude_sweep_size, self.neighbours_mask
        )
        peak_sample_ind += self.margin

        spikes = np.zeros(peak_sample_ind.size, dtype=_base_matching_dtype)
        spikes["sample_index"] = peak_sample_ind
        spikes["channel_index"] = peak_chan_ind  # TODO need to put the channel from template

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
    
        
# from .main import BaseTemplateMatchingEngine

# class NaiveMatching(BaseTemplateMatchingEngine):
#     """
#     This is a naive template matching that does not resolve collision
#     and does not take in account sparsity.
#     It just minimizes the distance to templates for detected peaks.

#     It is implemented for benchmarking against this low quality template matching.
#     And also as an example how to deal with methods_kwargs, margin, intit, func, ...
#     """

#     default_params = {
#         "templates": None,
#         "peak_sign": "neg",
#         "exclude_sweep_ms": 0.1,
#         "detect_threshold": 5,
#         "noise_levels": None,
#         "radius_um": 100,
#         "random_chunk_kwargs": {},
#     }

#     @classmethod
#     def initialize_and_check_kwargs(cls, recording, kwargs):
#         d = cls.default_params.copy()
#         d.update(kwargs)

#         assert isinstance(d["templates"], Templates), (
#             f"The templates supplied is of type {type(d['templates'])} " f"and must be a Templates"
#         )

#         templates = d["templates"]

#         if d["noise_levels"] is None:
#             d["noise_levels"] = get_noise_levels(recording, **d["random_chunk_kwargs"], return_scaled=False)

#         d["abs_threholds"] = d["noise_levels"] * d["detect_threshold"]

#         channel_distance = get_channel_distances(recording)
#         d["neighbours_mask"] = channel_distance < d["radius_um"]

#         d["nbefore"] = templates.nbefore
#         d["nafter"] = templates.nafter

#         d["exclude_sweep_size"] = int(d["exclude_sweep_ms"] * recording.get_sampling_frequency() / 1000.0)

#         return d

#     @classmethod
#     def get_margin(cls, recording, kwargs):
#         margin = max(kwargs["nbefore"], kwargs["nafter"])
#         return margin

#     @classmethod
#     def serialize_method_kwargs(cls, kwargs):
#         kwargs = dict(kwargs)
#         return kwargs

#     @classmethod
#     def unserialize_in_worker(cls, kwargs):
#         return kwargs

#     @classmethod
#     def main_function(cls, traces, method_kwargs):
#         peak_sign = method_kwargs["peak_sign"]
#         abs_threholds = method_kwargs["abs_threholds"]
#         exclude_sweep_size = method_kwargs["exclude_sweep_size"]
#         neighbours_mask = method_kwargs["neighbours_mask"]
#         templates_array = method_kwargs["templates"].get_dense_templates()

#         nbefore = method_kwargs["nbefore"]
#         nafter = method_kwargs["nafter"]

#         margin = method_kwargs["margin"]

#         if margin > 0:
#             peak_traces = traces[margin:-margin, :]
#         else:
#             peak_traces = traces
#         peak_sample_ind, peak_chan_ind = DetectPeakLocallyExclusive.detect_peaks(
#             peak_traces, peak_sign, abs_threholds, exclude_sweep_size, neighbours_mask
#         )
#         peak_sample_ind += margin

#         spikes = np.zeros(peak_sample_ind.size, dtype=spike_dtype)
#         spikes["sample_index"] = peak_sample_ind
#         spikes["channel_index"] = peak_chan_ind  # TODO need to put the channel from template

#         # naively take the closest template
#         for i in range(peak_sample_ind.size):
#             i0 = peak_sample_ind[i] - nbefore
#             i1 = peak_sample_ind[i] + nafter

#             waveforms = traces[i0:i1, :]
#             dist = np.sum(np.sum((templates_array - waveforms[None, :, :]) ** 2, axis=1), axis=1)
#             cluster_index = np.argmin(dist)

#             spikes["cluster_index"][i] = cluster_index
#             spikes["amplitude"][i] = 0.0

#         return spikes
