"""Sorting components: template matching."""

import numpy as np
from spikeinterface.core import WaveformExtractor
from spikeinterface.core import (
    get_noise_levels,
    get_channel_distances,
    get_chunk_with_margin,
    get_random_data_chunks,
    compute_sparsity,
)
from spikeinterface.postprocessing import get_template_channel_sparsity, get_template_extremum_channel

from spikeinterface.sortingcomponents.peak_detection import DetectPeakLocallyExclusive

spike_dtype = [
    ("sample_index", "int64"),
    ("channel_index", "int64"),
    ("cluster_index", "int64"),
    ("amplitude", "float64"),
    ("segment_index", "int64"),
]


from .main import BaseTemplateMatchingEngine


class NaiveMatching(BaseTemplateMatchingEngine):
    """
    This is a naive template matching that does not resolve collision
    and does not take in account sparsity.
    It just minimizes the distance to templates for detected peaks.

    It is implemented for benchmarking against this low quality template matching.
    And also as an example how to deal with methods_kwargs, margin, intit, func, ...
    """

    default_params = {
        "peak_sign": "neg",
        "exclude_sweep_ms": 0.1,
        "detect_threshold": 5,
        "noise_levels": None,
        "local_radius_um": 75,
        "random_chunk_kwargs": {},
    }

    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):
        d = cls.default_params.copy()
        d.update(kwargs)

        if d["noise_levels"] is None:
            d["noise_levels"] = get_noise_levels(recording, **d["random_chunk_kwargs"])

        d["abs_thresholds"] = d["noise_levels"] * d["detect_threshold"]

        channel_distance = get_channel_distances(recording)
        d["neighbours_mask"] = channel_distance < d["local_radius_um"]
        d["exclude_sweep_size"] = int(d["exclude_sweep_ms"] * d["sampling_frequency"] / 1000.0)

        return d

    @classmethod
    def get_margin(cls, recording, kwargs):
        templates = kwargs['templates']
        margin = max(templates.nbefore, templates.nafter)
        return margin

    @classmethod
    def unserialize_in_worker(cls, kwargs):
        return kwargs

    @classmethod
    def main_function(cls, traces, method_kwargs):
        peak_sign = method_kwargs["peak_sign"]
        abs_thresholds = method_kwargs["abs_thresholds"]
        exclude_sweep_size = method_kwargs["exclude_sweep_size"]
        neighbours_mask = method_kwargs["neighbours_mask"]
        templates = method_kwargs["templates"]
        sparsity_mask = templates.sparsity_mask
        nbefore = templates.nbefore
        nafter = templates.nafter
        margin = method_kwargs["margin"]

        if margin > 0:
            peak_traces = traces[margin:-margin, :]
        else:
            peak_traces = traces
        peak_sample_ind, peak_chan_ind = DetectPeakLocallyExclusive.detect_peaks(
            peak_traces, peak_sign, abs_thresholds, exclude_sweep_size, neighbours_mask
        )
        peak_sample_ind += margin

        spikes = np.zeros(peak_sample_ind.size, dtype=spike_dtype)
        spikes["sample_index"] = peak_sample_ind
        spikes["channel_index"] = peak_chan_ind  # TODO need to put the channel from template

        # naively take the closest template
        for i in range(peak_sample_ind.size):
            i0 = peak_sample_ind[i] - nbefore
            i1 = peak_sample_ind[i] + nafter

            waveforms = traces[i0:i1, :]
            dist = np.sum(np.sum((templates.data - waveforms[None, :, :]) ** 2, axis=1), axis=1)
            cluster_index = np.argmin(dist)

            spikes["cluster_index"][i] = cluster_index
            spikes["amplitude"][i] = 0.0

        return spikes
