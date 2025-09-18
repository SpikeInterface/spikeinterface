"""Sorting components: peak detection."""

from __future__ import annotations

import numpy as np


from spikeinterface.core.node_pipeline import (
    PeakDetector,
    base_peak_dtype,
)

from spikeinterface.core.recording_tools import get_channel_distances, get_random_data_chunks
from spikeinterface.postprocessing.localization_tools import get_convolution_weights

import importlib.util

numba_spec = importlib.util.find_spec("numba")
if numba_spec is not None:
    HAVE_NUMBA = True
else:
    HAVE_NUMBA = False

from .by_channel import ByChannelPeakDetector


class MatchedFilteringPeakDetector(PeakDetector):
    """Detect peaks using the 'matched_filtering' method."""

    name = "matched_filtering"
    engine = "numba"
    need_noise_levels = False
    preferred_mp_context = None
    params_doc = (
        ByChannelPeakDetector.params_doc
        + """
    radius_um : float
        The radius to use to select neighbour channels for locally exclusive detection.
    prototype : array
        The canonical waveform of action potentials
    ms_before : float
        The time in ms before the maximial value of the absolute prototype
    weight_method : dict
        Parameter that should be provided to the get_convolution_weights() function
        in order to know how to estimate the positions. One argument is mode that could
        be either gaussian_2d (KS like) or exponential_3d (default)
    """
    )

    def __init__(
        self,
        recording,
        prototype,
        ms_before,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        radius_um=50,
        random_chunk_kwargs={"num_chunks_per_segment": 5},
        weight_method={},
        return_output=True,
    ):
        PeakDetector.__init__(self, recording, return_output=return_output)
        from scipy.sparse import csr_matrix

        if not HAVE_NUMBA:
            raise ModuleNotFoundError('matched_filtering" needs numba which is not installed')

        self.exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.0)
        channel_distance = get_channel_distances(recording)
        self.neighbours_mask = channel_distance <= radius_um

        self.conv_margin = prototype.shape[0]

        assert peak_sign in ("both", "neg", "pos")
        self.nbefore = int(ms_before * recording.sampling_frequency / 1000)
        if peak_sign == "neg":
            assert prototype[self.nbefore] < 0, "Prototype should have a negative peak"
            peak_sign = "pos"
        elif peak_sign == "pos":
            assert prototype[self.nbefore] > 0, "Prototype should have a positive peak"

        self.peak_sign = peak_sign
        self.prototype = np.flip(prototype) / np.linalg.norm(prototype)

        contact_locations = recording.get_channel_locations()
        dist = np.linalg.norm(contact_locations[:, np.newaxis] - contact_locations[np.newaxis, :], axis=2)
        self.weights, self.z_factors = get_convolution_weights(dist, **weight_method)
        self.num_z_factors = len(self.z_factors)
        self.num_channels = recording.get_num_channels()
        self.num_templates = self.num_channels
        if peak_sign == "both":
            self.weights = np.hstack((self.weights, self.weights))
            self.weights[:, self.num_templates :, :] *= -1
            self.num_templates *= 2

        self.weights = self.weights.reshape(self.num_templates * self.num_z_factors, -1)
        self.weights = csr_matrix(self.weights)
        random_data = get_random_data_chunks(recording, return_in_uV=False, **random_chunk_kwargs)
        conv_random_data = self.get_convolved_traces(random_data)
        medians = np.median(conv_random_data, axis=1)
        self.medians = medians[:, None]
        noise_levels = np.median(np.abs(conv_random_data - self.medians), axis=1) / 0.6744897501960817
        self.abs_thresholds = noise_levels * detect_threshold
        self.detect_threshold = detect_threshold
        self._dtype = np.dtype(base_peak_dtype + [("z", "float32")])

    def get_dtype(self):
        return self._dtype

    def get_trace_margin(self):
        return self.exclude_sweep_size + self.conv_margin

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):

        assert HAVE_NUMBA, "You need to install numba"
        conv_traces = self.get_convolved_traces(traces)
        # conv_traces -= self.medians
        conv_traces /= self.abs_thresholds[:, None]
        conv_traces = conv_traces[:, self.conv_margin : -self.conv_margin]
        traces_center = conv_traces[:, self.exclude_sweep_size : -self.exclude_sweep_size]

        traces_center = traces_center.reshape(self.num_z_factors, self.num_templates, traces_center.shape[1])
        conv_traces = conv_traces.reshape(self.num_z_factors, self.num_templates, conv_traces.shape[1])
        peak_mask = traces_center > 1

        peak_mask = _numba_detect_peak_matched_filtering(
            conv_traces,
            traces_center,
            peak_mask,
            self.exclude_sweep_size,
            self.abs_thresholds,
            self.peak_sign,
            self.neighbours_mask,
            self.num_channels,
        )

        # Find peaks and correct for time shift
        z_ind, peak_chan_ind, peak_sample_ind = np.nonzero(peak_mask)
        if self.peak_sign == "both":
            peak_chan_ind = peak_chan_ind % self.num_channels

        # If we want to estimate z
        # peak_chan_ind = peak_chan_ind % num_channels
        # z = np.zeros(len(peak_sample_ind), dtype=np.float32)
        # for count in range(len(peak_chan_ind)):
        #     channel = peak_chan_ind[count]
        #     peak = peak_sample_ind[count]
        #     data = traces[channel::num_channels, peak]
        #     z[count] = np.dot(data, z_factors)/data.sum()

        if peak_sample_ind.size == 0 or peak_chan_ind.size == 0:
            return (np.zeros(0, dtype=self._dtype),)

        peak_sample_ind += self.exclude_sweep_size + self.conv_margin + self.nbefore
        peak_amplitude = traces[peak_sample_ind, peak_chan_ind]

        local_peaks = np.zeros(peak_sample_ind.size, dtype=self._dtype)
        local_peaks["sample_index"] = peak_sample_ind
        local_peaks["channel_index"] = peak_chan_ind
        local_peaks["amplitude"] = peak_amplitude
        local_peaks["segment_index"] = segment_index
        local_peaks["z"] = z_ind

        # return is always a tuple
        return (local_peaks,)

    def get_convolved_traces(self, traces):
        from scipy.signal import oaconvolve

        tmp = oaconvolve(self.prototype[None, :], traces.T, axes=1, mode="valid")
        scalar_products = self.weights.dot(tmp)
        return scalar_products


if HAVE_NUMBA:
    import numba

    @numba.jit(nopython=True, parallel=False)
    def _numba_detect_peak_matched_filtering(
        traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask, num_channels
    ):
        num_z = traces_center.shape[0]
        num_templates = traces_center.shape[1]
        for template_ind in range(num_templates):
            for z in range(num_z):
                for s in range(peak_mask.shape[2]):
                    if not peak_mask[z, template_ind, s]:
                        continue
                    for neighbour in range(num_templates):
                        for j in range(num_z):
                            if not neighbours_mask[template_ind % num_channels, neighbour % num_channels]:
                                continue
                            for i in range(exclude_sweep_size):
                                if template_ind >= neighbour and z >= j:
                                    peak_mask[z, template_ind, s] &= (
                                        traces_center[z, template_ind, s] >= traces_center[j, neighbour, s]
                                    )
                                else:
                                    peak_mask[z, template_ind, s] &= (
                                        traces_center[z, template_ind, s] > traces_center[j, neighbour, s]
                                    )
                                peak_mask[z, template_ind, s] &= (
                                    traces_center[z, template_ind, s] > traces[j, neighbour, s + i]
                                )
                                peak_mask[z, template_ind, s] &= (
                                    traces_center[z, template_ind, s]
                                    >= traces[j, neighbour, exclude_sweep_size + s + i + 1]
                                )
                                if not peak_mask[z, template_ind, s]:
                                    break
                            if not peak_mask[z, template_ind, s]:
                                break
                        if not peak_mask[z, template_ind, s]:
                            break

        return peak_mask
