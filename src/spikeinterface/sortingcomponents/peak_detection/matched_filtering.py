"""Sorting components: peak detection."""

from __future__ import annotations

import importlib.util
import numpy as np

from spikeinterface.core.base import base_peak_dtype
from spikeinterface.core.node_pipeline import PeakDetector
from spikeinterface.core.recording_tools import get_channel_distances, get_random_data_chunks
from spikeinterface.postprocessing.localization_tools import get_convolution_weights

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
    params_doc = ByChannelPeakDetector.params_doc + """
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

    def __init__(
        self,
        recording,
        prototype,
        ms_before,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=1.0,
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
        self.abs_thresholds = (noise_levels * detect_threshold).reshape(self.num_z_factors, self.num_templates)
        self.detect_threshold = detect_threshold
        self._dtype = np.dtype(base_peak_dtype + [("z", "float32")])

    def get_dtype(self):
        return self._dtype

    def get_trace_margin(self):
        return self.exclude_sweep_size + self.conv_margin + 1

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):

        assert HAVE_NUMBA, "You need to install numba"
        conv_traces = self.get_convolved_traces(traces)
        conv_traces = conv_traces[:, self.conv_margin : -self.conv_margin]
        conv_traces = conv_traces.reshape(self.num_z_factors, self.num_templates, conv_traces.shape[1])

        z_inds, template_inds, samples_inds = _numba_detect_peak_matched_filtering(
            conv_traces,
            self.exclude_sweep_size,
            self.abs_thresholds,
            self.neighbours_mask,
            self.num_channels,
        )
        if z_inds.size == 0:
            return (np.zeros(0, dtype=self._dtype),)

        # the template index is the channel index because one template per channel
        chan_inds = template_inds
        if self.peak_sign == "both":
            chan_inds = chan_inds % self.num_channels

        samples_inds = samples_inds + self.conv_margin + self.nbefore

        peak_amplitudes = traces[samples_inds, chan_inds]

        local_peaks = np.zeros(samples_inds.size, dtype=self._dtype)
        local_peaks["sample_index"] = samples_inds
        local_peaks["channel_index"] = chan_inds
        local_peaks["amplitude"] = peak_amplitudes
        local_peaks["segment_index"] = segment_index
        local_peaks["z"] = z_inds

        # return is always a tuple
        return (local_peaks,)

    def get_convolved_traces(self, traces):
        from scipy.signal import oaconvolve

        tmp = oaconvolve(self.prototype[None, :], traces.T, axes=1, mode="valid")
        scalar_products = self.weights.dot(tmp)
        return scalar_products


if HAVE_NUMBA:
    import numba

    @numba.jit(nopython=True, parallel=False, nogil=True, fastmath=True)
    def _numba_detect_peak_matched_filtering(
        conv_traces, exclude_sweep_size, abs_thresholds, neighbours_mask, num_channels
    ):
        num_z = conv_traces.shape[0]
        num_templates = conv_traces.shape[1]
        num_samples = conv_traces.shape[2]

        # first find peaks
        peak_mask = np.zeros(conv_traces.shape, dtype="bool")
        for z_ind in range(num_z):
            for temp_ind in range(num_templates):
                for s in range(1, num_samples - 1):
                    value = conv_traces[z_ind, temp_ind, s]
                    if (
                        (value >= abs_thresholds[z_ind, temp_ind])
                        and (value > conv_traces[z_ind, temp_ind, s - 1])
                        and (value >= conv_traces[z_ind, temp_ind, s + 1])
                    ):
                        peak_mask[z_ind, temp_ind, s] = True

        (
            z_inds,
            template_inds,
            samples_inds,
        ) = np.nonzero(peak_mask)
        # order = np.lexsort((z_inds, template_inds, samples_inds))
        order = np.argsort(samples_inds)
        z_inds, template_inds, samples_inds = z_inds[order], template_inds[order], samples_inds[order]

        npeaks = samples_inds.size
        keep_peak = np.ones(npeaks, dtype="bool")
        next_start = 0
        for i in range(npeaks):
            if (samples_inds[i] < exclude_sweep_size + 1) or (
                samples_inds[i] >= (num_samples - exclude_sweep_size - 1)
            ):
                keep_peak[i] = False
                continue

            for j in range(next_start, npeaks):
                if i == j:
                    continue

                if samples_inds[i] + exclude_sweep_size < samples_inds[j]:
                    break

                if samples_inds[i] - exclude_sweep_size > samples_inds[j]:
                    next_start = j
                    continue

                # search for neighbors with higher amplitudes
                # note : % num_channels is because when 'both' is used because templates are twice
                if neighbours_mask[template_inds[i] % num_channels, template_inds[j] % num_channels]:
                    # if inside spatial zone ...
                    if abs(samples_inds[i] - samples_inds[j]) <= exclude_sweep_size:
                        # ...and if inside tempral zone ...
                        value_i = (
                            conv_traces[z_inds[i], template_inds[i], samples_inds[i]]
                            / abs_thresholds[z_inds[i], template_inds[i]]
                        )
                        value_j = (
                            conv_traces[z_inds[j], template_inds[j], samples_inds[j]]
                            / abs_thresholds[z_inds[j], template_inds[j]]
                        )

                        if value_j > value_i:
                            # ... and if smaller
                            keep_peak[i] = False
                            break
                        if (value_j == value_i) & (samples_inds[i] > samples_inds[j]):
                            keep_peak[i] = False
                            # ... equal but after
                            break
                        if (value_j == value_i) & (samples_inds[i] == samples_inds[j]) & (z_inds[i] > z_inds[j]):
                            # ... equal + same time but not same depth (z)
                            keep_peak[i] = False
                            break

        z_inds, template_inds, samples_inds = z_inds[keep_peak], template_inds[keep_peak], samples_inds[keep_peak]

        return z_inds, template_inds, samples_inds
