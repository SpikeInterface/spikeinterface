from __future__ import annotations

import math
import threading

import numpy as np

from spikeinterface.core.core_tools import define_function_handling_dict_from_class

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from .filter import fix_dtype


def _detect_l2_bytes():
    """Per-core L2 size from Linux sysfs, else 1 MB, which fits in any current core's L2."""
    for index in range(5):
        cache = f"/sys/devices/system/cpu/cpu0/cache/index{index}"
        try:
            with open(f"{cache}/level") as f:
                level = int(f.read())
            with open(f"{cache}/type") as f:
                cache_type = f.read().strip()
            if level != 2 or cache_type not in ("Unified", "Data"):
                continue
            with open(f"{cache}/size") as f:
                size = f.read().strip()
        except (OSError, ValueError):
            continue
        scale = {"K": 1024, "M": 1024**2}.get(size[-1:], 1)
        return int(size.rstrip("KM")) * scale
    return 1_000_000


_L2_BYTES = _detect_l2_bytes()


def _overlap_save_fft_length(signal_length, num_taps):
    """FFT length minimising FFT work per output sample, as ``scipy.signal.oaconvolve`` chooses it.

    Unlike ``oaconvolve``, the length is rounded up to one that is fast for a real FFT: lengths with a factor of 7
    or 11 are fast for a complex FFT but up to 1.5 times slower for ``rfft``.
    """
    import scipy.fft
    from scipy.special import lambertw

    whole = int(scipy.fft.next_fast_len(signal_length + num_taps - 1, real=True))
    if num_taps == 1 or num_taps >= signal_length / 2:
        return whole
    overlap = num_taps - 1
    optimal = -overlap * lambertw(-1 / (2 * math.e * overlap), k=-1).real
    return min(whole, int(scipy.fft.next_fast_len(math.ceil(optimal), real=True)))


def _channels_per_block(num_channels, fft_length, bytes_per_sample, pool_size=1, headroom=0.6):
    """
    Channels per overlap-save block: as many as keep the block's buffers in L2, but no more than gives each of
    `pool_size` workers a block.
    """
    fit_l2 = max(1, int(headroom * _L2_BYTES / (bytes_per_sample * fft_length)))
    return min(fit_l2, max(1, math.ceil(num_channels / pool_size)))


class FIRFilterRecording(BasePreprocessor):
    """
    Applies a FIR filter to each channel, by overlap-save convolution.

    Output sample ``n`` is ``sum_j taps[j] * x[n + center - j]``, so with the default `center` a symmetric
    kernel is zero-phase. Every output sample depends only on the ``len(taps) - 1`` input samples around it,
    which the filter reads from the parent, so the output does not depend on how it is chunked.

    Parameters
    ----------
    recording : Recording
        The recording to filter.
    taps : array-like
        Kernel of shape ``(num_taps,)``, applied to every channel, or ``(num_kernels, num_taps)``, with each
        channel's row given by `channel_groups`. Designed with, for example, ``design_fir_bandpass`` or
        ``scipy.signal.firwin``.
    channel_groups : array-like | None, default: None
        For a 2D `taps`, the row of `taps` for each channel. None requires one row per channel.
    center : int | None, default: None
        Index of the tap aligned with the output sample. None means ``(num_taps - 1) // 2``, zero-phase for a
        symmetric kernel; 0 makes the filter causal.
    dtype : dtype | None, default: None
        The dtype of the output. None uses the parent's. Computation is in float32, and integer output is
        rounded.
    add_reflect_padding : bool, default: False
        If True, the input is reflected beyond the recording's edges, otherwise it is zero there.

    Returns
    -------
    filtered_recording : FIRFilterRecording
        The filtered recording.
    """

    def __init__(self, recording, taps, channel_groups=None, center=None, dtype=None, add_reflect_padding=False):
        taps = np.asarray(taps, dtype=np.float64)
        if taps.ndim not in (1, 2) or taps.shape[-1] == 0:
            raise ValueError(f"taps must have shape (num_taps,) or (num_kernels, num_taps), got {taps.shape}")
        kernels = np.atleast_2d(taps)
        num_kernels, num_taps = kernels.shape
        num_channels = recording.get_num_channels()
        if channel_groups is None:
            if num_kernels not in (1, num_channels):
                raise ValueError(f"{num_kernels} kernels for {num_channels} channels need channel_groups")
            groups = np.zeros(num_channels, dtype=np.intp) if num_kernels == 1 else np.arange(num_channels)
        else:
            groups = np.asarray(channel_groups, dtype=np.intp)
            if groups.shape != (num_channels,) or groups.min() < 0 or groups.max() >= num_kernels:
                raise ValueError(f"channel_groups must give a row of taps, 0 to {num_kernels - 1}, for each channel")
        center = (num_taps - 1) // 2 if center is None else int(center)
        if not 0 <= center < num_taps:
            raise ValueError(f"center must be in [0, {num_taps}), got {center}")
        dtype = fix_dtype(recording, dtype)

        BasePreprocessor.__init__(self, recording, dtype=dtype)
        self.annotate(is_filtered=True)
        if "offset_to_uV" in self.get_property_keys():
            # A constant offset passes through scaled by each kernel's DC gain.
            self.set_channel_offsets(self.get_channel_offsets() * kernels.sum(axis=1)[groups])

        for parent_segment in recording.segments:
            self.add_recording_segment(
                FIRFilterRecordingSegment(parent_segment, kernels, groups, center, dtype, add_reflect_padding)
            )

        self._kwargs = dict(
            recording=recording,
            taps=taps.tolist(),
            channel_groups=None if channel_groups is None else groups.tolist(),
            center=center,
            dtype=dtype.str,
            add_reflect_padding=add_reflect_padding,
        )


class FIRFilterRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, kernels, channel_groups, center, dtype, add_reflect_padding):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.kernels = kernels
        self.num_taps = kernels.shape[1]
        self.channel_groups = channel_groups
        self.center = center
        self.dtype = dtype
        self.add_reflect_padding = add_reflect_padding
        # Kernel spectra depend on the FFT length, which depends on the chunk length.
        self._kernel_spectra = {}
        self._kernel_spectra_lock = threading.Lock()

    def _get_kernel_spectra(self, fft_length):
        import scipy.fft

        with self._kernel_spectra_lock:
            spectra = self._kernel_spectra.get(fft_length)
            if spectra is None:
                spectra = scipy.fft.rfft(self.kernels.T, n=fft_length, axis=0).astype(np.complex64)
                self._kernel_spectra[fft_length] = spectra
        return spectra

    def _get_input(self, start_frame, end_frame, channel_indices):
        """Parent traces for every input sample the kernel reaches, padded beyond the recording's edges."""
        left = self.num_taps - 1 - self.center
        right = self.center
        fetch_start = max(0, start_frame - left)
        fetch_end = min(self.parent_recording_segment.get_num_samples(), end_frame + right)
        traces = self.parent_recording_segment.get_traces(fetch_start, fetch_end, channel_indices)
        pad = ((fetch_start - (start_frame - left), end_frame + right - fetch_end), (0, 0))
        if pad[0] != (0, 0):
            traces = np.pad(traces, pad, mode="reflect" if self.add_reflect_padding else "constant")
        return traces

    def get_traces(self, start_frame, end_frame, channel_indices):
        import scipy.fft

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()
        if channel_indices is None:
            channel_indices = slice(None)
        traces = self._get_input(start_frame, end_frame, channel_indices)
        groups = self.channel_groups[channel_indices]
        num_samples, num_channels = end_frame - start_frame, traces.shape[1]
        num_taps = self.num_taps

        fft_length = _overlap_save_fft_length(num_samples + num_taps - 1, num_taps)
        step = fft_length - num_taps + 1
        spectra = self._get_kernel_spectra(fft_length)
        one_kernel = spectra.shape[1] == 1
        # The float32 block and its complex64 spectrum take 8 bytes per sample, and per-channel kernel spectra 4 more.
        slab = _channels_per_block(num_channels, fft_length, bytes_per_sample=8 if one_kernel else 12)

        # Each block is rounded and cast as it is written, so no full float32 copy of the chunk is held.
        out = np.empty((num_samples, num_channels), dtype=self.dtype)
        round_to_integer = np.issubdtype(self.dtype, np.integer)
        buffer = np.empty((fft_length, slab), dtype=np.float32)
        for c0 in range(0, num_channels, slab):
            c1 = min(num_channels, c0 + slab)
            tile = buffer[:, : c1 - c0]
            tile_spectra = spectra if one_kernel else spectra[:, groups[c0:c1]]
            for t0 in range(0, num_samples, step):
                num_out = min(step, num_samples - t0)
                num_in = num_out + num_taps - 1
                tile[:num_in] = traces[t0 : t0 + num_in, c0:c1]
                tile[num_in:] = 0
                spectrum = scipy.fft.rfft(tile, axis=0)
                spectrum *= tile_spectra
                # The first num_taps - 1 samples of each circular convolution wrap around and are discarded.
                result = scipy.fft.irfft(spectrum, n=fft_length, axis=0, overwrite_x=True)
                valid = result[num_taps - 1 : num_taps - 1 + num_out]
                if round_to_integer:
                    np.rint(valid, out=valid)
                out[t0 : t0 + num_out, c0:c1] = valid

        return out


fir_filter = define_function_handling_dict_from_class(source_class=FIRFilterRecording, name="fir_filter")
