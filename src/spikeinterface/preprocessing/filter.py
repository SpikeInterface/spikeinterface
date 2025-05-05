from __future__ import annotations

import numpy as np

from spikeinterface.core.core_tools import define_function_handling_dict_from_class
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

from spikeinterface.core import get_chunk_with_margin


_common_filter_docs = """**filter_kwargs : dict
        Certain keyword arguments for `scipy.signal` filters:
            filter_order : order
                The order of the filter. Note as filtering is applied with scipy's
                `filtfilt` functions (i.e. acausal, zero-phase) the effective
                order will be double the `filter_order`.
            filter_mode :  "sos" | "ba", default: "sos"
                Filter form of the filter coefficients:
                - second-order sections ("sos")
                - numerator/denominator : ("ba")
            ftype : str, default: "butter"
                Filter type for `scipy.signal.iirfilter` e.g. "butter", "cheby1"."""


class FilterRecording(BasePreprocessor):
    """
    A generic filter class based on:
        For filter coefficient generation:
            * scipy.signal.iirfilter
        For filter application:
            * scipy.signal.filtfilt or scipy.signal.sosfiltfilt when direction = "forward-backward"
            * scipy.signal.lfilter or scipy.signal.sosfilt when direction = "forward" or "backward"

    BandpassFilterRecording is built on top of it.

    Parameters
    ----------
    recording : Recording
        The recording extractor to be re-referenced
    band : float or list, default: [300.0, 6000.0]
        If float, cutoff frequency in Hz for "highpass" filter type
        If list. band (low, high) in Hz for "bandpass" filter type
    btype : "bandpass" | "highpass", default: "bandpass"
        Type of the filter
    margin_ms : float, default: 5.0
        Margin in ms on border to avoid border effect
    coeff : array | None, default: None
        Filter coefficients in the filter_mode form.
    dtype : dtype or None, default: None
        The dtype of the returned traces. If None, the dtype of the parent recording is used
    add_reflect_padding : Bool, default False
        If True, uses a left and right margin during calculation.
    filter_order : order
        The order of the filter for `scipy.signal.iirfilter`
    filter_mode :  "sos" | "ba", default: "sos"
        Filter form of the filter coefficients for `scipy.signal.iirfilter`:
        - second-order sections ("sos")
        - numerator/denominator : ("ba")
    ftype : str, default: "butter"
        Filter type for `scipy.signal.iirfilter` e.g. "butter", "cheby1".
    direction : "forward" | "backward" | "forward-backward", default: "forward-backward"
        Direction of filtering:
        - "forward" - filter is applied to the timeseries in one direction, creating phase shifts
        - "backward" - the timeseries is reversed, the filter is applied and filtered timeseries reversed again. Creates phase shifts in the opposite direction to "forward"
        - "forward-backward" - Applies the filter in the forward and backward direction, resulting in zero-phase filtering. Note this doubles the effective filter order.

    Returns
    -------
    filter_recording : FilterRecording
        The filtered recording extractor object
    """

    def __init__(
        self,
        recording,
        band=[300.0, 6000.0],
        btype="bandpass",
        filter_order=5,
        ftype="butter",
        filter_mode="sos",
        margin_ms=5.0,
        add_reflect_padding=False,
        coeff=None,
        dtype=None,
        direction="forward-backward",
    ):
        import scipy.signal

        assert filter_mode in ("sos", "ba"), "'filter' mode must be 'sos' or 'ba'"
        fs = recording.get_sampling_frequency()
        if coeff is None:
            assert btype in ("bandpass", "highpass"), "'bytpe' must be 'bandpass' or 'highpass'"
            # coefficient
            # self.coeff is 'sos' or 'ab' style
            filter_coeff = scipy.signal.iirfilter(
                filter_order, band, fs=fs, analog=False, btype=btype, ftype=ftype, output=filter_mode
            )
        else:
            filter_coeff = coeff
            if not isinstance(coeff, list):
                if filter_mode == "ba":
                    coeff = [c.tolist() for c in coeff]
                else:
                    coeff = coeff.tolist()
        dtype = fix_dtype(recording, dtype)

        BasePreprocessor.__init__(self, recording, dtype=dtype)
        self.annotate(is_filtered=True)

        if "offset_to_uV" in self.get_property_keys():
            self.set_channel_offsets(0)

        margin = int(margin_ms * fs / 1000.0)
        for parent_segment in recording._recording_segments:
            self.add_recording_segment(
                FilterRecordingSegment(
                    parent_segment,
                    filter_coeff,
                    filter_mode,
                    margin,
                    dtype,
                    add_reflect_padding=add_reflect_padding,
                    direction=direction,
                )
            )

        self._kwargs = dict(
            recording=recording,
            band=band,
            btype=btype,
            filter_order=filter_order,
            ftype=ftype,
            filter_mode=filter_mode,
            coeff=coeff,
            margin_ms=margin_ms,
            add_reflect_padding=add_reflect_padding,
            dtype=dtype.str,
            direction=direction,
        )


class FilterRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment,
        coeff,
        filter_mode,
        margin,
        dtype,
        add_reflect_padding=False,
        direction="forward-backward",
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.coeff = coeff
        self.filter_mode = filter_mode
        self.direction = direction
        self.margin = margin
        self.add_reflect_padding = add_reflect_padding
        self.dtype = dtype

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces_chunk, left_margin, right_margin = get_chunk_with_margin(
            self.parent_recording_segment,
            start_frame,
            end_frame,
            channel_indices,
            self.margin,
            add_reflect_padding=self.add_reflect_padding,
        )

        traces_dtype = traces_chunk.dtype
        # if uint --> force int
        if traces_dtype.kind == "u":
            traces_chunk = traces_chunk.astype("float32")

        import scipy.signal

        if self.direction == "forward-backward":
            if self.filter_mode == "sos":
                filtered_traces = scipy.signal.sosfiltfilt(self.coeff, traces_chunk, axis=0)
            elif self.filter_mode == "ba":
                b, a = self.coeff
                filtered_traces = scipy.signal.filtfilt(b, a, traces_chunk, axis=0)
        else:
            if self.direction == "backward":
                traces_chunk = np.flip(traces_chunk, axis=0)

            if self.filter_mode == "sos":
                filtered_traces = scipy.signal.sosfilt(self.coeff, traces_chunk, axis=0)
            elif self.filter_mode == "ba":
                b, a = self.coeff
                filtered_traces = scipy.signal.lfilter(b, a, traces_chunk, axis=0)

            if self.direction == "backward":
                filtered_traces = np.flip(filtered_traces, axis=0)

        if right_margin > 0:
            filtered_traces = filtered_traces[left_margin:-right_margin, :]
        else:
            filtered_traces = filtered_traces[left_margin:, :]

        if np.issubdtype(self.dtype, np.integer):
            filtered_traces = filtered_traces.round()

        return filtered_traces.astype(self.dtype)


class BandpassFilterRecording(FilterRecording):
    """
    Bandpass filter of a recording

    Parameters
    ----------
    recording : Recording
        The recording extractor to be re-referenced
    freq_min : float
        The highpass cutoff frequency in Hz
    freq_max : float
        The lowpass cutoff frequency in Hz
    margin_ms : float
        Margin in ms on border to avoid border effect
    dtype : dtype or None
        The dtype of the returned traces. If None, the dtype of the parent recording is used
    {}

    Returns
    -------
    filter_recording : BandpassFilterRecording
        The bandpass-filtered recording extractor object
    """

    def __init__(self, recording, freq_min=300.0, freq_max=6000.0, margin_ms=5.0, dtype=None, **filter_kwargs):
        FilterRecording.__init__(
            self, recording, band=[freq_min, freq_max], margin_ms=margin_ms, dtype=dtype, **filter_kwargs
        )
        dtype = fix_dtype(recording, dtype)
        self._kwargs = dict(
            recording=recording, freq_min=freq_min, freq_max=freq_max, margin_ms=margin_ms, dtype=dtype.str
        )
        self._kwargs.update(filter_kwargs)


class HighpassFilterRecording(FilterRecording):
    """
    Highpass filter of a recording

    Parameters
    ----------
    recording : Recording
        The recording extractor to be re-referenced
    freq_min : float
        The highpass cutoff frequency in Hz
    margin_ms : float
        Margin in ms on border to avoid border effect
    dtype : dtype or None
        The dtype of the returned traces. If None, the dtype of the parent recording is used
    {}

    Returns
    -------
    filter_recording : HighpassFilterRecording
        The highpass-filtered recording extractor object
    """

    def __init__(self, recording, freq_min=300.0, margin_ms=5.0, dtype=None, **filter_kwargs):
        FilterRecording.__init__(
            self, recording, band=freq_min, margin_ms=margin_ms, dtype=dtype, btype="highpass", **filter_kwargs
        )
        dtype = fix_dtype(recording, dtype)
        self._kwargs = dict(recording=recording, freq_min=freq_min, margin_ms=margin_ms, dtype=dtype.str)
        self._kwargs.update(filter_kwargs)


class NotchFilterRecording(BasePreprocessor):
    """
    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to be notch-filtered
    freq : int or float
        The target frequency in Hz of the notch filter
    q : int
        The quality factor of the notch filter
    dtype : None | dtype, default: None
        dtype of recording. If None, will take from `recording`
    margin_ms : float, default: 5.0
        Margin in ms on border to avoid border effect

    Returns
    -------
    filter_recording : NotchFilterRecording
        The notch-filtered recording extractor object
    """

    def __init__(self, recording, freq=3000, q=30, margin_ms=5.0, dtype=None):
        # coeef is 'ba' type
        fn = 0.5 * float(recording.get_sampling_frequency())
        import scipy.signal

        coeff = scipy.signal.iirnotch(freq / fn, q)

        if dtype is None:
            dtype = recording.get_dtype()
        dtype = np.dtype(dtype)

        # if uint --> unsupported
        if dtype.kind == "u":
            raise TypeError(
                "The notch filter only supports signed types. Use the 'dtype' argument"
                "to specify a signed type (e.g. 'int16', 'float32')"
            )

        BasePreprocessor.__init__(self, recording, dtype=dtype)
        self.annotate(is_filtered=True)

        sf = recording.get_sampling_frequency()
        margin = int(margin_ms * sf / 1000.0)
        for parent_segment in recording._recording_segments:
            self.add_recording_segment(FilterRecordingSegment(parent_segment, coeff, "ba", margin, dtype))

        self._kwargs = dict(recording=recording, freq=freq, q=q, margin_ms=margin_ms, dtype=dtype.str)


# functions for API
filter = define_function_handling_dict_from_class(source_class=FilterRecording, name="filter")
bandpass_filter = define_function_handling_dict_from_class(source_class=BandpassFilterRecording, name="bandpass_filter")
notch_filter = define_function_handling_dict_from_class(source_class=NotchFilterRecording, name="notch_filter")
highpass_filter = define_function_handling_dict_from_class(source_class=HighpassFilterRecording, name="highpass_filter")


def causal_filter(
    recording,
    direction="forward",
    band=[300.0, 6000.0],
    btype="bandpass",
    filter_order=5,
    ftype="butter",
    filter_mode="sos",
    margin_ms=5.0,
    add_reflect_padding=False,
    coeff=None,
    dtype=None,
):
    """
    Generic causal filter built on top of the filter function.

    Parameters
    ----------
    recording : Recording
        The recording extractor to be re-referenced
    direction : "forward" | "backward", default: "forward"
        Direction of causal filter. The "backward" option flips the traces in time before applying the filter
        and then flips them back.
    band : float or list, default: [300.0, 6000.0]
        If float, cutoff frequency in Hz for "highpass" filter type
        If list. band (low, high) in Hz for "bandpass" filter type
    btype : "bandpass" | "highpass", default: "bandpass"
        Type of the filter
    margin_ms : float, default: 5.0
        Margin in ms on border to avoid border effect
    coeff : array | None, default: None
        Filter coefficients in the filter_mode form.
    dtype : dtype or None, default: None
        The dtype of the returned traces. If None, the dtype of the parent recording is used
    add_reflect_padding : Bool, default False
        If True, uses a left and right margin during calculation.
    filter_order : order
        The order of the filter for `scipy.signal.iirfilter`
    filter_mode :  "sos" | "ba", default: "sos"
        Filter form of the filter coefficients for `scipy.signal.iirfilter`:
        - second-order sections ("sos")
        - numerator/denominator : ("ba")
    ftype : str, default: "butter"
        Filter type for `scipy.signal.iirfilter` e.g. "butter", "cheby1".

    Returns
    -------
    filter_recording : FilterRecording
        The causal-filtered recording extractor object
    """
    assert direction in ["forward", "backward"], "Direction must be either 'forward' or 'backward'"
    return filter(
        recording=recording,
        direction=direction,
        band=band,
        btype=btype,
        filter_order=filter_order,
        ftype=ftype,
        filter_mode=filter_mode,
        margin_ms=margin_ms,
        add_reflect_padding=add_reflect_padding,
        coeff=coeff,
        dtype=dtype,
    )


bandpass_filter.__doc__ = bandpass_filter.__doc__.format(_common_filter_docs)
highpass_filter.__doc__ = highpass_filter.__doc__.format(_common_filter_docs)


def fix_dtype(recording, dtype):
    if dtype is None:
        dtype = recording.get_dtype()
    dtype = np.dtype(dtype)

    # if uint --> force int
    if dtype.kind == "u":
        dtype = np.dtype(dtype.str.replace("u", "i"))

    return dtype
