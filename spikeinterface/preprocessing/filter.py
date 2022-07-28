import numpy as np
import scipy.signal

from spikeinterface.core.core_tools import define_function_from_class
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

from ..core import get_chunk_with_margin


_common_filter_docs = \
    """**filter_kwargs: keyword arguments for parallel processing:
            * filter_order: order
                The order of the filter
            * filter_mode: 'sos or 'ba'
                'sos' is bi quadratic and more stable than ab so thery are prefered.
            * ftype: str
                Filter type for iirdesign ('butter' / 'cheby1' / ... all possible of scipy.signal.iirdesign)
    """


class FilterRecording(BasePreprocessor):
    """
    Generic filter class based on:
      * scipy.signal.iirfilter
      * scipy.signal.filtfilt or scipy.signal.sosfilt
    BandpassFilterRecording is built on top of it.

    Parameters
    ----------
    recording: Recording
        The recording extractor to be re-referenced
    band: float or list
        If float, cutoff frequency in Hz for 'highpass' filter type
        If list. band (low, high) in Hz for 'bandpass' filter type
    btype: str
        Type of the filter ('bandpass', 'highpass')
    margin_ms: float
        Margin in ms on border to avoid border effect
    filter_mode: str 'sos' or 'ba'
        Filter form of the filter coefficients:
        - second-order sections (default): 'sos'
        - numerator/denominator: 'ba'
    coef: ndarray or None
        Filter coefficients in the filter_mode form. 
    dtype: dtype or None
        The dtype of the returned traces. If None, the dtype of the parent recording is used
    {}

    Returns
    -------
    filter_recording: FilterRecording
        The filtered recording extractor object

    """
    name = 'filter'

    def __init__(self, recording, band=[300., 6000.], btype='bandpass',
                 filter_order=5, ftype='butter', filter_mode='sos', margin_ms=5.0,
                 coeff=None, dtype=None):

        assert filter_mode in ('sos', 'ba')
        sf = recording.get_sampling_frequency()
        if coeff is None:
            assert btype in ('bandpass', 'highpass')
            # coefficient
            if btype in ('bandpass', 'bandstop'):
                assert len(band) == 2
                Wn = [e / sf * 2 for e in band]
            else:
                Wn = float(band) / sf * 2
            N = filter_order
            # self.coeff is 'sos' or 'ab' style
            filter_coeff = scipy.signal.iirfilter(N, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)
        else:
            filter_coeff = coeff
            if not isinstance(coeff, list):
                if filter_mode == 'ba':
                    coeff = [c.tolist() for c in coeff]
                else:
                    coeff = coeff.tolist()
        dtype = fix_dtype(recording, dtype)

        BasePreprocessor.__init__(self, recording, dtype=dtype)
        self.annotate(is_filtered=True)

        if "offset_to_uV" in self.get_property_keys():
            self.set_channel_offsets(0)

        margin = int(margin_ms * sf / 1000.)
        for parent_segment in recording._recording_segments:
            self.add_recording_segment(FilterRecordingSegment(parent_segment, filter_coeff, filter_mode, margin,
                                                              dtype))

        self._kwargs = dict(recording=recording.to_dict(), band=band, btype=btype,
                            filter_order=filter_order, ftype=ftype, filter_mode=filter_mode, coeff=coeff,
                            margin_ms=margin_ms)


class FilterRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, coeff, filter_mode, margin, dtype):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.coeff = coeff
        self.filter_mode = filter_mode
        self.margin = margin
        self.dtype = dtype

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces_chunk, left_margin, right_margin = get_chunk_with_margin(self.parent_recording_segment,
                                                                        start_frame, end_frame, channel_indices,
                                                                        self.margin)

        traces_dtype = traces_chunk.dtype
        # if uint --> force int
        if traces_dtype.kind == "u":
            traces_chunk = traces_chunk.astype("float32")

        if self.filter_mode == 'sos':
            filtered_traces = scipy.signal.sosfiltfilt(self.coeff, traces_chunk, axis=0)
        elif self.filter_mode == 'ba':
            b, a = self.coeff
            filtered_traces = scipy.signal.filtfilt(b, a, traces_chunk, axis=0)

        if right_margin > 0:
            filtered_traces = filtered_traces[left_margin:-right_margin, :]
        else:
            filtered_traces = filtered_traces[left_margin:, :]
        return filtered_traces.astype(self.dtype)


class BandpassFilterRecording(FilterRecording):
    """
    Bandpass filter of a recording

    Parameters
    ----------
    recording: Recording
        The recording extractor to be re-referenced
    freq_min: float
        The highpass cutoff frequency in Hz
    freq_max: float
        The lowpass cutoff frequency in Hz
    margin_ms: float
        Margin in ms on border to avoid border effect
    dtype: dtype or None
        The dtype of the returned traces. If None, the dtype of the parent recording is used
    {}
    Returns
    -------
    filter_recording: BandpassFilterRecording
        The bandpass-filtered recording extractor object
    """
    name = 'bandpass_filter'

    def __init__(self, recording, freq_min=300., freq_max=6000., margin_ms=5.0, dtype=None, **filter_kwargs):
        FilterRecording.__init__(self, recording, band=[freq_min, freq_max], margin_ms=margin_ms, dtype=dtype,
                                 **filter_kwargs)
        self._kwargs = dict(recording=recording.to_dict(), freq_min=freq_min, freq_max=freq_max, margin_ms=margin_ms)
        self._kwargs.update(filter_kwargs)


class HighpassFilterRecording(FilterRecording):
    """
    Highpass filter of a recording

    Parameters
    ----------
    recording: Recording
        The recording extractor to be re-referenced
    freq_min: float
        The highpass cutoff frequency in Hz
    margin_ms: float
        Margin in ms on border to avoid border effect
    dtype: dtype or None
        The dtype of the returned traces. If None, the dtype of the parent recording is used
    {}
    Returns
    -------
    filter_recording: HighpassFilterRecording
        The highpass-filtered recording extractor object
    """
    name = 'highpass_filter'

    def __init__(self, recording, freq_min=300., margin_ms=5.0, dtype=None, **filter_kwargs):
        FilterRecording.__init__(self, recording, band=freq_min, margin_ms=margin_ms, dtype=dtype,
                                 btype='highpass', **filter_kwargs)
        self._kwargs = dict(recording=recording.to_dict(), freq_min=freq_min, margin_ms=margin_ms)
        self._kwargs.update(filter_kwargs)


class NotchFilterRecording(BasePreprocessor):
    """
    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be notch-filtered
    freq: int or float
        The target frequency in Hz of the notch filter
    q: int
        The quality factor of the notch filter
    {}
    Returns
    -------
    filter_recording: NotchFilterRecording
        The notch-filtered recording extractor object
    """
    name = 'notch_filter'

    def __init__(self, recording, freq=3000, q=30, margin_ms=5.0, dtype=None):
        # coeef is 'ba' type
        fn = 0.5 * float(recording.get_sampling_frequency())
        coeff = scipy.signal.iirnotch(freq / fn, q)

        if dtype is None:
            dtype = recording.get_dtype()
        dtype = np.dtype(dtype)

        # if uint --> unsupported
        if dtype.kind == "u":
            raise TypeError("The notch filter only supports signed types. Use the 'dtype' argument"
                            "to specify a signed type (e.g. 'int16', 'float32'")

        BasePreprocessor.__init__(self, recording, dtype=dtype)
        self.annotate(is_filtered=True)

        sf = recording.get_sampling_frequency()
        margin = int(margin_ms * sf / 1000.)
        for parent_segment in recording._recording_segments:
            self.add_recording_segment(FilterRecordingSegment(parent_segment, coeff, 'ba', margin, dtype))

        self._kwargs = dict(recording=recording.to_dict(), freq=freq, q=q, margin_ms=margin_ms)


# functions for API
filter = define_function_from_class(source_class=FilterRecording, name="filter")
bandpass_filter = define_function_from_class(source_class=BandpassFilterRecording, name="bandpass_filter")
notch_filter = define_function_from_class(source_class=NotchFilterRecording, name="notch_filter")
highpass_filter = define_function_from_class(source_class=HighpassFilterRecording, name="highpass_filter")


def fix_dtype(recording, dtype):
    if dtype is None:
        dtype = recording.get_dtype()
    dtype = np.dtype(dtype)

    # if uint --> force int
    if dtype.kind == "u":
        dtype = np.dtype(dtype.str.replace("u", "i"))

    return dtype
