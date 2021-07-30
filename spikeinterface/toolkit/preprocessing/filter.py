import scipy.signal

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

from .tools import get_chunk_with_margin


class FilterRecording(BasePreprocessor):
    """
    Generic filter class based on:
      * scipy.signal.iirfilter
      * scipy.signal.filtfilt or scipy.signal.sosfilt
    BandpassFilterRecording is build on top of it.

    Parameters
    ----------
    recording: Recording
        The recording extractor to be re-referenced
    
    N: order
    filter_mode: 'sos or 'ba'
        'sos' is bi quadratic and more stable than ab so thery are prefered.
    ftypestr: 'butter' / 'cheby1' / ... all possible of scipy.signal.iirdesign
    
    
    margin: margin in second on border to avoid border effect
    
    """
    name = 'filter'

    def __init__(self, recording, band=[300., 6000.], btype='bandpass',
                 filter_order=5, ftype='butter', filter_mode='sos', margin_ms=5.0):

        assert btype in ('bandpass', 'lowpass', 'highpass', 'bandstop')
        assert filter_mode in ('sos', 'ba')

        # coefficient
        sf = recording.get_sampling_frequency()
        if btype in ('bandpass', 'bandstop'):
            assert len(band) == 2
            Wn = [e / sf * 2 for e in band]
        else:
            Wn = float(band) / sf * 2
        N = filter_order
        # self.coeff is 'sos' or 'ab' style
        coeff = scipy.signal.iirfilter(N, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)

        BasePreprocessor.__init__(self, recording)
        self.annotate(is_filtered=True)

        margin = int(margin_ms * sf / 1000.)
        for parent_segment in recording._recording_segments:
            self.add_recording_segment(FilterRecordingSegment(parent_segment, coeff, filter_mode, margin))

        self._kwargs = dict(recording=recording.to_dict(), band=band, btype=btype,
                            filter_order=filter_order, ftype=ftype, filter_mode=filter_mode, margin_ms=margin_ms)


class FilterRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, coeff, filter_mode, margin):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.coeff = coeff
        self.filter_mode = filter_mode
        self.margin = margin

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces_chunk, left_margin, right_margin = get_chunk_with_margin(self.parent_recording_segment,
                                                                        start_frame, end_frame, channel_indices,
                                                                        self.margin)

        if self.filter_mode == 'sos':
            filtered_traces = scipy.signal.sosfiltfilt(self.coeff, traces_chunk, axis=0)
        elif self.filter_mode == 'ba':
            b, a = self.coeff
            filtered_traces = scipy.signal.filtfilt(b, a, traces_chunk, axis=0)

        if right_margin > 0:
            filtered_traces = filtered_traces[left_margin:-right_margin, :]
        else:
            filtered_traces = filtered_traces[left_margin:, :]
        return filtered_traces


class BandpassFilterRecording(FilterRecording):
    """
    Simplied bandpass class on top of FilterRecording.
    """
    name = 'bandpass_filter'

    def __init__(self, recording, freq_min=300., freq_max=6000., margin_ms=5.0):
        FilterRecording.__init__(self, recording, band=[freq_min, freq_max], margin_ms=margin_ms)
        self._kwargs = dict(recording=recording.to_dict(), freq_min=freq_min, freq_max=freq_max, margin_ms=margin_ms)


class NotchFilterRecording(BasePreprocessor):
    """
    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be notch-filtered.
    freq: int or float
        The target frequency of the notch filter.
    q: int
        The quality factor of the notch filter.
    
    Returns
    -------
    filter_recording: NotchFilterRecording
        The notch-filtered recording extractor object
    """
    name = 'notch_filter'

    def __init__(self, recording, freq=3000, q=30, margin_ms=5.0):
        # coeef is 'ba' type
        fn = 0.5 * float(recording.get_sampling_frequency())
        coeff = scipy.signal.iirnotch(freq / fn, q)

        BasePreprocessor.__init__(self, recording)
        self.annotate(is_filtered=True)

        sf = recording.get_sampling_frequency()
        margin = int(margin_ms * sf / 1000.)
        for parent_segment in recording._recording_segments:
            self.add_recording_segment(FilterRecordingSegment(parent_segment, coeff, 'ba', margin))

        self._kwargs = dict(recording=recording.to_dict(), freq=freq, q=q, margin_ms=margin_ms)


# functions for API

def filter(recording, engine='scipy', **kwargs):
    if engine == 'scipy':
        return FilterRecording(recording, **kwargs)
    elif engine == 'opencl':
        from .filter_opencl import FilterOpenCLRecording
        return FilterOpenCLRecording(recording, **kwargs)


filter.__doc__ = FilterRecording.__doc__


def bandpass_filter(*args, **kwargs):
    return BandpassFilterRecording(*args, **kwargs)


bandpass_filter.__doc__ = BandpassFilterRecording.__doc__


def notch_filter(*args, **kwargs):
    return NotchFilterRecording(*args, **kwargs)


notch_filter.__doc__ = NotchFilterRecording.__doc__
