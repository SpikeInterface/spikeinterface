import scipy.signal

from .basepreprocessor import BasePreprocessor,BasePreprocessorSegment

from .tools import get_chunk_with_margin

class FilterRecording(BasePreprocessor):
    """
    Generic filter class based on:
      * scipy.signal.iirfilter
      * scipy.signal.filtfilt or scipy.signal.sosfilt
    BandpassFilterRecording is build on top of it.
    
    N: order
    filter_mode: 'sos or 'ba'
        'sos' is bi quadratic and more stable than ab so thery are prefered.
    ftypestr: 'butter' / 'cheby1' / ... all possible of scipy.signal.iirdesign
    
    
    margin: margin in second on border to avoid border effect
    
    """
    name = 'filter'
    def __init__(self, recording, band=[300., 6000.], btype='bandpass',
                filter_order=5,  ftype='butter', filter_mode='sos', margin=0.005):
        
        assert btype in  ('bandpass', 'lowpass', 'highpass', 'bandstop')
        assert filter_mode in ('sos', 'ba')
        
        # coefficient
        sf = recording.get_sampling_frequency()
        if btype in ('bandpass' , 'bandstop'):
            assert len(band) == 2
            Wn = [e / sf * 2 for e in band]
        else:
            Wn = float(band) / sf * 2
            print('Wn', Wn)
        N = filter_order
        # self.coeff is 'sos' or 'ab' style
        coeff = scipy.signal.iirfilter(N,Wn, analog=False, btype=btype, ftype=ftype, output =filter_mode)
        
        
        BasePreprocessor.__init__(self, recording)
        
        sample_margin = int(margin * sf)
        for parent_segment in recording._recording_segments:
            self.add_recording_segment(FilterRecordingSegment(parent_segment, coeff, filter_mode, sample_margin))
        
        
        
        self._kwargs = dict(recording=recording.to_dict(), band=band, btype=btype,
                filter_order=filter_order, ftype=ftype, filter_mode=filter_mode, margin=margin)
        
    #~ def 

class FilterRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, coeff, filter_mode,sample_margin):
        self.parent_recording_segment = parent_recording_segment
        self.coeff = coeff
        self.filter_mode = filter_mode
        self.sample_margin = sample_margin

    def get_num_samples(self):
        return self.parent_recording_segment.get_num_samples()

    def get_traces(self, start_frame, end_frame, channel_indices):
        print('FilterRecordingSegment.get_traces', start_frame, end_frame, self.get_num_samples())
        traces_chunk, left_margin, right_margin = get_chunk_with_margin(self.parent_recording_segment, 
                    start_frame, end_frame, channel_indices, self.sample_margin)
        
        if self.filter_mode == 'sos':
            filtered_traces = scipy.signal.sosfiltfilt(self.coeff, traces_chunk, axis=0)
        elif self.filter_mode == 'ba':
            b, a = self.coeff
            filtered_traces = scipy.signal.filtfilt(b, a, traces_chunk, axis=0)
        
        if right_margin >0:
            filtered_traces = filtered_traces[left_margin:-right_margin, :]
        else:
            filtered_traces = filtered_traces[left_margin:, :]
        return filtered_traces


class BandpassFilterRecording(FilterRecording):
    """
    Simplied bandpass class on top of FilterRecording.
    """
    name = 'bandpassfilter'
    def __init__(self, recording, freq_min=300., freq_max=6000.):
        FilterRecording.__init__(self, recording, band=[freq_min, freq_max])
        self._kwargs = dict(recording=recording.to_dict(), freq_min=freq_min, freq_max=freq_max)


def filter(*args, **kwargs):
    __doc__ = FilterRecording.__doc__
    return FilterRecording(*args, **kwargs)

def bandpass_filter(*args, **kwargs):
    __doc__ = BandpassFilterRecording.__doc__
    return BandpassFilterRecording(*args, **kwargs)


