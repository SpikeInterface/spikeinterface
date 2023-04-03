import numpy as np
import scipy.interpolate

from spikeinterface.core.core_tools import define_function_from_class
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

from ..core import get_random_data_chunks, get_noise_levels

class SilencedPeriodsRecording(BasePreprocessor):
    """
    Silence user-defined periods from recording extractor traces. By default, 
    periods are zeroed-out (mode = 'zeros'). This is only recommended 
    for traces that are centered around zero (e.g. through a prior highpass
    filter); if this is not the case, you can also fill the periods with noise.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to remove artifacts from
    list_periods: list of lists/arrays
        One list per segment of tuples (start_frame, end_frame) to silence

    mode: str
        Determines what periods are replaced by. Can be one of the following:
            
        - 'zeros' (default): Artifacts are replaced by zeros.

        - 'noise': The periods are filled with a gaussion noise that has the
                   same variance that the one in the recordings, on a per channel
                   basis
    num_chunks_per_segment: int
        Number of chunks per segment for random chunk, by default 20
    chunk_size : int
        Size of a chunk in number for random chunk, by default 10000
    seed : int
        Random seed for random chunk, by default None

    Returns
    -------
    silence_recording: SilencedPeriodsRecording
        The recording extractor after silencing some periods    
    """
    name = 'silence_periods'

    def __init__(self, recording, list_periods, mode='zeros', num_chunks_per_segment=20,
                 chunk_size=10000, seed=None):

        available_modes = ('zeros', 'noise')
        num_seg = recording.get_num_segments()


        if num_seg == 1:
            if isinstance(list_periods, (list, np.ndarray)) and not np.isscalar(list_periods[0]):
                # when unique segment accept list instead of of list of list/arrays
                list_periods = [list_periods]

        # some checks
        assert isinstance(list_periods, list), "'list_periods' must be a list (one per segment)"
        assert len(list_periods) == num_seg, "'list_periods' must have the same length as the number of segments"
        assert all(isinstance(list_periods[i], (list, np.ndarray)) for i in range(num_seg)), \
            "Each element of 'list_periods' must be array-like"

        for periods in list_periods:
            if len(periods) > 0:
                assert np.all(np.diff(np.array(periods), axis=1) > 0), "t_stops should be larger than t_starts"
                assert all(e < s for (_, e), (s, _) in zip(periods, periods[1:])), "Intervals should not overlap"

        assert mode in available_modes, f"mode {mode} is not an available mode: {available_modes}"

        if mode in ['noise']:
            noise_levels = get_noise_levels(recording, return_scaled=False, 
                                    num_chunks_per_segment=num_chunks_per_segment,
                                    chunk_size=chunk_size, concatenated=True, seed=seed)
        else:
            noise_levels = None

        BasePreprocessor.__init__(self, recording)
        for seg_index, parent_segment in enumerate(recording._recording_segments):
            periods = list_periods[seg_index]
            periods = np.asarray(periods, dtype='int64')
            periods = np.sort(self.periods, axis=0)
            rec_segment = SilencedPeriodsRecordingSegment(parent_segment, periods, mode, noise_levels)
            self.add_recording_segment(rec_segment)
        
        self._kwargs = dict(recording=recording.to_dict(), list_periods=list_periods,
                            mode=mode, noise_levels=noise_levels)


class SilencedPeriodsRecordingSegment(BasePreprocessorSegment):

    def __init__(self, parent_recording_segment, periods, mode, noise_levels):

        BasePreprocessorSegment.__init__(self, parent_recording_segment)
        self.periods = periods
        self.mode = mode
        self.noise_levels = noise_levels

    def get_traces(self, start_frame, end_frame, channel_indices):

        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        traces = traces.copy()
        num_channels = traces.shape[1]

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()

        if len(self.periods) > 0:
            new_interval = np.array([start_frame, end_frame])
            lower_index = np.searchsorted(self.periods[:,1], new_interval[0])
            upper_index = np.searchsorted(self.periods[:,0], new_interval[1])

            if upper_index > lower_index:

                intersection = self.periods[lower_index:upper_index]

                for i in intersection:

                    onset = max(0, i[0] - start_frame)
                    offset = min(i[1] - start_frame, end_frame)

                    if self.mode == 'zeros':
                        traces[onset:offset, :] = 0
                    elif self.mode == 'noise':
                        traces[onset:offset, :] = self.noise_levels[channel_indices] * np.random.randn(offset-onset, num_channels)

        return traces


# function for API
silence_periods = define_function_from_class(source_class=SilencedPeriodsRecording, name="silence_periods")
