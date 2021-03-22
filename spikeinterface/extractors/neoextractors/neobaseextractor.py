import numpy as np
from spikeinterface.core import BaseRecording, BaseSorting, BaseRecordingSegment, BaseSortingSegment

import neo


class _NeoBaseExtractor:
    NeoRawIOClass = None
    installed = True
    is_writable = False

    def __init__(self, **neo_kwargs):
        neoIOclass = eval('neo.rawio.' + self.NeoRawIOClass)
        self.neo_reader = neoIOclass(**neo_kwargs)
        self.neo_reader.parse_header()
        
        assert self.neo_reader.block_count() == 1, \
            'This file is neo multi block spikeinterface support one block only dataset'

        self._kwargs = neo_kwargs


class NeoBaseRecordingExtractor(_NeoBaseExtractor, BaseRecording):

    def __init__(self, stream_id=None, **neo_kwargs):
        
        _NeoBaseExtractor.__init__(self, **neo_kwargs)

        # check channel
        # TODO propose a meachanisim to select the appropriate channel groups
        # in neo one channel group have the same dtype/sampling_rate/group_id
        #~ channel_indexes_list = self.neo_reader.get_group_signal_channel_indexes()
        stream_channels = self.neo_reader.header['signal_streams']
        stream_ids = stream_channels['id']
        if stream_id is None:
            if stream_channels.size > 1:
                raise ValueError(f'This reader have several streams({stream_ids}), specify it with stream_id=')
            else:
                stream_id = stream_ids[0]
        else:
            assert stream_id in stream_ids, f'stream_id {stream_id} is no in {stream_ids}'
        
        self.stream_index = list(stream_ids).index(stream_id)
        self.stream_id = stream_id
        
        # need neo 0.10.0
        signal_channels = self.neo_reader.header['signal_channels']
        mask = signal_channels['stream_id'] == stream_id
        signal_channels = signal_channels[mask]
        
        # check channel groups
        chan_ids = signal_channels['id']
        raw_dtypes = signal_channels['dtype']
        
        sampling_frequency = self.neo_reader.get_signal_sampling_rate(stream_index=self.stream_index)
        # TODO propose a mechanism to select scaled/raw dtype
        scaled_dtype = 'float32'
        BaseRecording.__init__(self, sampling_frequency, chan_ids, scaled_dtype)
        
        # spikeinterface for units to be uV implicitly
        units = signal_channels['units']
        if not np.all(np.isin(units, ['V', 'mV', 'uV'])):
            # check that units are V, mV or uV
            error = f'This extractor based on  neo.{self.NeoRawIOClass} have strange units not in (V, mV, uV) {units}'
            print(error)
        self.additional_gain = np.ones(units.size, dtype='float')
        self.additional_gain[units == 'V'] = 1e6
        self.additional_gain[units == 'mV'] = 1e3
        self.additional_gain[units == 'uV'] = 1.
        self.additional_gain = self.additional_gain.reshape(1, -1)
        
        nseg = self.neo_reader.segment_count(block_index=0)
        for segment_index in range(nseg):
            rec_segment = NeoRecordingSegment(self.neo_reader, segment_index, self.stream_index, self.additional_gain)
            self.add_recording_segment(rec_segment)

class NeoRecordingSegment(BaseRecordingSegment):
    def __init__(self, neo_reader, segment_index, stream_index, additional_gain):
        BaseRecordingSegment.__init__(self)
        self.neo_reader = neo_reader
        self.segment_index = segment_index
        self.stream_index = stream_index
        self.additional_gain = additional_gain

    def get_num_samples(self):
        n = self.neo_reader.get_signal_size(block_index=0,
                        seg_index=self.segment_index,
                        stream_index=self.stream_index)
        return n

    def get_traces(self, start_frame, end_frame, channel_indices, return_scaled=True):
        # in neo rawio channel can acces by names/ids/indexes
        # there is no garranty that ids/names are unique on some formats
        raw_traces = self.neo_reader.get_analogsignal_chunk(
                            block_index=0,
                            seg_index=self.segment_index,
                            i_start=start_frame,
                            i_stop=end_frame,
                            stream_index=self.stream_index,
                            channel_indexes=channel_indices,
                        )
        
        if return_scaled:
            # rescale traces to natural units (can be anything)
            scaled_traces = self.neo_reader.rescale_signal_raw_to_float(raw_traces,
                                                                dtype='float32', stream_index=self.stream_index,
                                                                channel_indexes=channel_indices)
            # and then to uV
            if channel_indices is None:
                channel_indices = slice(None)
            scaled_traces *= self.additional_gain[:, channel_indices]
            return scaled_traces
        else:
            return raw_traces



class NeoBaseSortingExtractor(_NeoBaseExtractor, BaseSorting):
    
    # this will depend on each reader
    handle_raw_spike_directly = True
    
    def __init__(self, sampling_frequency=None, use_natural_unit_ids=False, **neo_kwargs):
        _NeoBaseExtractor.__init__(self,  **neo_kwargs)

        self.use_natural_unit_ids = use_natural_unit_ids
        
        if sampling_frequency is None:
            sampling_frequency = self._auto_guess_sampling_frequency()
        
        spike_channels = self.neo_reader.header['spike_channels']
        
        if use_natural_unit_ids:
            unit_ids = spike_channels['id']
            assert np.unique(unit_ids).size == unit_ids.size, 'unit_ids is have duplications'
        else:
            # use interger based unit_ids
            unit_ids = np.arange(spike_channels.size, dtype='int64')
        
        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        nseg = self.neo_reader.segment_count(block_index=0)
        for segment_index in range(nseg):
            if self.handle_raw_spike_directly:
                t_start = None
            else: 
                t_start = self.neo_reader.get_signal_t_start(0, segment_index)

            sorting_segment = NeoSortingSegment(self.neo_reader, segment_index,
                                self.use_natural_unit_ids, t_start, sampling_frequency)
            self.add_sorting_segment(sorting_segment)
            
    def _auto_guess_sampling_frequency(self):
        """
        Because neo handle spike in times (s or ms) but spikeinterface in frames related to signals.
        spikeinterface need so the sampling frequency.
        Getting the sampling rate in for psike is quite tricky because in neo
        spike are handle in s or ms
        internally many format do have have the spike time stamps
        at the same speed as the signal but at a higher clocks speed.
        here in spikeinterface we need spike index to be at the same speed
        that signal it do not make sens to have spikes at 50kHz sample
        when the sig is 10kHz.
        neo handle this but not spieinterface
        
        In neo spikes can have diffrents sampling rate than signals so conversion from
        signals frames to times is format dependent
        """

        # here the generic case
        #  all channels are in the same neo group so
        sig_channels = self.neo_reader.header['signal_channels']
        assert sig_channels.size > 0, 'samplinf_frequqency is not given and it is hard to guess it'
        sampling_frequency = np.max(sig_channels['sampling_rate'])
        
        # print('_auto_guess_sampling_frequency', sampling_frequency)
        return sampling_frequency


class NeoSortingSegment(BaseSortingSegment):
    def __init__(self,  neo_reader, segment_index, use_natural_unit_ids, t_start, sampling_freq):
        BaseSortingSegment.__init__(self)
        self.neo_reader = neo_reader
        self.segment_index = segment_index
        self.use_natural_unit_ids = use_natural_unit_ids
        self._t_start = t_start
        self._sampling_freq = sampling_freq
        
        self._natural_ids = None
    
    def get_natural_ids(self):
        if self._natural_ids is None:
            self._natural_ids = list(self._parent_extractor().neo_reader.header['spike_channels']['id'])
        return self._natural_ids
        
    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        if self.use_natural_unit_ids:
            unit_index =  self.get_natural_ids().index(unit_id)
        else:
            # already int
            unit_index = unit_id
        
        spike_timestamps = self.neo_reader.get_spike_timestamps(block_index=0,
                        seg_index=self.segment_index,
                        spike_channel_index=unit_index)
        
        if self._t_start is None:
            # because handle_raw_spike_directly=True
            spike_frames = spike_timestamps
        else:
            # convert to second second
            spike_times = self.neo_reader.rescale_spike_timestamp(spike_timestamps, dtype='float64')
            # convert to sample related to recording signals
            spike_frames = ((spike_times - self._t_start) * self._sampling_freq).astype('int64')

        # clip
        if start_frame is not None:
            spike_frames = spike_frames[spike_frames >= start_frame]

        if end_frame is not None:
            spike_frames = spike_frames[spike_frames <= end_frame]

        return spike_frames        

