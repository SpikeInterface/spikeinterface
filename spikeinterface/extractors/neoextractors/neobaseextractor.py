from copy import copy
from typing import Union, List

import numpy as np

import neo

from spikeinterface.core import (BaseRecording, BaseSorting, BaseEvent,
                                 BaseRecordingSegment, BaseSortingSegment, BaseEventSegment)


def get_reader(raw_class, **neo_kwargs):
    neoIOclass = eval('neo.rawio.' + raw_class)
    neo_reader = neoIOclass(**neo_kwargs)
    neo_reader.parse_header()

    return neo_reader


class _NeoBaseExtractor:
    NeoRawIOClass = None


    def __init__(self, block_index, **neo_kwargs):
        self.neo_reader = get_reader(self.NeoRawIOClass, **neo_kwargs)

        if self.neo_reader.block_count() > 1 and block_index is None:
            raise Exception("This dataset is multi-block. Spikeinterface can load one block at a time. "
                            "Use 'block_index' to select the block to be loaded.")
        if block_index is None:
            block_index = 0
        self.block_index = block_index

    @classmethod
    def map_to_neo_kwargs(cls, *args, **kwargs):
        raise NotImplementedError


class NeoBaseRecordingExtractor(_NeoBaseExtractor, BaseRecording):
    def __init__(self, stream_id=None, stream_name=None, 
                 block_index=None, all_annotations=False, **neo_kwargs):

        _NeoBaseExtractor.__init__(self, block_index, **neo_kwargs)

        kwargs = dict(all_annotations=all_annotations)
        if block_index is not None:
            kwargs['block_index'] = block_index
        if stream_name is not None:
            kwargs['stream_name'] = stream_name
        if stream_id is not None:
            kwargs['stream_id'] = stream_id

        stream_channels = self.neo_reader.header['signal_streams']
        stream_names = list(stream_channels['name'])
        stream_ids = list(stream_channels['id'])

        if stream_id is None and stream_name is None:
            if stream_channels.size > 1:
                raise ValueError(f"This reader have several streams: \nNames: {stream_names}\nIDs: {stream_ids}. "
                                 f"Specify it with the 'stram_name' or 'stream_id' arguments")
            else:
                stream_id = stream_ids[0]
                stream_name = stream_names[0]
        else:
            assert stream_id or stream_name, "Pass either 'stream_id' or 'stream_name"
            if stream_id:
                assert stream_id in stream_ids, f'stream_id {stream_id} is not in {stream_ids}'
                stream_name = stream_names[stream_ids.index(stream_id)]
            if stream_name:
                assert stream_name in stream_names, f'stream_name {stream_name} is not in {stream_names}'
                stream_id = stream_ids[stream_names.index(stream_name)]

        self.stream_index = list(stream_ids).index(stream_id)
        self.stream_id = stream_id
        self.stream_name = stream_name

        # need neo 0.10.0
        signal_channels = self.neo_reader.header['signal_channels']
        mask = signal_channels['stream_id'] == stream_id
        signal_channels = signal_channels[mask]

        # check channel groups
        chan_ids = signal_channels['id']

        sampling_frequency = self.neo_reader.get_signal_sampling_rate(stream_index=self.stream_index)
        dtype = signal_channels['dtype'][0]
        BaseRecording.__init__(self, sampling_frequency, chan_ids, dtype)
        self.extra_requirements.append('neo')

        # find the gain to uV
        gains = signal_channels['gain']
        offsets = signal_channels['offset']

        units = signal_channels['units']

        # mark that units are V, mV or uV
        self.has_non_standard_units = False
        if not np.all(np.isin(units, ['V', 'Volt', 'mV', 'uV'])):
            self.has_non_standard_units = True

        additional_gain = np.ones(units.size, dtype='float')
        additional_gain[units == 'V'] = 1e6
        additional_gain[units == 'Volt'] = 1e6
        additional_gain[units == 'mV'] = 1e3
        additional_gain[units == 'uV'] = 1.
        additional_gain = additional_gain

        final_gains = gains * additional_gain
        final_offsets = offsets * additional_gain

        self.set_property('gain_to_uV', final_gains)
        self.set_property('offset_to_uV', final_offsets)
        self.set_property('channel_name', signal_channels["name"])

        if all_annotations:
            block_ann = self.neo_reader.raw_annotations['blocks'][self.block_index]
            # in neo annotation are for every segment!
            # Here we take only the first segment to annotate the object
            # Generally annotation for multi segment are duplicated
            seg_ann = block_ann['segments'][0]
            sig_ann = seg_ann['signals'][self.stream_index]

            # scalar annotations
            for k, v in sig_ann.items():
                if not k.startswith('__'):
                    self.annotate(k=v)
            # vector array_annotations are channel properties
            for k, values in sig_ann['__array_annotations__'].items():
                self.set_property(k, values)

        nseg = self.neo_reader.segment_count(block_index=self.block_index)
        for segment_index in range(nseg):
            rec_segment = NeoRecordingSegment(self.neo_reader, self.block_index, 
                                              segment_index, self.stream_index)
            self.add_recording_segment(rec_segment)

        self._kwargs.update(kwargs)


    @classmethod
    def get_streams(cls, *args, **kwargs):
        neo_kwargs = cls.map_to_neo_kwargs(*args, **kwargs)
        neo_reader = get_reader(cls.NeoRawIOClass, **neo_kwargs)

        stream_channels = neo_reader.header['signal_streams']
        stream_names = list(stream_channels['name'])
        stream_ids = list(stream_channels['id'])
        return stream_names, stream_ids


    @classmethod
    def get_num_blocks(cls, *args, **kwargs):
        neo_kwargs = cls.map_to_neo_kwargs(*args, **kwargs)
        neo_reader = get_reader(cls.NeoRawIOClass, **neo_kwargs)
        return neo_reader.block_count()


class NeoRecordingSegment(BaseRecordingSegment):
    def __init__(self, neo_reader, block_index, segment_index, stream_index):
        sampling_frequency = neo_reader.get_signal_sampling_rate(stream_index=stream_index)
        t_start = neo_reader.get_signal_t_start(block_index, segment_index, 
                                                stream_index=stream_index)
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency, t_start=t_start)
        self.neo_reader = neo_reader
        self.segment_index = segment_index
        self.stream_index = stream_index
        self.block_index = block_index

    def get_num_samples(self):
        n = self.neo_reader.get_signal_size(block_index=self.block_index,
                                            seg_index=self.segment_index,
                                            stream_index=self.stream_index)
        return n

    def get_traces(self,
                   start_frame: Union[int, None] = None,
                   end_frame: Union[int, None] = None,
                   channel_indices: Union[List, None] = None,
                   ) -> np.ndarray:
        raw_traces = self.neo_reader.get_analogsignal_chunk(
            block_index=self.block_index,
            seg_index=self.segment_index,
            i_start=start_frame,
            i_stop=end_frame,
            stream_index=self.stream_index,
            channel_indexes=channel_indices,
        )
        return raw_traces


class NeoBaseSortingExtractor(_NeoBaseExtractor, BaseSorting):
    # this will depend on each reader
    handle_spike_frame_directly = True

    def __init__(self, sampling_frequency=None, use_natural_unit_ids=False, 
                 block_index=None, **neo_kwargs):
        _NeoBaseExtractor.__init__(self, block_index, **neo_kwargs)

        self.use_natural_unit_ids = use_natural_unit_ids
        if sampling_frequency is None:
            sampling_frequency, stream_id = self._auto_guess_sampling_frequency()
        
        # Get the stream index corresponding to the extracted frequency
        stream_index = None
        if stream_id is not None:
            stream_index = np.where(self.neo_reader.header["signal_streams"]["id"] == stream_id)[0][0]


        spike_channels = self.neo_reader.header['spike_channels']

        if use_natural_unit_ids:
            unit_ids = spike_channels['id']
            assert np.unique(unit_ids).size == unit_ids.size, 'unit_ids is have duplications'
        else:
            # use interger based unit_ids
            unit_ids = np.arange(spike_channels.size, dtype='int64')

        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        nseg = self.neo_reader.segment_count(block_index=self.block_index)
        for segment_index in range(nseg):
            if self.handle_spike_frame_directly:
                t_start = None
            else:
                t_start = self.neo_reader.get_signal_t_start(block_index=self.block_index, 
                                                            seg_index=segment_index, 
                                                            stream_index=stream_index
                                                            )



            sorting_segment = NeoSortingSegment(self.neo_reader, self.block_index, segment_index,
                                                self.use_natural_unit_ids, t_start, 
                                                sampling_frequency)
            self.add_sorting_segment(sorting_segment)

    def _auto_guess_sampling_frequency(self):
        """
        When the signal channels are available the sampling rate is set that of the channel with the higher frequency
        and the corresponding stream_id from which the frequency was extracted.
        
        Because neo handle spike in times (s or ms) but spikeinterface in frames related to signals, spikeinterface 
        need the sampling frequency. 
        
        Internally many format do have the spike timestamps at the same speed as the signal but at a higher
        clocks speed. Here in spikeinterface we need spike index to be at the same speed that signal, therefore it does
        not make sense to have spikes at 50kHz when the signal is 10kHz. Neo handle this but not spikeinterface.

        In neo spikes can have diffrents sampling rate than signals so conversion from signals frames to times is
        format dependent.
        """

        # here the generic case
        #  all channels are in the same neo group so
        sig_channels = self.neo_reader.header['signal_channels']
        assert sig_channels.size > 0, 'sampling_frequency could not be inferred from the signals, set it manually'
        argmax = np.argmax(sig_channels['sampling_rate'])
        sampling_frequency = sig_channels[argmax]["sampling_rate"]
        stream_id = sig_channels[argmax]["stream_id"]
        return sampling_frequency, stream_id


class NeoSortingSegment(BaseSortingSegment):
    def __init__(self, neo_reader, block_index, segment_index, use_natural_unit_ids, t_start, 
                 sampling_frequency):
        BaseSortingSegment.__init__(self)
        self.neo_reader = neo_reader
        self.segment_index = segment_index
        self.block_index = block_index
        self.use_natural_unit_ids = use_natural_unit_ids
        self._t_start = t_start
        self._sampling_frequency = sampling_frequency

        self._natural_ids = None

    def get_natural_ids(self):
        if self._natural_ids is None:
            self._natural_ids = list(self._parent_extractor().neo_reader.header['spike_channels']['id'])
        return self._natural_ids

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        if self.use_natural_unit_ids:
            unit_index = self.get_natural_ids().index(unit_id)
        else:
            # already int
            unit_index = unit_id

        spike_timestamps = self.neo_reader.get_spike_timestamps(block_index=self.block_index,
                                                                seg_index=self.segment_index,
                                                                spike_channel_index=unit_index)
        if self._t_start is None:
            # When handle_spike_frame_directly=True, the extractors handles timestamps as frames
            spike_frames = spike_timestamps
        else:
            # Convert timestamps to seconds
            spike_times = self.neo_reader.rescale_spike_timestamp(spike_timestamps, dtype='float64')
            # Re-center to zero for each segment and multiply by frequency to convert seconds to frames
            spike_frames = ((spike_times - self._t_start) * self._sampling_frequency).astype('int64')

        # clip
        if start_frame is not None:
            spike_frames = spike_frames[spike_frames >= start_frame]

        if end_frame is not None:
            spike_frames = spike_frames[spike_frames <= end_frame]

        return spike_frames


class NeoBaseEventExtractor(_NeoBaseExtractor, BaseEvent):
    handle_event_frame_directly = False

    def __init__(self, block_index=None, **neo_kwargs):
        _NeoBaseExtractor.__init__(self, block_index, **neo_kwargs)

        # TODO load feature from neo array_annotations

        event_channels = self.neo_reader.header['event_channels']

        channel_ids = event_channels['id']

        BaseEvent.__init__(self, channel_ids, structured_dtype=False)

        nseg = self.neo_reader.segment_count(block_index=0)
        for segment_index in range(nseg):
            if self.handle_event_frame_directly:
                t_start = None
            else:
                t_start = self.neo_reader.get_signal_t_start(self.block_index, segment_index)

            event_segment = NeoEventSegment(self.neo_reader, self.block_index, segment_index,
                                            t_start)
            self.add_event_segment(event_segment)


class NeoEventSegment(BaseEventSegment):
    def __init__(self, neo_reader, block_index, segment_index, t_start):
        BaseEventSegment.__init__(self)
        self.neo_reader = neo_reader
        self.segment_index = segment_index
        self.block_index = block_index
        self._t_start = t_start
        self._natural_ids = None

    def get_event_times(self, channel_id, start_frame, end_frame):
        channel_index = list(self.neo_reader.header['event_channels']['id']).index(channel_id)

        event_timestamps, event_duration, event_labels = self.neo_reader.get_event_timestamps(
            block_index=self.block_index,
            seg_index=self.segment_index,
            event_channel_index=channel_index)

        event_times = self.neo_reader.rescale_event_timestamp(event_timestamps,
                                                              dtype='float64', event_channel_index=channel_index)

        return event_times
