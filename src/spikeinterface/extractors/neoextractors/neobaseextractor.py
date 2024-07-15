from __future__ import annotations

from typing import Optional, Union, Dict, Any, List, Tuple
import warnings
from math import isclose

import numpy as np
import importlib

from spikeinterface.core import (
    BaseRecording,
    BaseSorting,
    BaseEvent,
    BaseRecordingSegment,
    BaseSortingSegment,
    BaseEventSegment,
)


class _NeoBaseExtractor:
    NeoRawIOClass = None

    def __init__(self, block_index, **neo_kwargs):

        # Avoids double initiation of the neo reader if it was already done in the __init__ of the child class
        if not hasattr(self, "neo_reader"):
            self.neo_reader = self.get_neo_io_reader(self.NeoRawIOClass, **neo_kwargs)

        if self.neo_reader.block_count() > 1 and block_index is None:
            raise Exception(
                "This dataset is multi-block. Spikeinterface can load one block at a time. "
                "Use 'block_index' to select the block to be loaded."
            )
        if block_index is None:
            block_index = 0
        self.block_index = block_index

    @classmethod
    def map_to_neo_kwargs(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_neo_io_reader(cls, raw_class: str, **neo_kwargs):
        """
        Dynamically creates an instance of a NEO IO reader class using the specified class name and keyword arguments.

        Note that the function parses the header which makes all the information available to the extractor.

        Parameters
        ----------
        raw_class : str
            The name of the NEO IO reader class to create an instance of.
        **neo_kwargs : dict
            Initialization keyword arguments to be passed to the NEO IO reader class constructor.

        Returns
        -------
        BaseRawIO
            An instance of the specified NEO IO reader class

        """

        rawio_module = importlib.import_module("neo.rawio")
        neoIOclass = getattr(rawio_module, raw_class)
        neo_reader = neoIOclass(**neo_kwargs)
        neo_reader.parse_header()

        return neo_reader

    @classmethod
    def get_streams(cls, *args, **kwargs):
        neo_kwargs = cls.map_to_neo_kwargs(*args, **kwargs)
        neo_reader = cls.get_neo_io_reader(cls.NeoRawIOClass, **neo_kwargs)

        stream_channels = neo_reader.header["signal_streams"]
        stream_names = list(stream_channels["name"])
        stream_ids = list(stream_channels["id"])
        return stream_names, stream_ids

    def build_stream_id_to_sampling_frequency_dict(self) -> Dict[str, float]:
        """
        Build a mapping from stream_id to sampling frequencies.

        This function creates a dictionary mapping each stream_id to its corresponding sampling
        frequency, as extracted from the signal channels in the Neo header.

        Returns
        -------
        dict of {str: float}
            Dictionary mapping stream_ids to their corresponding sampling frequencies.

        Raises
        ------
        AssertionError
            If there are no signal streams available from which to extract the sampling frequencies.
        """
        neo_header = self.neo_reader.header
        signal_channels = neo_header["signal_channels"]
        assert signal_channels.size > 0, "No signal streams to infer the sampling frequency. Set it manually"

        # Get unique pairs of channel_stream_id and channel_sampling_frequencies
        channel_sampling_frequencies = signal_channels["sampling_rate"]
        channel_stream_id = signal_channels["stream_id"]
        unique_pairs = np.unique(np.vstack((channel_stream_id, channel_sampling_frequencies)).T, axis=0)

        # Form a dictionary of stream_id to sampling_frequency
        stream_to_sampling_frequencies = {}
        for stream_id, sampling_frequency in unique_pairs:
            stream_to_sampling_frequencies[stream_id] = float(sampling_frequency)

        return stream_to_sampling_frequencies

    def build_stream_id_to_t_start_dict(self, segment_index: int) -> Dict[str, float]:
        """
        Builds a dictionary mapping stream IDs to their respective t_start values for a given segment.

        Parameters
        ----------
        segment_index : int
            The index of the segment for which to build the dictionary.

        Returns
        -------
        Dict[str, float]
            A dictionary where keys are the stream IDs and the values are their respective t_start values.

        """
        neo_header = self.neo_reader.header
        signal_streams = neo_header["signal_streams"]
        stream_ids = signal_streams["id"]

        stream_id_to_t_start = dict()
        for stream_id in stream_ids:
            stream_index = (signal_streams["id"] == stream_id).nonzero()[0][0]
            t_start = self.neo_reader.get_signal_t_start(
                block_index=self.block_index,
                seg_index=segment_index,
                stream_index=stream_index,
            )
            stream_id_to_t_start[stream_id] = t_start

        return stream_id_to_t_start

    def build_stream_name_to_stream_id_dict(self) -> Dict[str, str]:
        neo_header = self.neo_reader.header
        signal_streams = neo_header["signal_streams"]
        stream_ids = signal_streams["id"]
        stream_names = signal_streams["name"]

        stream_name_to_stream_id = dict()
        for stream_name, stream_id in zip(stream_names, stream_ids):
            stream_name_to_stream_id[stream_name] = stream_id

        return stream_name_to_stream_id


class NeoBaseRecordingExtractor(_NeoBaseExtractor, BaseRecording):
    def __init__(
        self,
        stream_id: Optional[str] = None,
        stream_name: Optional[str] = None,
        block_index: Optional[int] = None,
        all_annotations: bool = False,
        use_names_as_ids: Optional[bool] = None,
        **neo_kwargs: Dict[str, Any],
    ) -> None:
        """
        Initialize a NeoBaseRecordingExtractor instance.

        Parameters
        ----------
        stream_id : Optional[str], default: None
            The ID of the stream to extract from the data.
        stream_name : Optional[str], default: None
            The name of the stream to extract from the data.
        block_index : Optional[int], default: None
            The index of the block to extract from the data.
        all_annotations : bool, default: False
            If True, include all annotations in the extracted data.
        use_names_as_ids : Optional[bool], default: None
            If True, use channel names as IDs. Otherwise, use default IDs.
            In NEO the ids are guaranteed to be unique. Names are user defined and can be duplicated.
        neo_kwargs : Dict[str, Any]
            Additional keyword arguments to pass to the NeoBaseExtractor for initialization.

        """

        _NeoBaseExtractor.__init__(self, block_index, **neo_kwargs)

        kwargs = dict(all_annotations=all_annotations)
        if block_index is not None:
            kwargs["block_index"] = block_index
        if stream_name is not None:
            kwargs["stream_name"] = stream_name
        if stream_id is not None:
            kwargs["stream_id"] = stream_id
        if use_names_as_ids is not None:
            kwargs["use_names_as_ids"] = use_names_as_ids
        else:
            use_names_as_ids = False

        stream_channels = self.neo_reader.header["signal_streams"]
        stream_names = list(stream_channels["name"])
        stream_ids = list(stream_channels["id"])

        if stream_id is None and stream_name is None:
            if stream_channels.size > 1:
                raise ValueError(
                    f"This reader have several streams: \nNames: {stream_names}\nIDs: {stream_ids}. \n"
                    f"Specify it from the options above with the 'stream_name' or 'stream_id' arguments"
                )
            else:
                stream_id = stream_ids[0]
                stream_name = stream_names[0]
        else:
            assert stream_id or stream_name, "Pass either 'stream_id' or 'stream_name"
            if stream_id:
                assert stream_id in stream_ids, f"stream_id {stream_id} is not in {stream_ids}"
                stream_name = stream_names[stream_ids.index(stream_id)]
            if stream_name:
                assert stream_name in stream_names, f"stream_name {stream_name} is not in {stream_names}"
                stream_id = stream_ids[stream_names.index(stream_name)]

        self.stream_index = list(stream_ids).index(stream_id)
        self.stream_id = stream_id
        self.stream_name = stream_name

        # need neo 0.10.0
        signal_channels = self.neo_reader.header["signal_channels"]
        mask = signal_channels["stream_id"] == stream_id
        signal_channels = signal_channels[mask]

        if use_names_as_ids:
            chan_names = signal_channels["name"]
            assert (
                chan_names.size == np.unique(chan_names).size
            ), "use_names_as_ids=True is not possible, channel names are not unique"
            chan_ids = chan_names
        else:
            # unique in all cases
            chan_ids = signal_channels["id"]

        sampling_frequency = self.neo_reader.get_signal_sampling_rate(stream_index=self.stream_index)
        dtype = np.dtype(signal_channels["dtype"][0])
        BaseRecording.__init__(self, sampling_frequency, chan_ids, dtype)
        self.extra_requirements.append("neo")

        # find the gain to uV
        gains = signal_channels["gain"]
        offsets = signal_channels["offset"]

        if dtype.kind == "i" and np.all(gains < 0) and np.all(offsets == 0):
            # special hack when all channel have negative gain: we put back the gain positive
            # this help the end user experience
            self.inverted_gain = True
            gains = -gains
        else:
            self.inverted_gain = False

        units = signal_channels["units"]

        # mark that units are V, mV or uV
        self.has_non_standard_units = False
        if not np.all(np.isin(units, ["V", "Volt", "mV", "uV"])):
            self.has_non_standard_units = True

        additional_gain = np.ones(units.size, dtype="float")
        additional_gain[units == "V"] = 1e6
        additional_gain[units == "Volt"] = 1e6
        additional_gain[units == "mV"] = 1e3
        additional_gain[units == "uV"] = 1.0
        additional_gain = additional_gain

        final_gains = gains * additional_gain
        final_offsets = offsets * additional_gain

        self.set_property("gain_to_uV", final_gains)
        self.set_property("offset_to_uV", final_offsets)
        if not use_names_as_ids:
            self.set_property("channel_names", signal_channels["name"])

        if all_annotations:
            block_ann = self.neo_reader.raw_annotations["blocks"][self.block_index]
            # in neo annotation are for every segment!
            # Here we take only the first segment to annotate the object
            # Generally annotation for multi segment are duplicated
            seg_ann = block_ann["segments"][0]
            sig_ann = seg_ann["signals"][self.stream_index]

            scalar_annotations = {name: value for name, value in sig_ann.items() if not name.startswith("__")}

            # name in neo corresponds to stream name
            # We don't propagate the name as an annotation because that has a differnt meaning on spikeinterface
            stream_name = scalar_annotations.pop("name", None)
            if stream_name:
                self.set_annotation(annotation_key="stream_name", value=stream_name)
            for annotation_key, value in scalar_annotations.items():
                self.set_annotation(annotation_key=annotation_key, value=value)

            array_annotations = sig_ann["__array_annotations__"]
            # We do not add this because is confusing for the user to have this repeated
            array_annotations.pop("channel_ids", None)
            # This is duplicated when using channel_names as ids
            if use_names_as_ids:
                array_annotations.pop("channel_names", None)

            # vector array_annotations are channel properties
            for key, values in array_annotations.items():
                self.set_property(key=key, values=values)

        nseg = self.neo_reader.segment_count(block_index=self.block_index)
        for segment_index in range(nseg):
            rec_segment = NeoRecordingSegment(
                self.neo_reader, self.block_index, segment_index, self.stream_index, self.inverted_gain
            )
            self.add_recording_segment(rec_segment)

        self._kwargs.update(kwargs)

    @classmethod
    def get_num_blocks(cls, *args, **kwargs):
        neo_kwargs = cls.map_to_neo_kwargs(*args, **kwargs)
        neo_reader = cls.get_neo_io_reader(cls.NeoRawIOClass, **neo_kwargs)
        return neo_reader.block_count()


class NeoRecordingSegment(BaseRecordingSegment):
    def __init__(self, neo_reader, block_index, segment_index, stream_index, inverted_gain):
        sampling_frequency = neo_reader.get_signal_sampling_rate(stream_index=stream_index)
        t_start = neo_reader.get_signal_t_start(block_index, segment_index, stream_index=stream_index)
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency, t_start=t_start)
        self.neo_reader = neo_reader
        self.segment_index = segment_index
        self.stream_index = stream_index
        self.block_index = block_index
        self.inverted_gain = inverted_gain

    def get_num_samples(self):
        num_samples = self.neo_reader.get_signal_size(
            block_index=self.block_index, seg_index=self.segment_index, stream_index=self.stream_index
        )

        return int(num_samples)

    def get_traces(
        self,
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
        if self.inverted_gain:
            raw_traces = -raw_traces
        return raw_traces


class NeoBaseSortingExtractor(_NeoBaseExtractor, BaseSorting):
    neo_returns_frames = True
    # `neo_returns_frames` is a class attribute indicating whether
    # `neo_reader.get_spike_timestamps` returns frames instead of timestamps (!),
    # If False, then the segments need to transform timestamps to to frames.
    # For formats that return timestamps (e.g. Mearec, Blackrock, Neuralynx) this should be set to
    # False in the format class that inherits from this.

    need_t_start_from_signal_stream = False
    # `need_t_start_from_signal_stream` is a class attribute indicating whether t_start should be inferred
    # from `neo_reader.get_signal_t_start`. If True, then we try to infer t_start from the signal stream with
    # corresponding sampling frequency. If False, then t_start is set to None which is 0 for practical purposes.
    # Neuralynx is an example of a format that needs t_start to be inferred from the signal stream.

    def __init__(
        self,
        block_index=None,
        sampling_frequency=None,
        use_format_ids=False,
        stream_id: Optional[str] = None,
        stream_name: Optional[str] = None,
        **neo_kwargs,
    ):
        _NeoBaseExtractor.__init__(self, block_index, **neo_kwargs)

        # Get stream_id from stream_name
        if stream_name:
            stream_id = self.build_stream_name_to_stream_id_dict()[stream_name]

        spike_channels = self.neo_reader.header["spike_channels"]
        if use_format_ids:
            unit_ids = spike_channels["id"]
            assert np.unique(unit_ids).size == unit_ids.size, "unit_ids is have duplications"
        else:
            # use interger based unit_ids
            unit_ids = np.arange(spike_channels.size, dtype="int32")

        if sampling_frequency is None:
            sampling_frequency = self._infer_sampling_frequency_from_analog_signal(stream_id=stream_id)

        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        num_segments = self.neo_reader.segment_count(block_index=self.block_index)
        for segment_index in range(num_segments):
            t_start = None  # This means t_start will be 0 for practical purposes
            if self.need_t_start_from_signal_stream:
                t_start = self._infer_t_start_from_signal_stream(segment_index=segment_index, stream_id=stream_id)

            sorting_segment = NeoSortingSegment(
                neo_reader=self.neo_reader,
                block_index=self.block_index,
                segment_index=segment_index,
                t_start=t_start,
                sampling_frequency=sampling_frequency,
                neo_returns_frames=self.neo_returns_frames,
            )

            self.add_sorting_segment(sorting_segment)

    def _infer_sampling_frequency_from_analog_signal(self, stream_id: Optional[str] = None) -> float:
        """
        Infer the sampling frequency from available signal channels.

        The function attempts to infer the sampling frequency by examining available signal
        channels. If there is only one unique sampling frequency available across all channels,
        that frequency is used. If streams have different sampling frequencies, a ValueError
        is raised and the user is instructed to manually specify the sampling frequency when
        initializing the sorting extractor.

        Parameters
        ----------
        stream_id : str, default: None
            The ID of the stream from which to infer the sampling frequency. If not provided,
            the function will look for a common sampling frequency across all streams

        Returns
        -------
        float
            The inferred sampling frequency.

        Raises
        ------
        ValueError
            If streams have different sampling frequencies.
        AssertionError
            If the provided stream_id is not found in the sorter.

        Warnings
        --------
        UserWarning
            If the function has to guess the sampling frequency.

        Notes
        -----
        Neo handles spikes in either seconds or milliseconds. However, SpikeInterface deals
        with frames related to signals so we need the sampling frequency to convert from
        seconds to frames.

        Most internal formats have spike timestamps at the same speed as the signal but at
        a higher clock speed. Yet, in SpikeInterface, spike indexes need to be at the same
        speed as the signal. Therefore, it is not logical to have spikes at 50kHz when the
        signal is 10kHz. Neo can handle this discrepancy, but SpikeInterface cannot.

        Another key point is that in Neo, spikes can have different sampling rates than signals,
        hence the conversion from signal frames to times is format dependent.
        """

        stream_id_to_sampling_frequencies = self.build_stream_id_to_sampling_frequency_dict()

        # If user provided a stream_id, use the sampling frequency of that stream
        if stream_id:
            assertion_msg = (
                "stream_id not found in sorter, the stream_id must be one of the following keys: \n"
                f"stream_id_to_sampling_frequency = {stream_id_to_sampling_frequencies}"
            )
            assert stream_id in stream_id_to_sampling_frequencies, assertion_msg
            sampling_frequency = stream_id_to_sampling_frequencies[stream_id]
            return sampling_frequency

        available_sampling_frequencies = list(stream_id_to_sampling_frequencies.values())
        unique_sampling_frequencies = set(available_sampling_frequencies)

        # If there is only one stream or multiple one with the same sampling frequency use that one
        if len(unique_sampling_frequencies) == 1:
            sampling_frequency = available_sampling_frequencies[0]
            warning_about_inference = (
                "SpikeInterface will use the following sampling frequency: \n"
                f"sampling_frequency = {sampling_frequency} \n"
                "Corresponding to the following stream_id: \n"
                f"stream_id = {stream_id} \n"
                "To avoid this warning pass explicitly the sampling frequency or the stream_id "
                "when initializing the sorting extractor. \n"
                "The following stream_ids with corresponding sampling frequencies were found: \n"
                f"stream_id_to_sampling_frequencies = {stream_id_to_sampling_frequencies} \n"
            )
            warnings.warn(warning_about_inference)
        else:
            instructions_for_user = (
                "Multiple streams ids with different sampling frequencies found in the file: \n"
                f"{stream_id_to_sampling_frequencies} \n"
                f"Please specify one of the sampling frequencies above "
                "when initializing the sorting extractor with the sampling frequency parameter."
            )
            raise ValueError(instructions_for_user)

        return sampling_frequency

    def _infer_t_start_from_signal_stream(self, segment_index: int, stream_id: Optional[str] = None) -> float | None:
        """
        Infers the t_start value from the signal stream, either using the provided stream ID or by finding
        streams with matching frequency to the sorter's sampling frequency.

        If multiple streams with matching frequency exist and have different t_starts, the t_start is set to None.
        If no matching streams are found, the t_start is also set to None.

        Parameters
        ----------
        segment_index : int
            The index of the segment in which to look for the stream.
        stream_id : str, default: None
            The ID of the stream from which to infer t_start. If not provided,
            the function will look for streams with a matching sampling frequency.

        Returns
        -------
        float | None
            The inferred t_start value, or None if it couldn't be inferred.

        Raises
        ------
        AssertionError
            If the provided stream_id is not found in the sorter.

        Warnings
        --------
        UserWarning
            If no streams with matching frequency are found, or if multiple matching streams are found.

        """
        stream_id_to_t_start = self.build_stream_id_to_t_start_dict(segment_index=segment_index)
        if stream_id:
            assertion_msg = (
                "stream_id not found in sorter, the stream_id must be one of the following keys: \n"
                f"stream_id_to_t_start = {stream_id_to_t_start}"
            )
            assert stream_id in stream_id_to_t_start, assertion_msg
            return stream_id_to_t_start[stream_id]

        # Otherwise see if there are any stream ids with matching frequency
        sorter_sampling_frequency = self._sampling_frequency
        stream_id_to_sampling_frequencies = self.build_stream_id_to_sampling_frequency_dict()
        stream_ids_with_matching_frequency = [
            stream_id
            for stream_id, stream_sampling_frequency in stream_id_to_sampling_frequencies.items()
            if isclose(stream_sampling_frequency, sorter_sampling_frequency, rel_tol=1e-9)
        ]

        matching_t_starts = [stream_id_to_t_start[stream_id] for stream_id in stream_ids_with_matching_frequency]
        unique_t_starts = set(matching_t_starts)

        if len(unique_t_starts) == 0:
            t_start = None
            warning_message = (
                "No stream ids with corresponding sampling frequency found \n " "Setting t_start to None. \n"
            )
            warnings.warn(warning_message)

        if len(unique_t_starts) == 1:
            t_start = matching_t_starts[0]
            warning_about_inference = (
                "SpikeInterface will use the following t_start: \n"
                f"t_start = {t_start} \n"
                "Corresponding to the following stream_id: \n"
                f"stream_id = {stream_id} \n"
                "To avoid this warning pass explicitly the stream_id "
                "when initializing the sorting extractor. \n"
                "The following stream_ids with corresponding t_starts were found: \n"
                f"{stream_id_to_t_start} \n"
            )
            warnings.warn(warning_about_inference)
        else:
            t_start = None
            warning_message = (
                "Multiple streams ids with corresponding sampling frequency found \n "
                "Each stream has the correspoding t_start: \n"
                f"{stream_id_to_t_start} \n"
                f"Setting t_start to None. \n"
            )
            warnings.warn(warning_message)

        return t_start


class NeoSortingSegment(BaseSortingSegment):
    def __init__(
        self,
        neo_reader,
        block_index,
        segment_index,
        t_start,
        sampling_frequency,
        neo_returns_frames,
    ):
        BaseSortingSegment.__init__(self)
        self.neo_reader = neo_reader
        self.segment_index = segment_index
        self.block_index = block_index
        self._t_start = t_start
        self._sampling_frequency = sampling_frequency
        self.neo_returns_frames = neo_returns_frames

    def map_from_unit_id_to_spike_channel_index(self, unit_id):
        unit_ids_list = list(self.parent_extractor.get_unit_ids())

        return unit_ids_list.index(unit_id)

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        spike_channel_index = self.map_from_unit_id_to_spike_channel_index(unit_id)
        spike_timestamps = self.neo_reader.get_spike_timestamps(
            block_index=self.block_index,
            seg_index=self.segment_index,
            spike_channel_index=spike_channel_index,
        )

        if self.neo_returns_frames:
            spike_frames = spike_timestamps
        else:
            spike_times_seconds = self.neo_reader.rescale_spike_timestamp(spike_timestamps, dtype="float64")
            # Re-center to zero for each segment and multiply by frequency to convert seconds to frames
            t_start = 0 if self._t_start is None else self._t_start
            spike_frames = ((spike_times_seconds - t_start) * self._sampling_frequency).astype("int64")

        # clip
        if start_frame is not None:
            spike_frames = spike_frames[spike_frames >= start_frame]

        if end_frame is not None:
            spike_frames = spike_frames[spike_frames <= end_frame]

        return spike_frames


_neo_event_dtype = np.dtype([("time", "float64"), ("duration", "float64"), ("label", "<U100")])


class NeoBaseEventExtractor(_NeoBaseExtractor, BaseEvent):
    handle_event_frame_directly = False

    def __init__(self, block_index=None, **neo_kwargs):
        _NeoBaseExtractor.__init__(self, block_index, **neo_kwargs)

        # TODO load feature from neo array_annotations

        event_channels = self.neo_reader.header["event_channels"]

        channel_ids = event_channels["id"]

        BaseEvent.__init__(self, channel_ids, structured_dtype=_neo_event_dtype)

        block_index = block_index if block_index is not None else 0
        nseg = self.neo_reader.segment_count(block_index=block_index)
        for segment_index in range(nseg):
            if self.handle_event_frame_directly:
                t_start = None
            else:
                t_start = self.neo_reader.get_signal_t_start(self.block_index, segment_index, stream_index=0)

            event_segment = NeoEventSegment(self.neo_reader, self.block_index, segment_index, t_start)
            self.add_event_segment(event_segment)


class NeoEventSegment(BaseEventSegment):
    def __init__(self, neo_reader, block_index, segment_index, t_start):
        BaseEventSegment.__init__(self)
        self.neo_reader = neo_reader
        self.segment_index = segment_index
        self.block_index = block_index
        self._t_start = t_start
        self._natural_ids = None

    def get_events(self, channel_id, start_time, end_time):
        channel_index = list(self.neo_reader.header["event_channels"]["id"]).index(channel_id)

        event_timestamps, event_duration, event_labels = self.neo_reader.get_event_timestamps(
            block_index=self.block_index, seg_index=self.segment_index, event_channel_index=channel_index
        )

        event_times = self.neo_reader.rescale_event_timestamp(
            event_timestamps, dtype="float64", event_channel_index=channel_index
        )

        event = np.zeros(len(event_times), dtype=_neo_event_dtype)
        event["time"] = event_times
        event["duration"] = event_duration
        event["label"] = event_labels

        if start_time is not None:
            event = event[event["time"] >= start_time]
        if end_time is not None:
            event = event[event["time"] < end_time]
        return event
