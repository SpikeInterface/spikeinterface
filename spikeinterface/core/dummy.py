from probeinterface import ProbeGroup

from .baserecording import BaseRecording, BaseRecordingSegment
from .basesorting import BaseSorting, BaseSortingSegment
from .core_tools import define_function_from_class


class DummyRecording(BaseRecording):
    """
    Dummy recording class that enables to retrieve Recording metadata
    with the same interface as a BaseRecording object.

    The get_traces() method is not available.

    Parameters
    ----------
    recording: BaseRecording
        The recording to make "dummy"

    Returns
    -------
    BaseRecording
        The dummy recording object

    """

    def __init__(self, recording=None, channel_ids=None, num_segments=None,
                 sampling_frequency=None, dtype=None, probegroup=None):

        if recording is not None:
            channel_ids = recording.channel_ids
            sampling_frequency = recording.get_sampling_frequency()
            num_segments = recording.get_num_segments()
            dtype = recording.get_dtype()
            probegroup = recording.get_probegroup()
        else:
            assert channel_ids is not None
            assert sampling_frequency is not None
            assert num_segments is not None
            assert dtype is not None

        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)

        for segment_index in range(num_segments):
            segment = DummyRecordingSegment(
                sampling_frequency=sampling_frequency)
            self.add_recording_segment(segment)

        if probegroup is not None:
            if isinstance(probegroup, dict):
                probegroup = ProbeGroup.from_dict(probegroup)
            self.set_probegroup(probegroup, in_place=True)

        self._kwargs = dict(
            channel_ids=channel_ids, sampling_frequency=sampling_frequency,
            num_segments=num_segments, dtype=str(dtype),
            probegroup=probegroup.to_dict() if probegroup is not None else None
        )


class DummyRecordingSegment(BaseRecordingSegment):

    def get_num_samples(self) -> int:
        return 0


dummy_recording = define_function_from_class(DummyRecording, "dummy_recording")


class DummySorting(BaseSorting):
    """
    Dummy sorting class that enables to retrieve Sorting metadata
    with the same interface as a BaseSorting object.

    The get_unit_spike_train() method is not available.

    Parameters
    ----------
    sorting: BaseSorting
        The sorting to make "dummy"

    Returns
    -------
    BaseSorting
        The dummy sorting object  
    """

    def __init__(self, sorting=None, unit_ids=None, num_segments=None,
                 sampling_frequency=None):

        if sorting is not None:
            unit_ids = sorting.unit_ids
            sampling_frequency = sorting.get_sampling_frequency()
            num_segments = sorting.get_num_segments()
        else:
            assert unit_ids is not None
            assert sampling_frequency is not None
            assert num_segments is not None

        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        for segment_index in range(num_segments):
            segment = DummySortingSegment()
            self.add_sorting_segment(segment)

        self._kwargs = dict(
            unit_ids=unit_ids, sampling_frequency=sampling_frequency,
            num_segments=num_segments
        )


class DummySortingSegment(BaseSortingSegment):
    pass


dummy_sorting = define_function_from_class(DummySorting, "dummy_sorting")
