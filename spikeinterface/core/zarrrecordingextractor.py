from typing import List, Union

import zarr
from pathlib import Path
from probeinterface import Probe

import numpy as np

from spike_sorting.phy.phy.phy.cluster.tests.test_supervisor import data

from .baserecording import BaseRecording, BaseRecordingSegment
from .core_tools import read_binary_recording, write_binary_recording
from .job_tools import _shared_job_kwargs_doc


class ZarrRecordingExtractor(BaseRecording):
    """
    RecordingExtractor for a zarr format

    Parameters
    ----------
    root_path: str or Path
        Path to the zarr root file
    sampling_frequency: float
        The sampling frequency
    t_starts: None or list of float
        Times in seconds of the first sample for each segment
    channel_ids: list (optional)
        A list of channel ids
    gain_to_uV: float or array-like (optional)
        The gain to apply to the traces
    offset_to_uV: float or array-like
        The offset to apply to the traces
    is_filtered: bool or None
        If True, the recording is assumed to be filtered. If None, is_filtered is not set.

    Returns
    -------
    recording: ZarrRecordingExtractor
        The recording Extractor
    """
    extractor_name = 'ZarrRecordingExtractor'
    has_default_locations = False
    installed = True  # check at class level if installed or not
    is_writable = True
    mode = 'file'
    installation_mesg = ""  # error message when not installed

    def __init__(self, root_path):
        root_path = Path(root_path)
        self._root = zarr.open(str(root_path), mode="r")
        sampling_frequency = self._root.attrs.get("sampling_frequency", None)
        channel_ids = self._root.attrs.get("channel_ids", None)
        num_segments = self._root.attrs.get("num_segments", None)

        assert sampling_frequency is not None
        assert channel_ids is not None
        assert num_segments is not None

        channel_ids = np.array(channel_ids)

        dtype = self._root['traces_seg0'].dtype

        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)

        dtype = np.dtype(dtype)
        t_starts = self._root.get("t_starts", None)

        for segment_index in range(num_segments):
            trace_name = f"traces_seg{segment_index}"
            assert len(channel_ids) == self._root[trace_name].shape[1], \
                f'Segment {segment_index} has the wrong number of channels!'

            time_kwargs = {}
            time_vector = self._root.get(f"times_seg{segment_index}", None)
            if time_vector is not None:
                time_kwargs["time_vector"] = time_vector
            else:
                if t_starts is None:
                    t_start = None
                else:
                    t_start = t_starts[segment_index]
                    if np.isnan(t_start):
                        t_start = None
                time_kwargs["t_start"] = t_start
                time_kwargs["sampling_frequency"] = sampling_frequency

            rec_segment = ZarrRecordingSegment(self._root, trace_name, **time_kwargs)
            self.add_recording_segment(rec_segment)

        # load probe
        probe_dict = self._root.attrs.get("probe", None)
        if probe_dict is not None:
            probe = Probe.from_dict(probe_dict)
            self.set_probe(probe, in_place=True)

        # load properties
        if 'properties' in self._root:
            prop_group = self._root['properties']
            for key in prop_group.keys():
                values = self._root['properties'][key]
                self.set_property(key, values)

        self._kwargs = {'root_path': str(root_path.absolute())}


class ZarrRecordingSegment(BaseRecordingSegment):
    def __init__(self, root, dataset_name, **time_kwargs):
        BaseRecordingSegment.__init__(self, **time_kwargs)
        self._timeseries = root[dataset_name]

    def get_num_samples(self) -> int:
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex: Number of samples in the signal block
        """
        return self._timeseries.shape[0]

    def get_traces(self,
                   start_frame: Union[int, None] = None,
                   end_frame: Union[int, None] = None,
                   channel_indices: Union[List, None] = None,
                   ) -> np.ndarray:
        traces = self._timeseries[start_frame:end_frame]
        if channel_indices is not None:
            traces = traces[:, channel_indices]

        if self._timeseries.dtype.str.startswith('uint'):
            exp_idx = self._dtype.find('int') + 3
            exp = int(self._dtype[exp_idx:])
            traces = traces.astype('float32') - 2 ** (exp - 1)

        return traces


def read_zarr(*args, **kwargs):
    recording = ZarrRecordingExtractor(*args, **kwargs)
    return recording


read_zarr.__doc__ = ZarrRecordingExtractor.__doc__
