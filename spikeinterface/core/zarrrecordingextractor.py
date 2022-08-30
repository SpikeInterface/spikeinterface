from typing import List, Union

from pathlib import Path
from probeinterface import ProbeGroup

import numpy as np

from .baserecording import BaseRecording, BaseRecordingSegment


try:
    import zarr
    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False


class ZarrRecordingExtractor(BaseRecording):
    """
    RecordingExtractor for a zarr format

    Parameters
    ----------
    root_path: str or Path
        Path to the zarr root file
    storage_options: dict or None
        Storage options for zarr `store`. E.g., if "s3://" or "gcs://" they can provide authentication methods, etc.

    Returns
    -------
    recording: ZarrRecordingExtractor
        The recording Extractor
    """
    extractor_name = 'ZarrRecording'
    installed = HAVE_ZARR  # check at class level if installed or not
    mode = 'file'
    # error message when not installed
    installation_mesg = "To use the ZarrRecordingExtractor install zarr: \n\n pip install zarr\n\n"
    name = "zarr"

    def __init__(self, root_path: Union[Path, str], storage_options=None):
        assert self.installed, self.installation_mesg
        
        if storage_options is None:
            if isinstance(root_path, str):
                root_path_init = root_path
                root_path = Path(root_path)
            else:
                root_path_init = str(root_path)
            root_path_kwarg = str(root_path.absolute())
        else:
            root_path_init = root_path
            root_path_kwarg = root_path_init
        
        self._root = zarr.open(root_path_init, mode="r", storage_options=storage_options)

        sampling_frequency = self._root.attrs.get("sampling_frequency", None)
        num_segments = self._root.attrs.get("num_segments", None)
        assert "channel_ids" in self._root.keys(), "'channel_ids' dataset not found!"
        channel_ids = self._root["channel_ids"][:]

        assert sampling_frequency is not None, "'sampling_frequency' attiribute not found!"
        assert num_segments is not None, "'num_segments' attiribute not found!"

        channel_ids = np.array(channel_ids)

        dtype = self._root['traces_seg0'].dtype

        BaseRecording.__init__(self, sampling_frequency, channel_ids, dtype)

        dtype = np.dtype(dtype)
        t_starts = self._root.get("t_starts", None)

        total_nbytes = 0
        total_nbytes_stored = 0
        cr_by_segment = {}
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
            
            nbytes_segment = self._root[trace_name].nbytes
            nbytes_stored_segment = self._root[trace_name].nbytes_stored
            cr_by_segment[segment_index] = nbytes_segment / nbytes_stored_segment
            
            total_nbytes += nbytes_segment
            total_nbytes_stored += nbytes_stored_segment
            self.add_recording_segment(rec_segment)

        # load probe
        probe_dict = self._root.attrs.get("probe", None)
        if probe_dict is not None:
            probegroup = ProbeGroup.from_dict(probe_dict)
            self.set_probegroup(probegroup, in_place=True)

        # load properties
        if 'properties' in self._root:
            prop_group = self._root['properties']
            for key in prop_group.keys():
                values = self._root['properties'][key]
                self.set_property(key, values)

        # load annotations
        annotations = self._root.attrs.get("annotations", None)
        if annotations is not None:
            self.annotate(**annotations)
        # annotate compression ratios
        cr = total_nbytes / total_nbytes_stored
        self.annotate(compression_ratio=cr, compression_ratio_segments=cr_by_segment)
        
        self._kwargs = {'root_path': root_path_kwarg,
                        'storage_options': storage_options}


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
        return traces


def read_zarr(*args, **kwargs):
    recording = ZarrRecordingExtractor(*args, **kwargs)
    return recording


read_zarr.__doc__ = ZarrRecordingExtractor.__doc__


def get_default_zarr_compressor(clevel=5):
    """
    Return default Zarr compressor object for good preformance in int16 
    electrophysiology data.

    cname: zstd (zstandard)
    clevel: 5
    shuffle: BITSHUFFLE

    Parameters
    ----------
    clevel : int, optional
        Compression level (higher -> more compressed).
        Minimum 1, maximum 9. By default 5

    Returns
    -------
    Blosc.compressor
        The compressor object that can be used with the save to zarr function
    """
    assert ZarrRecordingExtractor.installed, ZarrRecordingExtractor.installation_mesg
    from numcodecs import Blosc
    return Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.BITSHUFFLE)
