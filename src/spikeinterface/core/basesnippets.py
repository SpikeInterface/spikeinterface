from __future__ import annotations

from typing import Union
from .base import BaseSegment
from .baserecordingsnippets import BaseRecordingSnippets
import numpy as np
from warnings import warn

# snippets segments?


class BaseSnippets(BaseRecordingSnippets):
    """
    Abstract class representing several multichannel snippets.
    """

    _main_properties = ["group", "location", "gain_to_uV", "offset_to_uV"]
    _main_features = []

    def __init__(
        self, sampling_frequency: float, nbefore: Union[int, None], snippet_len: int, channel_ids: list, dtype
    ):
        BaseRecordingSnippets.__init__(
            self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=dtype
        )
        self._nbefore = nbefore
        self._snippet_len = snippet_len

        self._snippets_segments: list[BaseSnippetsSegment] = []
        # initialize main annotation and properties

    def __repr__(self):
        clsname = self.__class__.__name__
        nchan = self.get_num_channels()
        nseg = self.get_num_segments()
        sf_khz = self.get_sampling_frequency() / 1000.0
        txt = f"{clsname}: {nchan} channels - {nseg} segments -  {sf_khz:0.1f}kHz \n snippet_len:{self._snippet_len} before peak:{self._nbefore}"
        return txt

    @property
    def nbefore(self):
        return self._nbefore

    @property
    def snippet_len(self):
        return self._snippet_len

    def get_num_segments(self):
        return len(self._snippets_segments)

    def add_snippets_segment(self, snippets_segment):
        # todo: check channel count and sampling frequency
        self._snippets_segments.append(snippets_segment)
        snippets_segment.set_parent_extractor(self)

    @property
    def nafter(self):
        if self._nbefore is None:
            return None
        return self._snippet_len - self._nbefore

    @property
    def snippet_len(self):
        return self._snippet_len

    def get_num_snippets(self, segment_index=None):
        segment_index = self._check_segment_index(segment_index)
        return self._snippets_segments[segment_index].get_num_snippets()

    def get_total_snippets(self):
        s = 0
        for segment_index in range(self.get_num_segments()):
            s += self.get_num_snippets(segment_index)
        return s

    def is_aligned(self):
        return self._nbefore is not None

    def get_num_segments(self):
        return len(self._snippets_segments)

    def has_scaled_snippets(self):
        warn(
            "`has_scaled_snippets` is deprecated and will be removed in version 0.103.0. Please use `has_scaleable_traces()` instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.has_scaleable_traces()

    def get_frames(self, indices=None, segment_index: Union[int, None] = None):
        segment_index = self._check_segment_index(segment_index)
        spts = self._snippets_segments[segment_index]
        return spts.get_frames(indices)

    def get_snippets(
        self,
        indices=None,
        segment_index: Union[int, None] = None,
        channel_ids: Union[list, None] = None,
        return_scaled=False,
    ):
        segment_index = self._check_segment_index(segment_index)
        spts = self._snippets_segments[segment_index]
        channel_indices = self.ids_to_indices(channel_ids, prefer_slice=True)
        wfs = spts.get_snippets(indices, channel_indices=channel_indices)

        if return_scaled:
            if not self.has_scaleable_traces():
                raise ValueError(
                    "These snippets do not support return_scaled=True (need gain_to_uV and offset_" "to_uV properties)"
                )
            else:
                gains = self.get_property("gain_to_uV")
                offsets = self.get_property("offset_to_uV")
                gains = gains[channel_indices].astype("float32")
                offsets = offsets[channel_indices].astype("float32")
                wfs = wfs.astype("float32") * gains + offsets
        return wfs

    def get_snippets_from_frames(
        self,
        segment_index: Union[int, None] = None,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        channel_ids: Union[list, None] = None,
        return_scaled=False,
    ):
        segment_index = self._check_segment_index(segment_index)
        spts = self._snippets_segments[segment_index]
        indices = spts.frames_to_indices(start_frame, end_frame)

        return self.get_snippets(indices, channel_ids=channel_ids, return_scaled=return_scaled)

    def _save(self, format="binary", **save_kwargs):
        raise NotImplementedError

    def select_channels(self, channel_ids: list | np.array | tuple) -> "BaseSnippets":
        from .channelslice import ChannelSliceSnippets

        return ChannelSliceSnippets(self, channel_ids)

    def _channel_slice(self, channel_ids, renamed_channel_ids=None):
        from .channelslice import ChannelSliceSnippets
        import warnings

        warnings.warn(
            "Snippets.channel_slice will be removed in version 0.103, use `select_channels` or `rename_channels` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        sub_recording = ChannelSliceSnippets(self, channel_ids, renamed_channel_ids=renamed_channel_ids)
        return sub_recording

    def _remove_channels(self, remove_channel_ids):
        from .channelslice import ChannelSliceSnippets

        new_channel_ids = self.channel_ids[~np.isin(self.channel_ids, remove_channel_ids)]
        sub_recording = ChannelSliceSnippets(self, new_channel_ids)
        return sub_recording

    def _select_segments(self, segment_indices):
        from .segmentutils import SelectSegmentSnippets

        return SelectSegmentSnippets(self, segment_indices=segment_indices)

    def _save(self, format="npy", **save_kwargs):
        """
        At the moment only "npy" and "memory" avaiable:
        """

        if format == "npy":
            from spikeinterface.core.npysnippetsextractor import NpySnippetsExtractor

            folder = save_kwargs["folder"]
            file_paths = [folder / f"traces_cached_seg{i}.npy" for i in range(self.get_num_segments())]
            dtype = save_kwargs.get("dtype", None)
            if dtype is None:
                dtype = self.get_dtype()

            from spikeinterface.core.npysnippetsextractor import NpySnippetsExtractor

            NpySnippetsExtractor.write_snippets(snippets=self, file_paths=file_paths, dtype=dtype)
            cached = NpySnippetsExtractor(
                file_paths=file_paths,
                sampling_frequency=self.get_sampling_frequency(),
                channel_ids=self.get_channel_ids(),
                nbefore=self.nbefore,
                gain_to_uV=self.get_channel_gains(),
                offset_to_uV=self.get_channel_offsets(),
            )
            cached.dump(folder / "npy.json", relative_to=folder)

            from spikeinterface.core.npyfoldersnippets import NpyFolderSnippets

            cached = NpyFolderSnippets(folder_path=folder)

        elif format == "memory":
            snippets_list = []
            spikesframes_list = []
            for i in range(self.get_num_segments()):
                spikesframes_list.append(self.get_frames(segment_index=i))
                snippets_list.append(self.get_snippets(segment_index=i))

            from .numpyextractors import NumpySnippets

            cached = NumpySnippets(
                snippets_list=snippets_list,
                spikesframes_list=spikesframes_list,
                sampling_frequency=self.get_sampling_frequency(),
                nbefore=self.nbefore,
                channel_ids=self.channel_ids,
            )

        else:
            raise ValueError(f"format {format} not supported")

        if self.get_property("contact_vector") is not None:
            probegroup = self.get_probegroup()
            cached.set_probegroup(probegroup)

        return cached

    def get_times(self):
        return self.get_frames() / self.sampling_frequency


class BaseSnippetsSegment(BaseSegment):
    """
    Abstract class representing multichannel snippets
    """

    def __init__(self):
        BaseSegment.__init__(self)

    def get_snippets(
        self,
        indices,
        channel_indices: Union[list, None] = None,
    ) -> np.ndarray:
        """
        Return the snippets, optionally for a subset of samples and/or channels

        Parameters
        ----------
        indices : list[int]
            Indices of the snippets to return
        channel_indices : Union[list, None], default: None
            Indices of channels to return, or all channels if None

        Returns
        -------
        snippets : np.ndarray
            Array of snippets, num_snippets x num_samples x num_channels
        """
        raise NotImplementedError

    def get_num_snippets(self):
        """Returns the number of snippets in this segment

        Returns:
            SampleIndex : Number of snippets in the segment
        """
        raise NotImplementedError

    def get_frames(self, indices):
        """Returns the frames of the snippets in this  segment

        Returns:
            SampleIndex : Number of samples in the  segment
        """
        raise NotImplementedError

    def frames_to_indices(self, start_frame: Union[int, None] = None, end_frame: Union[int, None] = None):
        """
        Return the slice of snippets

        Parameters
        ----------
        start_frame : Union[int, None], default: None
            start sample index, or zero if None
        end_frame : Union[int, None], default: None
            end_sample, or number of samples if None

        Returns
        -------
        snippets : slice
            slice of selected snippets
        """
        raise NotImplementedError
