from __future__ import annotations
from pathlib import Path

import numpy as np

from spikeinterface.core.basesnippets import BaseSnippets, BaseSnippetsSegment
from .core_tools import define_function_from_class


class NpySnippetsExtractor(BaseSnippets):
    """
    Dead simple and super light format based on the NPY numpy format.

    It is in fact an archive of several .npy format.
    All spike are store in two columns maner index+labels
    """

    mode = "file"
    name = "npy"

    def __init__(
        self, file_paths, sampling_frequency, channel_ids=None, nbefore=None, gain_to_uV=None, offset_to_uV=None
    ):
        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        num_segments = len(file_paths)
        data = np.load(file_paths[0], mmap_mode="r")

        if channel_ids is None:
            channel_ids = np.arange(data["snippet"].shape[2])

        BaseSnippets.__init__(
            self,
            sampling_frequency,
            nbefore=nbefore,
            snippet_len=data["snippet"].shape[1],
            dtype=data["snippet"].dtype,
            channel_ids=channel_ids,
        )

        for i in range(num_segments):
            snp_segment = NpySnippetsSegment(file_paths[i])
            self.add_snippets_segment(snp_segment)

        if gain_to_uV is not None:
            self.set_channel_gains(gain_to_uV)

        if offset_to_uV is not None:
            self.set_channel_offsets(offset_to_uV)

        self._kwargs = {
            "file_paths": [str(Path(f).absolute()) for f in file_paths],
            "sampling_frequency": sampling_frequency,
            "channel_ids": channel_ids,
            "nbefore": nbefore,
            "gain_to_uV": gain_to_uV,
            "offset_to_uV": offset_to_uV,
        }

    @staticmethod
    def write_snippets(snippets, file_paths, dtype=None):
        """
        Save snippet extractor in binary .npy format.

        Parameters
        ----------
        snippets: SnippetsExtractor
            The snippets extractor object to be saved in .npy format
        file_paths: str
            The paths to the files.
        dtype: None, str or dtype
            Typecode or data-type to which the snippets will be cast.
        {}
        """
        if not isinstance(file_paths, list):
            file_paths = [file_paths]
        if dtype is None:
            dtype = snippets.dtype
        assert len(file_paths) == snippets.get_num_segments()
        snippets_t = np.dtype(
            [("frame", np.int64), ("snippet", dtype, (snippets.snippet_len, snippets.get_num_channels()))]
        )

        for i in range(snippets.get_num_segments()):
            n = snippets.get_num_snippets(i)
            arr = np.empty(n, dtype=snippets_t, order="F")
            arr["frame"] = snippets.get_frames(segment_index=i)
            arr["snippet"] = snippets.get_snippets(segment_index=i).astype(dtype, copy=False)
            file_paths[i].parent.mkdir(parents=True, exist_ok=True)
            np.save(file_paths[i], arr)


class NpySnippetsSegment(BaseSnippetsSegment):
    def __init__(self, file):
        BaseSnippetsSegment.__init__(self)

        npy = np.load(file, mmap_mode="r")
        self._snippets = npy["snippet"]
        self._spikestimes = npy["frame"]

    def get_snippets(
        self,
        indices: list[int],
        channel_indices: list | None = None,
    ) -> np.ndarray:
        """
        Return the snippets, optionally for a subset of samples and/or channels

        Parameters
        ----------
        indices: list[int]
            Indices of the snippets to return, or all if None
        channel_indices: list | None, default: None
            Indices of channels to return, or all channels if None

        Returns
        -------
        snippets: np.ndarray
            Array of snippets, num_snippets x num_samples x num_channels
        """
        if indices is None:
            return self._snippets[:, :, channel_indices]
        return self._snippets[indices, :, channel_indices]

    def get_num_snippets(self):
        return self._spikestimes.shape[0]

    def frames_to_indices(self, start_frame: int | None = None, end_frame: int | None = None):
        """
        Return the slice of snippets

        Parameters
        ----------
        start_frame: int | None, default: None
            start sample index, or zero if None
        end_frame: int | None, default: None
            end_sample, or number of samples if None

        Returns
        -------
        snippets: slice
            slice of selected snippets
        """
        # must be implemented in subclass
        if start_frame is None:
            init = 0
        else:
            init = np.searchsorted(self._spikestimes, start_frame, side="left")
        if end_frame is None:
            endi = self._spikestimes.shape[0]
        else:
            endi = np.searchsorted(self._spikestimes, end_frame, side="left")
        return slice(init, endi, 1)

    def get_frames(self, indices=None):
        """Returns the frames of the snippets in this segment

        Returns:
            SampleIndex: Number of samples in the segment
        """
        if indices is None:
            return self._spikestimes
        raise self._spikestimes[indices]


read_npy_snippets = define_function_from_class(source_class=NpySnippetsExtractor, name="read_npy_snippets")
