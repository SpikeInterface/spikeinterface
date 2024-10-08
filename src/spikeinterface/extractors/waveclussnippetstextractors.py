from __future__ import annotations

from pathlib import Path
import numpy as np

from spikeinterface.core import BaseSnippets, BaseSnippetsSegment
from spikeinterface.core.core_tools import define_function_from_class
from .matlabhelpers import MatlabHelper
from typing import List, Union


class WaveClusSnippetsExtractor(MatlabHelper, BaseSnippets):

    def __init__(self, file_path):
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        MatlabHelper.__init__(self, file_path)
        wc_snippets = self._getfield("spikes")
        # handle both types of waveclus results

        # the spikes can be in the times_file
        if file_path.name.startswith("times_"):
            times = self._getfield("cluster_class")[:, 1]
        elif file_path.name.endswith("_spikes.mat"):
            times = np.ravel(self._getfield("index"))
        else:
            raise ("Filename not compatible with waveclus file.")

        sampling_frequency = float(self._getfield("par/sr"))
        pre = int(self._getfield("par/w_pre")) - 1
        post = int(self._getfield("par/w_post")) + 1

        sp_len = pre + post

        nchannels = int(wc_snippets.shape[1] / sp_len)
        # waveclus use: #snippets,#concatenated_samples(sample * nchannels)
        # spikeinterface use: #snippets,#samples, #nchannels
        snippets = np.dstack(np.array_split(wc_snippets, nchannels, 1))

        BaseSnippets.__init__(
            self,
            sampling_frequency=sampling_frequency,
            nbefore=pre,
            snippet_len=sp_len,
            dtype=wc_snippets.dtype,
            channel_ids=np.arange(nchannels),
        )

        snp_segment = WaveClustSnippetsSegment(
            snippets=snippets, spikesframes=np.round(times * (sampling_frequency / 1000))
        )
        self.add_snippets_segment(snp_segment)

        self._kwargs = {"file_path": str(Path(file_path).absolute())}

    @staticmethod
    def write_snippets(snippets_extractor, save_file_path):
        assert snippets_extractor.is_aligned(), "Waveclus requires aligned spikes"
        save_file_path = Path(save_file_path)
        assert save_file_path.name.endswith("_spikes.mat"), "Waveclus snippets files should end with _spikes.mat"
        frame_to_ms = snippets_extractor.get_sampling_frequency() / 1000
        index = np.concatenate(
            [
                snippets_extractor.get_frames(segment_index=sinx) * frame_to_ms
                for sinx in range(snippets_extractor.get_num_segments())
            ]
        )
        spikes = np.concatenate(
            [
                snippets_extractor.get_snippets(segment_index=sinx) * frame_to_ms
                for sinx in range(snippets_extractor.get_num_segments())
            ]
        )
        spikes = np.swapaxes(spikes, 1, 2).reshape([spikes.shape[0], spikes.shape[1] * spikes.shape[2]], order="C")
        par = dict(
            sr=snippets_extractor.get_sampling_frequency(),
            w_pre=snippets_extractor.nbefore + 1,  # waveclus includes the peak in the pre samples
            w_post=snippets_extractor.nafter - 1,
        )
        MatlabHelper.write_dict_to_mat(
            mat_file_path=save_file_path, dict_to_write={"index": index, "spikes": spikes, "par": par}
        )


class WaveClustSnippetsSegment(BaseSnippetsSegment):
    def __init__(self, snippets, spikesframes):
        BaseSnippetsSegment.__init__(self)
        self._snippets = snippets
        self._spikestimes = spikesframes

    def get_snippets(
        self,
        indices,
        channel_indices: Union[List, None] = None,
    ) -> np.ndarray:
        """
        Return the snippets, optionally for a subset of samples and/or channels

        Parameters
        ----------
        indices: list[int]
            Indices of the snippets to return
        channel_indices: Union[list, None], default: None
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

    def frames_to_indices(self, start_frame: Union[int, None] = None, end_frame: Union[int, None] = None):
        """
        Return the slice of snippets

        Parameters
        ----------
        start_frame: Union[int, None], default: None
            start sample index, or zero if Non
        end_frame: Union[int, None], default: None
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
        return self._spikestimes[indices]


read_waveclus_snippets = define_function_from_class(
    source_class=WaveClusSnippetsExtractor, name="read_waveclus_snippets"
)
