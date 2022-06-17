import numpy as np
from spikeinterface.core import BaseSnippets, BaseSnippetsSegment
from typing import List, Union


class NumpySnippetsExtractor(BaseSnippets):
    """
    In memory recording.
    Contrary to previous version this class does not handle npy files.

    Parameters
    ----------
    snippets_list:  list of array or array (if mono segment)
        The snippets to instantiate a mono or multisegment basesnippet
    spikesframes_list: list of array or array (if mono segment)
        Frame of each snippet
    sampling_frequency: float
        The sampling frequency in Hz

    channel_ids: list
        An optional list of channel_ids. If None, linear channels are assumed
    """

    is_writable = False

    def __init__(self, snippets_list, spikesframes_list, sampling_frequency, nafter=None, channel_ids=None):
        if isinstance(snippets_list, list):
            assert all(isinstance(e, np.ndarray) for e in snippets_list), 'must give a list of numpy array'
        else:
            assert isinstance(snippets_list, np.ndarray), 'must give a list of numpy array'
            snippets_list = [snippets_list]
        if isinstance(spikesframes_list, list):
            assert all(isinstance(e, np.ndarray) for e in spikesframes_list), 'must give a list of numpy array'
        else:
            assert isinstance(spikesframes_list, np.ndarray), 'must give a list of numpy array'
            spikesframes_list = [spikesframes_list]

        dtype = snippets_list[0].dtype
        assert all(dtype == ts.dtype for ts in snippets_list)

        if channel_ids is None:
            channel_ids = np.arange(snippets_list[0].shape[2])
        else:
            channel_ids = np.asarray(channel_ids)
            assert channel_ids.size == snippets_list[0].shape[2]
        BaseSnippets.__init__(self, sampling_frequency,  nafter=nafter, 
                              snippet_len=snippets_list[0].shape[1], channel_ids=channel_ids,
                              dtype=dtype)

        self.is_dumpable = False

        for snippets,spikesframes in zip(snippets_list, spikesframes_list):
            snp_segment = NumpySnippetsSegment(snippets, spikesframes)
            self.add_snippets_segment(snp_segment)

        self._kwargs = {'snippets_list': snippets_list, 
                        'spikesframes_list': spikesframes_list,
                        'nafter': nafter,
                        'sampling_frequency': sampling_frequency,
                        'channel_ids':channel_ids
                        }

class NumpySnippetsSegment(BaseSnippetsSegment):
    def __init__(self, snippets, spikesframes):
        BaseSnippetsSegment.__init__(self)
        self._snippets = snippets
        self._spikestimes = spikesframes

    def get_snippets(self,
                    indices,
                    channel_indices: Union[List, None] = None,
                    ) -> np.ndarray:
        """
        Return the snippets, optionally for a subset of samples and/or channels

        Parameters
        ----------
        indexes: (Union[int, None], optional)
            start sample index, or zero if None. Defaults to None.
        end_frame: (Union[int, None], optional)
            end_sample, or number of samples if None. Defaults to None.
        channel_indices: (Union[List, None], optional)
            Indices of channels to return, or all channels if None. Defaults to None.

        Returns
        -------
        snippets: np.ndarray
            Array of snippets, num_snippets x num_samples x num_channels
        """
        return self._snippets[indices,:,channel_indices]

    def get_num_snippets(self):
        return self._spikestimes.shape[0]

    def frames_to_indices(self,
                        start_frame: Union[int, None] = None,
                        end_frame: Union[int, None] = None):
        """
        Return the slice of snippets

        Parameters
        ----------
        start_frame: (Union[int, None], optional)
            start sample index, or zero if None. Defaults to None.
        end_frame: (Union[int, None], optional)
            end_sample, or number of samples if None. Defaults to None.

        Returns
        -------
        snippets: slice
            slice of selected snippets
        """
        # must be implemented in subclass
        if start_frame is None:
            init = 0
        else:
            init = np.searchsorted(self._spikestimes, start_frame, side='left')
        if end_frame is None:
            endi = self._spikestimes.shape[0]
        else:
            endi = np.searchsorted(self._spikestimes, end_frame, side='left')
        return slice(init,endi,1)

    def get_frames(self, indeces=None):
        """Returns the frames of the snippets in this segment

        Returns:
            SampleIndex: Number of samples in the segment
        """
        if indeces is None:
            return self._spikestimes
        raise self._spikestimes[indeces]