from __future__ import annotations

from typing import List, Union

import numpy as np

from spikeinterface.core import BaseSnippets, BaseSnippetsSegment


class AlignSnippets(BaseSnippets):
    installation_mesg = ""  # err

    def __init__(self, snippets, new_nbefore, new_nafter, mode="main_peak", interpolate=1, det_sign=0):
        assert isinstance(snippets, BaseSnippets), "'snippets' must be a SnippetsExtractor"
        assert mode in ("ch_peak", "main_peak"), "mode must be " "ch_peak" " or " "main_peak" ""

        BaseSnippets.__init__(
            self,
            sampling_frequency=snippets.get_sampling_frequency(),
            nbefore=new_nbefore,
            snippet_len=new_nbefore + new_nafter,
            channel_ids=snippets.channel_ids,
            dtype=snippets.get_dtype(),
        )
        assert self.snippet_len >= new_nbefore + new_nafter, "snippet_len smaller than new_nbefore+new_nafter"

        snippets.copy_metadata(self, only_main=False, ids=None)
        self._parent = snippets

        for i in range(snippets.get_num_segments()):
            self.add_snippets_segment(
                AlignSnippetsSegment(
                    snippets._snippets_segments[i],
                    snippets.snippet_len,
                    new_nbefore,
                    new_nafter,
                    mode,
                    interpolate,
                    det_sign,
                )
            )

        self._kwargs = dict(
            snippets=snippets,
            new_nbefore=new_nbefore,
            new_nafter=new_nafter,
            mode=mode,
            interpolate=interpolate,
            det_sign=det_sign,
        )


class AlignSnippetsSegment(BaseSnippetsSegment):
    def __init__(self, parent_snippets_segment, org_splen, new_nbefore, new_nafter, mode, interpolate, det_sign):
        BaseSnippetsSegment.__init__(self)
        self.parent_snippets_segment = parent_snippets_segment
        self._interpolate = interpolate
        self._new_nbefore = new_nbefore
        self._new_nafter = new_nafter
        self._org_splen = org_splen
        self._det_sign = det_sign
        start_search = int(new_nbefore * interpolate)
        end_search = int((org_splen - new_nafter + 1) * interpolate)
        if det_sign == 0:
            self._find_peak = lambda x: start_search + np.argmax(np.abs(x[:, start_search:end_search, :]), axis=1)
        elif det_sign > 0:
            self._find_peak = lambda x: start_search + np.argmax(x[:, start_search:end_search, :], axis=1)
        else:
            self._find_peak = lambda x: start_search + np.argmin(x[:, start_search:end_search, :], axis=1)
        self._mode = mode
        if mode == "main_peak":
            if det_sign == 0:
                self._find_main_ch = lambda x: np.argmax(np.abs(x))
            elif det_sign > 0:
                self._find_main_ch = np.argmax
            else:
                self._find_main_ch = np.argmin

    def get_snippets(
        self,
        indices=None,
        end_frame: Union[int, None] = None,
        channel_indices: Union[List, None] = None,
    ) -> np.ndarray:
        snippets = self.parent_snippets_segment.get_snippets(indices=indices, channel_indices=channel_indices)
        if self._interpolate > 1:
            xs = np.arange(0, self._org_splen, 1 / self._interpolate)
            from scipy.interpolate import CubicSpline

            snippets = CubicSpline(xs[:: self._interpolate], snippets, axis=1, bc_type="natural")(xs)

        peaks_pos = self._find_peak(snippets)

        aligned_snippets = np.empty(
            [snippets.shape[0], self._new_nbefore + self._new_nafter, snippets.shape[2]],
        )
        inp = self._interpolate
        pres = self._new_nbefore * self._interpolate
        posts = self._new_nafter * self._interpolate
        if self._mode == "main_peak":
            for i, pos in enumerate(peaks_pos):
                jpeak = pos[self._find_main_ch([snippets[i, p, ch] for ch, p in enumerate(pos)])]
                aligned_snippets[i, :, :] = snippets[i, jpeak - pres : jpeak + posts : inp, :]
        else:
            for i, pos in enumerate(peaks_pos):
                for chi, chpos in enumerate(pos):
                    aligned_snippets[i, :, chi] = snippets[i, chpos - pres : chpos + posts : inp, chi]
        return aligned_snippets

    def get_num_snippets(self):
        return self.parent_snippets_segment.get_num_snippets()

    def get_frames(self, indices):
        return self.parent_snippets_segment.get_num_snippets(indices)

    def frames_to_indices(self, start_frame: Union[int, None] = None, end_frame: Union[int, None] = None):
        return self.parent_snippets_segment.frames_to_indices(start_frame, end_frame)
