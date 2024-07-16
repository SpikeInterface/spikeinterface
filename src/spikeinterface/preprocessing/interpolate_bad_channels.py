from __future__ import annotations

import numpy as np

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.preprocessing import preprocessing_tools


class InterpolateBadChannelsRecording(BasePreprocessor):
    """
    Interpolate the channel labeled as bad channels using linear interpolation.
    This is based on the distance (Gaussian kernel) from the bad channel,
    as determined from x,y channel coordinates.

    Details of the interpolation function (written by Olivier Winter) used in the IBL pipeline
    can be found at:

    International Brain Laboratory et al. (2022). Spike sorting pipeline for the
    International Brain Laboratory. https://www.internationalbrainlab.com/repro-ephys

    Parameters
    ----------
    recording : BaseRecording
        The parent recording
    bad_channel_ids : list or 1d np.array
        Channel ids of the bad channels to interpolate.
    sigma_um : float or None, default: None
        Distance between sequential channels in um. If None, will use
        the most common distance between y-axis channels
    p : float, default: 1.3
        Exponent of the Gaussian kernel. Determines rate of decay
        for distance weightings
    weights : np.array or None, default: None
        The weights to give to bad_channel_ids at interpolation.
        If None, weights are automatically computed

    Returns
    -------
    interpolated_recording : InterpolateBadChannelsRecording
        The recording object with interpolated bad channels
    """

    def __init__(self, recording, bad_channel_ids, sigma_um=None, p=1.3, weights=None):
        BasePreprocessor.__init__(self, recording)

        bad_channel_ids = np.array(bad_channel_ids)
        self.check_inputs(recording, bad_channel_ids)

        self.bad_channel_ids = bad_channel_ids
        self._bad_channel_idxs = recording.ids_to_indices(self.bad_channel_ids)
        self._good_channel_idxs = ~np.isin(np.arange(recording.get_num_channels()), self._bad_channel_idxs)
        self._bad_channel_idxs.setflags(write=False)

        if sigma_um is None:
            sigma_um = estimate_recommended_sigma_um(recording)

        if weights is None:
            locations = recording.get_channel_locations()
            locations_good = locations[self._good_channel_idxs]
            locations_bad = locations[self._bad_channel_idxs]
            weights = preprocessing_tools.get_kriging_channel_weights(locations_good, locations_bad, sigma_um, p)

        for parent_segment in recording._recording_segments:
            rec_segment = InterpolateBadChannelsSegment(
                parent_segment, self._good_channel_idxs, self._bad_channel_idxs, weights
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording, bad_channel_ids=bad_channel_ids, p=p, sigma_um=sigma_um, weights=weights
        )

    def check_inputs(self, recording, bad_channel_ids):
        if bad_channel_ids.ndim != 1:
            raise TypeError("'bad_channel_ids' must be a 1d array or list.")

        if not recording.has_channel_location():
            raise ValueError("A probe must be attached to use bad channel interpolation. Use set_probe(...)")

        if recording.get_probe().si_units != "um":
            raise NotImplementedError("Channel spacing units must be um")


class InterpolateBadChannelsSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment, good_channel_indices, bad_channel_indices, weights):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self._good_channel_indices = good_channel_indices
        self._bad_channel_indices = bad_channel_indices
        self._weights = weights

    def get_traces(self, start_frame, end_frame, channel_indices):
        if channel_indices is None:
            channel_indices = slice(None)

        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))

        traces = traces.copy()

        traces[:, self._bad_channel_indices] = traces[:, self._good_channel_indices] @ self._weights

        return traces[:, channel_indices]


def estimate_recommended_sigma_um(recording):
    """
    Get the most common distance between channels on the y-axis
    """
    y_sorted = np.sort(recording.get_channel_locations()[:, 1])
    import scipy.stats

    return scipy.stats.mode(np.diff(np.unique(y_sorted)), keepdims=False)[0]


interpolate_bad_channels = define_function_from_class(
    source_class=InterpolateBadChannelsRecording, name="interpolate_bad_channels"
)
