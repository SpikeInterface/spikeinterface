import numpy as np

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core.core_tools import define_function_from_class
import scipy.stats

class InterpolateBadChannels(BasePreprocessor):
    """
    Interpolate bad channels based on weighted distance using linear interpolation.

    Uses the interpolate_bad_channels() function of the International Brain Laboratory

    International Brain Laboratory et al. (2022). Spike sorting pipeline for the
    International Brain Laboratory. https://www.internationalbrainlab.com/repro-ephys

    Parameters
    ----------

    bad_channel_indexes: numpy array, indexes of the bad channels to interpolate.

    x: numpy array of channel x coordinates.

    y: numpy array of channel y coordinates.

    p: exponent of the Gaussian kernel. Determines rate of decay
       for distance weightings.

    kriging_distance_um: distance between sequential channels in um. If None, will use
                         the most common distance between y-axis channels.

    """
    name = 'interpolate_bad_channels'

    def __init__(self, recording, bad_channel_indexes, p=1.3, kriging_distance_um=None):
        BasePreprocessor.__init__(self, recording)

        self.bad_channel_indexes = bad_channel_indexes

        if recording.get_property('contact_vector') is None:
            raise ValueError('A probe must be attached to use bad channel interpolation. Use set_probe(...)')

        x = recording.get_probe().contact_positions[:, 0]
        y = recording.get_probe().contact_positions[:, 1]

        if kriging_distance_um is None:
            if recording.get_probe().si_units != "um":
                raise NotImplementedError("Channel spacing units must be um")

            kriging_distance_um = self.get_recommended_kriging_distance_um(recording, y)

        for parent_segment in recording._recording_segments:
            rec_segment = InterpolateBadChannelsSegment(parent_segment,
                                                        bad_channel_indexes,
                                                        x,
                                                        y,
                                                        p,
                                                        kriging_distance_um)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict(),
                            bad_channel_indexes=bad_channel_indexes,
                            p=p,
                            kriging_distance_um=kriging_distance_um)

    def get_recommended_kriging_distance_um(self, recording, y):
        """
        Get the most common distance between channels on the y-axis
        """
        return scipy.stats.mode(np.diff(np.unique(y)), keepdims=False)[0]

class InterpolateBadChannelsSegment(BasePreprocessorSegment):

    def __init__(self, parent_recording_segment, bad_channel_indexes, x, y, p, kriging_distance_um):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self._bad_channel_indexes = bad_channel_indexes
        self._x = x
        self._y = y
        self._p = p
        self._kriging_distance_um = kriging_distance_um
        self._all_weights = self.pre_calculate_channel_weights()

    def pre_calculate_channel_weights(self):
        """
        Pre-compute the channel weights for this InterpolateBadChannels
        instance. Code taken from original IBL function
        (see interpolate_bad_channels_ibl).
        """
        all_weights = np.empty((self._bad_channel_indexes.size, self._x.size))
        all_weights.fill(np.nan)

        for cnt, idx in enumerate(self._bad_channel_indexes):
            # compute the weights to apply to neighbouring traces
            offset = np.abs(self._x - self._x[idx] + 1j * (self._y - self._y[idx]))
            weights = np.exp(-(offset / self._kriging_distance_um) ** self._p)
            weights[self._bad_channel_indexes] = 0
            weights[weights < 0.005] = 0
            weights = weights / np.sum(weights)

            all_weights[cnt, :] = weights

        return all_weights

    def get_traces(self, start_frame, end_frame, channel_indices):

        traces = self.parent_recording_segment.get_traces(start_frame,
                                                          end_frame,
                                                          slice(None))

        traces = traces.copy()

        traces = interpolate_bad_channels_ibl(traces,  # TODO: check dims
                                              self._bad_channel_indexes,
                                              self._x,
                                              self._y,
                                              self._p,
                                              self._kriging_distance_um,
                                              self._all_weights)

        return traces

def interpolate_bad_channels_ibl(traces, bad_channel_indexes, x, y, p, kriging_distance_um, all_weights):
    """
    Interpolate the channel labeled as bad channels using linear interpolation.
    This is based on the distance from the bad channel, as determined from x,y
    channel coordinates. The weights applied to neighbouring channels come
    from an exponential decay function.

    Weights are pre-calculated (see pre_calculate_channel_weights()) and
    interpolated here.

    Details of the interpolation function (Olivier Winter) used in the IBL pipeline
    can be found at:

    International Brain Laboratory et al. (2022). Spike sorting pipeline for the
    International Brain Laboratory. https://www.internationalbrainlab.com/repro-ephys

    traces: (num_samples x num_channels) numpy array

    """
    for cnt, idx in enumerate(bad_channel_indexes):

        weights = all_weights[cnt, :]
        imult = np.where(weights > 0.005)[0]
        if imult.size == 0:
            traces[:, idx] = 0
            continue
        traces[:, idx] = np.matmul(traces[:, imult], weights[imult])

    return traces

interpolate_bad_channels = define_function_from_class(source_class=InterpolateBadChannels, name='interpolate_bad_channels')
