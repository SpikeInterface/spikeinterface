import numpy as np

from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core.core_tools import define_function_from_class

from ..core import get_random_data_chunks

class InterpolateBadChannels(BasePreprocessor):
    """
    Interpolate bad channels based on weighted distance using linear interpolation.

    Uses the interpolate_bad_channels() function of the International Brain Laboratory

    International Brain Laboratory et al. (2022). Spike sorting pipeline for the
    International Brain Laboratory. https://www.internationalbrainlab.com/repro-ephys

    Parameters
    ----------

    data: (num_channels x num_samples) numpy array

    bad_channel_indexes: numpy array, indexes of the bad channels to interpolate.

    x: numpy array of channel x coordinates.

    y: numpy array of channel y coordinates.

    p: exponent of the Gaussian kernel. Determines rate of decay
       for distance weightings.

    kriging_distance_um: distance between sequential channels in um.

    """
    def __init__(self, recording):
        BasePreprocessor.__init__(self, recording, bad_channel_indexes, p=1.3, kriging_distance_um=20)

        self.bad_channel_indexes = bad_channel_indexes

        for parent_segment in recording._recording_segments:
            rec_segment = InterpolateBadChannelSegment(parent_segment,
                                                       x,
                                                       y,
                                                       bad_channel_indexes,
                                                       p,
                                                       kriging_distance)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording.to_dict(), dtype=dtype)




def interpolate_bad_channels(data, bad_channel_indexes, x, y, p, kriging_distance_um):
    """
    Interpolate the channel labeled as bad channels using linear interpolation.
    This is based on the distance from the bad channel, as determined from x,y
    channel coordinates. The weights applied to neighbouring channels come
    from an exponential decay function.

    Details of the interpolation function (Olivier Winter) used in the IBL pipeline
    can be found at:

    International Brain Laboratory et al. (2022). Spike sorting pipeline for the
    International Brain Laboratory. https://www.internationalbrainlab.com/repro-ephys
    """
    for i in bad_channel_indexes:

        # compute the weights to apply to neighbouring traces
        offset = np.abs(x - x[i] + 1j * (y - y[i]))
        weights = np.exp(-(offset / kriging_distance_um) ** p)
        weights[bad_channel_indexes] = 0
        weights[weights < 0.005] = 0
        weights = weights / np.sum(weights)

        # apply interpolation
        imult = np.where(weights > 0.005)[0]
        if imult.size == 0:
            data[i, :] = 0
            continue
        data[i, :] = np.matmul(weights[imult], data[imult, :])

    return