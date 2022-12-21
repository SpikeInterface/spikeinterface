import numpy as np

from.recording_tools import get_channel_distances, get_noise_levels

from .template_tools import get_template_amplitudes, get_template_extremum_channel



class ChannelSparsity:
    """
    Handle channel sparsity for a set of units.

    Internally, sparsity is stored as a boolean mask.

    The ChannelSparsity object can also provide other sparsity representations:
        * ChannelSparsity.unit_id_to_channel_ids : unit_id to channel_ids
        * ChannelSparsity.unit_id_to_channel_indices : unit_id channel_inds

    By default it is constructed with a boolean array:
    >>> sparsity = ChannelSparsity(mask, unit_ids, channel_ids)

    But can be also constructed from a dictionary:
    >>> sparsity = ChannelSparsity.from_unit_id_to_channel_ids(unit_id_to_channel_ids, unit_ids, channel_ids)

    The class can also be used to construct/estimate the sparsity from a Waveformextractor
    with several methods::

    - Using the N best channels (largest template amplitude):
    >>> sparsity = ChannelSparsity.from_best_channels(we, num_channels, peak_sign='neg')

    - Using a neighborhood by radius:
    >>> sparsity = ChannelSparsity.from_radius(we, radius_um, peak_sign='neg')

    - Using a SNR threshold:
    >>> sparsity = ChannelSparsity.from_threshold(we, threshold, peak_sign='neg')

    - Using a recording/sorting property (e.g. 'group'):
    >>> sparsity = ChannelSparsity.from_property(we, by_property="group")


    Parameters
    ----------
    mask: np.array of bool
        The sparsity mask (num_units, num_channels)
    unit_ids: list or array
        Unit ids vector or list
    channel_ids: list or array
        Channel ids vector or list
    """
    def __init__(self, mask, unit_ids, channel_ids):
        self.unit_ids = np.asarray(unit_ids)
        self.channel_ids = np.asarray(channel_ids)
        self.mask = np.asarray(mask, dtype='bool')
        assert self.mask.shape[0] == self.unit_ids.shape[0]
        assert self.mask.shape[1] == self.channel_ids.shape[0]

        # some precomputed dict
        self._unit_id_to_channel_ids = None
        self._unit_id_to_channel_indices = None

    def __repr__(self):
        ratio = np.mean(self.mask)
        txt = f'ChannelSparsity - units:{self.unit_ids.size} - channels:{self.channel_ids.size} - ratio{ratio:0.2f}'
        return txt

    @property
    def unit_id_to_channel_ids(self):
        if self._unit_id_to_channel_ids is None:
            self._unit_id_to_channel_ids = {}
            for unit_ind, unit_id in enumerate(self.unit_ids):
                channel_inds = np.flatnonzero(self.mask[unit_ind, :])
                self._unit_id_to_channel_ids[unit_id] = self.channel_ids[channel_inds]
        return self._unit_id_to_channel_ids
    
    @property
    def unit_id_to_channel_indices(self):
        if self._unit_id_to_channel_indices is None:
            self._unit_id_to_channel_indices = {}
            for unit_ind, unit_id in enumerate(self.unit_ids):
                channel_inds = np.flatnonzero(self.mask[unit_ind, :])
                self._unit_id_to_channel_indices[unit_id] = channel_inds
        return self._unit_id_to_channel_indices

    @classmethod
    def from_unit_id_to_channel_ids(cls, unit_id_to_channel_ids, unit_ids, channel_ids):
        """
        Create a sparsity object from dict unit_id to channel_ids.
        """
        unit_ids = list(unit_ids)
        channel_ids = list(channel_ids)
        mask = np.zeros((len(unit_ids), len(channel_ids)), dtype='bool')
        for unit_id, chan_ids in unit_id_to_channel_ids.items():
            unit_ind = unit_ids.index(unit_id)
            channel_inds = [channel_ids.index(chan_id) for chan_id in chan_ids]
            mask[unit_ind, channel_inds] = True
        return cls(mask, unit_ids, channel_ids)

    def to_dict(self):
        """
        Return a serializable dict.
        """
        return dict(
            unit_id_to_channel_ids={k: list(v) for k, v in self.unit_id_to_channel_ids.items()},
            channel_ids=list(self.channel_ids),
            unit_ids=list(self.unit_ids),
        )

    @classmethod
    def from_dict(cls, d):
        return cls.from_unit_id_to_channel_ids(**d)

    ## Some convinient function to compute sparsity from several strategy
    @classmethod
    def from_best_channels(cls, we, num_channels, peak_sign='neg'):
        """
        Construct sparsity from N best channels with the largest amplitude.
        Use the 'num_channels' argument to specify the number of channels.
        """
        mask = np.zeros((we.unit_ids.size, we.channel_ids.size), dtype='bool')
        peak_values = get_template_amplitudes(we, peak_sign=peak_sign)
        for unit_ind, unit_id in enumerate(we.unit_ids):
            chan_inds = np.argsort(np.abs(peak_values[unit_id]))[::-1]
            chan_inds = chan_inds[:num_channels]
            mask[unit_ind, chan_inds] = True
        return cls(mask, we.unit_ids, we.channel_ids)

    @classmethod
    def from_radius(cls, we, radius_um, peak_sign='neg'):
        """
        Construct sparsity from a radius around the best channel.
        Use the 'radius_um' argument to specify the radius in um
        """
        mask = np.zeros((we.unit_ids.size, we.channel_ids.size), dtype='bool')
        distances = get_channel_distances(we.recording)
        best_chan = get_template_extremum_channel(we, peak_sign=peak_sign, outputs="index")
        for unit_ind, unit_id in enumerate(we.unit_ids):
            chan_ind = best_chan[unit_id]
            chan_inds, = np.nonzero(distances[chan_ind, :] <= radius_um)
            mask[unit_ind, chan_inds] = True
        return cls(mask, we.unit_ids, we.channel_ids)

    @classmethod
    def from_threshold(cls, we, threshold, peak_sign='neg'):
        """
        Construct sparsity from a thresholds based on template signal-to-noise ratio.
        Use the 'threshold' argument to specify the SNR threshold.
        """
        mask = np.zeros((we.unit_ids.size, we.channel_ids.size), dtype='bool')

        peak_values = get_template_amplitudes(we, peak_sign=peak_sign, mode="extremum")
        noise = get_noise_levels(we.recording, return_scaled=we.return_scaled)
        for unit_ind, unit_id in enumerate(we.unit_ids):
            chan_inds = np.nonzero((np.abs(peak_values[unit_id]) / noise) >= threshold)
            mask[unit_ind, chan_inds] = True
        return cls(mask, we.unit_ids, we.channel_ids)

    @classmethod
    def from_property(cls, we, by_property):
        """
        Construct sparsity witha property of the recording and sorting(e.g. 'group').
        Use the 'by_property' argument to specify the property name.
        """
        # check consistency
        assert by_property in we.recording.get_property_keys(), f"Property {by_property} is not a recording property"
        assert by_property in we.sorting.get_property_keys(), f"Property {by_property} is not a sorting property"

        mask = np.zeros((we.unit_ids.size, we.channel_ids.size), dtype='bool')
        rec_by = we.recording.split_by(by_property)
        for unit_ind, unit_id in enumerate(we.unit_ids):
            unit_property = we.sorting.get_property(by_property)[unit_ind]
            assert unit_property in rec_by.keys(), (
                   f"Unit property {unit_property} cannot be found in the recording properties")
            chan_inds = we.recording.ids_to_indices(rec_by[unit_property].get_channel_ids())
            mask[unit_ind, chan_inds] = True
        return cls(mask, we.unit_ids, we.channel_ids)
