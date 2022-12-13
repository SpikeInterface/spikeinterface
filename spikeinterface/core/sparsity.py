import numpy as np


class ChannelSparsity:
    """
    Handle channel sparsity for a set of units.

    Internally stored as a boolean mask.

    Can also provide other dict to manipulate this sparsity:
        * ChannelSparsity.unit_id_to_channel_ids : unit_id to channel_ids
        * ChannelSparsity.id_to_iindex : unit_id channel_inds

    By default it is constructed with a boolean array.
    But can be also constructed by dict

    sparsity = ChannelSparsity(mask, unit_ids, channel_ids)
    sparsity = ChannelSparsity.from_unit_id_to_channel_ids(unit_id_to_channel_ids, unit_ids, channel_ids)

    Parameters
    ----------
    mask: np.array of bool
        
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
            unit_id_to_channel_ids={k:list(v) for k, v in self.unit_id_to_channel_ids.items()},
            channel_ids=list(self.channel_ids),
            unit_ids=list(self.unit_ids),
        )
    
    @classmethod
    def from_dict(cls, d):
        return cls.from_unit_id_to_channel_ids(**d)
    


    # @alessio : I also would like to have this here and slowly remove
    # get_template_channel_sparsity() but this would lead to move more or less
    # all the emplates_tools.py into core.
    # to be discussed
    @classmethod
    def from_radius(cls, we, radius_um):
        """
        

        """
        pass

    @classmethod
    def from_best_channels(cls, we, num_channels):
        """
        

        """
        pass

    @classmethod
    def from_threshold(cls, we, threshold):
        """
        

        """
        pass

    @classmethod
    def from_property(cls, we, property):
        """
        

        """
        pass


    