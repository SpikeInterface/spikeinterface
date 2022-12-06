import numpy as np


class ChannelSparsity:
    """
    Handle channel sparsity for a set of units.

    Internally stored as a boolean mask.

    Can also provide other dict to manipulate this sparsity:
        * ChannelSparsity.id_to_id : unit_id to channel_ids
        * ChannelSparsity.id_to_iindex : unit_id channel_inds

    By default it is constructed with a boolean array.
    But can be also constructed by dict

    sparsity = ChannelSparsity(mask, unit_ids, channel_ids)
    sparsity = ChannelSparsity.from_id_to_id(id_to_id, unit_ids, channel_ids)

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
        self.mask = np.asarray(mask)
        assert self.mask.shape[0] == self.unit_ids.shape[0]
        assert self.mask.shape[1] == self.channel_ids.shape[0]

        # some precomputed dict
        self._id_to_id = None
        self._id_to_index = None

    def __repr__(self):
        ratio = np.mean(self.mask)
        txt = f'ChannelSparsity - units:{self.unit_ids.size} - channels:{self.channel_ids.size} - ratio{ratio:0.2f}'
        return txt

    @property
    def id_to_id(self):
        if self._id_to_id is None:
            self._id_to_id = {}
            for unit_ind, unit_id in enumerate(self.unit_ids):
                channel_inds = np.flatnonzero(self.mask[unit_ind, :])
                self._id_to_id[unit_id] = self.channel_ids[channel_inds]
        return self._id_to_id
    
    @property
    def id_to_index(self):
        if self._id_to_index is None:
            self._id_to_index = {}
            for unit_ind, unit_id in enumerate(self.unit_ids):
                channel_inds = np.flatnonzero(self.mask[unit_ind, :])
                self._id_to_index[unit_id] = channel_inds
        return self._id_to_index

    @classmethod
    def from_id_to_id(cls, id_to_id, unit_ids, channel_ids):
        unit_ids = list(unit_ids)
        channel_ids = list(channel_ids)
        mask = np.zeros((len(unit_ids), len(channel_ids)), dtype='bool')
        for unit_id, chan_ids in id_to_id.items():
            unit_ind = unit_ids.index(unit_id)
            channel_inds = [channel_ids.index(chan_id) for chan_id in chan_ids]
            mask[unit_ind, channel_inds] = True
        return cls(mask, unit_ids, channel_ids)

    # @alessio : maybe this is unnecessary
    @classmethod
    def from_id_to_index(cls, id_to_index, unit_ids, channel_ids):
        unit_ids = list(unit_ids)
        mask = np.zeros((len(unit_ids), len(channel_ids)), dtype='bool')
        for unit_id, channel_inds in id_to_index.items():
            unit_ind = unit_ids.index(unit_id)
            mask[unit_ind, channel_inds] = True
        return cls(mask, unit_ids, channel_ids)
    
    def to_dict(self):
        """
        Return a serializable dict.
        """
        return dict(
            id_to_id={k:list(v) for k, v in self.id_to_id.items()},
            channel_ids=list(self.channel_ids),
            unit_ids=list(self.unit_ids),
        )
    
    @classmethod
    def from_dict(cls, d):
        return cls.from_id_to_id(**d)

