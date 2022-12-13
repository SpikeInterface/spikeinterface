import pytest

import numpy as np

from spikeinterface.core import ChannelSparsity

def test_ChannelSparsity():

    unit_ids = ['a', 'b', 'c', 'd']
    channel_ids = [1, 2, 3]
    mask = np.zeros((4, 3), dtype='bool')
    mask[0, [0, ]] = True
    mask[1, [0, 1, 2]] = True
    mask[2, [0, 2]] = True
    mask[3, [0,]] = True

    sparsity = ChannelSparsity(mask, unit_ids, channel_ids)
    print(sparsity)

    with pytest.raises(AssertionError):
        sparsity = ChannelSparsity(mask, unit_ids, channel_ids[:2])

    for key, v in sparsity.unit_id_to_channel_ids.items():
        assert key in unit_ids
        assert np.all(np.in1d(v, channel_ids))

    for key, v in sparsity.unit_id_to_channel_indices.items():
        assert key in unit_ids
        assert np.all(v<len(channel_ids))

    sparsity2 = ChannelSparsity.from_unit_id_to_channel_ids(sparsity.unit_id_to_channel_ids, unit_ids, channel_ids)
    # print(sparsity2)
    assert np.array_equal(sparsity.mask, sparsity2.mask)

    d = sparsity.to_dict()
    # print(d)
    sparsity3 = ChannelSparsity.from_dict(d)
    assert np.array_equal(sparsity.mask, sparsity3.mask)
    # print(sparsity3)


if __name__ == '__main__':
    test_ChannelSparsity()