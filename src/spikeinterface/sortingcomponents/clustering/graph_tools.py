
import numpy as np


from scipy.sparse import csr_matrix, bsr_matrix, coo_matrix
from scipy.spatial.distance import pdist, cdist, squareform

from spikeinterface.sortingcomponents.clustering.tools import aggregate_sparse_features



def create_graph_from_peak_features(
    peaks,
    peak_feats,
    sparse_mask,
    peak_locs,
    channel_locations,
    bin_um=20.,
    dim=1,
    mode="full_connected_bin",
    direction="y",
    nn=20,
):
    """
    Create a sparse garph of peaks distances.
    This is done. This done using a binarization along an axis.
    Each peaks can connect to peaks of the same bin and neibour bins.


    """

    loc0 = min(channel_locations[:, dim])
    loc1 = max(channel_locations[:, dim])


    dim = "xyz".index(direction)

    # todo center this bons like in peak localization
    eps = bin_um/10.
    bins = np.arange(loc0-bin_um/2, loc1+bin_um/2+eps, bin_um)

    indices0 = []
    indices1 = []
    data = []
    for b in range(bins.size-1):
        
        # limits for peaks
        l0, l1 = bins[b], bins[b+1]
        mask = (peak_locs["y"] > l0) & (peak_locs["y"] <= l1)
        # in_bin_indices = np.flatnonzero(mask)


        # limits for features
        b0, b1 = l0 - bin_um, l1 + bin_um
        local_chans = np.flatnonzero((channel_locations[:, dim] > (b0 )) & (channel_locations[:, dim] <= (b1)))

        mask = (peak_locs["y"] > b0) & (peak_locs["y"] <= b1)
        peak_indices = np.flatnonzero(mask)

        local_locs = peak_locs[peak_indices]

        target_mask = (local_locs["y"] > l0) & (local_locs["y"] <= l1)
        target_indices = peak_indices[target_mask]


        local_feats, dont_have_channels = aggregate_sparse_features(peaks, peak_indices,
                                                                 peak_feats, sparse_mask, local_chans)
        # print(b)
        # print("dont_have_channels", np.sum(dont_have_channels))
        dont_have_channels_target = dont_have_channels[target_mask]

        flatten_feat = local_feats.reshape(local_feats.shape[0], -1)

        if mode == "full_connected_bin":
            local_dists = cdist(flatten_feat[target_mask], flatten_feat)
            n = peak_indices.size
            for i, ind0 in enumerate(target_indices):
                if dont_have_channels_target[i]:
                    # this spike is not valid because do not have local chans
                    continue
                indices0.append(np.full(n, ind0, dtype='int64'))
                indices1.append(peak_indices)
                data.append(local_dists[i, :])

        elif mode == "knn":
            # TODO use a better KNN tools (like pynndescnedt)
            # here we do brut force at the moment
            local_dists = cdist(flatten_feat[target_mask], flatten_feat)
            n = peak_indices.size
            for i, ind0 in enumerate(target_indices):
                if dont_have_channels_target[i]:
                    # this spike is not valid because do not have local chans
                    continue
                order = np.argsort(local_dists[i, :])
                final_nn = min(nn, n)
                order = order[:final_nn]

                indices0.append(np.full(final_nn, ind0, dtype='int64'))
                indices1.append(peak_indices[order])
                data.append(local_dists[i, order])

        else:
            raise ValueError("create_graph_from_peak_features wrong mode")
            
    indices0 = np.concatenate(indices0)
    indices1 = np.concatenate(indices1)
    data = np.concatenate(data)
    distances = csr_matrix((data, (indices0, indices1)), shape=(peaks.size, peaks.size))

    return distances
