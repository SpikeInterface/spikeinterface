
import numpy as np



from spikeinterface.sortingcomponents.clustering.tools import aggregate_sparse_features



def create_graph_from_peak_features(
    recording,
    peaks,
    peak_features,
    sparse_mask,
    peak_locations=None,
    bin_um=20.,
    dim=1,
    mode="full_connected_bin",
    direction="y",
    n_neighbors=20,
):
    """
    Create a sparse garph of peaks distances.
    This done using a binarization along the depth axis.
    Each peaks can connect to peaks of the same bin and neighbour bins.

    The distances are locally computed on a local sparse set of channels that depend on thev detph.
    So the original features sparsity must be big enougth to cover local channel (actual bin+neighbour).

    2 possible modes:
      * "full_connected_bin" : compute the distances from all peaks in a bin to all peaks in the same bin + neighbour
      * "knn" : keep the k neareast neighbour for each peaks in bin + neighbour
    
    Important, peak_locations can be:
      * the peak location from the channel (fast)
      * the estimated peak location
      * the corrected peak location if the peak_features is computed with motion_awre in mind

    Note : the binarization works for linear probe only. This need to be extended to 2d grid binarization for planar mea.
    """

    from scipy.sparse import csr_matrix
    from scipy.spatial.distance import cdist

    

    dim = "xyz".index(direction)
    channel_locations = recording.get_channel_locations()
    channel_depth = channel_locations[:, dim]

    if peak_locations is None:
        # we use the max channel location instead
        peak_depths = channel_depth[peaks["channel_index"]]
    else:
        peak_depths = peak_locations[direction]

    # todo center this bons like in peak localization
    loc0 = min(channel_depth)
    loc1 = max(channel_depth)
    eps = bin_um/10.
    bins = np.arange(loc0-bin_um/2, loc1+bin_um/2+eps, bin_um)


    indices0 = []
    indices1 = []
    data = []
    for b in range(bins.size-1):
        
        # limits for peaks
        l0, l1 = bins[b], bins[b+1]
        mask = (peak_depths> l0) & (peak_depths<= l1)

        # limits for features
        b0, b1 = l0 - bin_um, l1 + bin_um
        local_chans = np.flatnonzero((channel_locations[:, dim] > (b0 )) & (channel_locations[:, dim] <= (b1)))

        # print(l0, l1, b0, b1)

        mask = (peak_depths> b0) & (peak_depths<= b1)
        peak_indices = np.flatnonzero(mask)
        # print(peak_indices.size)

        local_depths = peak_depths[peak_indices]

        target_mask = (local_depths > l0) & (local_depths <= l1)
        target_indices = peak_indices[target_mask]

        # print(local_chans)
        if target_indices.size == 0:
            continue

        local_feats, dont_have_channels = aggregate_sparse_features(peaks, peak_indices,
                                                                 peak_features, sparse_mask, local_chans)
        # print(b)
        if np.sum(dont_have_channels) > 0:
            print("dont_have_channels", np.sum(dont_have_channels), "for n=", peak_indices.sze, "bin", b0, b1)
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
            # TODO use a better KNN tools (like pynndescent)
            # here we do brut force at the moment
            local_dists = cdist(flatten_feat[target_mask], flatten_feat)
            n = peak_indices.size
            for i, ind0 in enumerate(target_indices):
                if dont_have_channels_target[i]:
                    # this spike is not valid because do not have local chans
                    continue
                order = np.argsort(local_dists[i, :])
                final_nn = min(n_neighbors, n)
                order = order[:final_nn]

                indices0.append(np.full(final_nn, ind0, dtype='int64'))
                indices1.append(peak_indices[order])
                data.append(local_dists[i, order])

        else:
            raise ValueError("create_graph_from_peak_features() wrong mode")
    
    if len(indices0) > 0:
        indices0 = np.concatenate(indices0)
        indices1 = np.concatenate(indices1)
        data = np.concatenate(data)
    distances = csr_matrix((data, (indices0, indices1)), shape=(peaks.size, peaks.size))

    return distances
