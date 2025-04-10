import numpy as np

from tqdm.auto import tqdm

from spikeinterface.sortingcomponents.clustering.tools import aggregate_sparse_features


def create_graph_from_peak_features(
    recording,
    peaks,
    peak_features,
    sparse_mask,
    peak_locations=None,
    bin_mode="channels",
    neighbors_radius_um=60.0,
    # bin_mode="vertical_bins",
    bin_um=20.0,
    direction="xy",
    # sparse_mode="connected_to_all_neighbors",
    sparse_mode="knn",
    apply_local_svd=False,
    n_components=10,
    normed_distances=False,
    n_neighbors=20,
    ensure_symetric=False,
    enforce_diagonal_to_zero=True,
    progress_bar=True,
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

    import scipy.sparse
    from scipy.spatial.distance import cdist

    if sparse_mode == "knn":
        from sklearn.neighbors import NearestNeighbors
    if apply_local_svd:
        from sklearn.decomposition import TruncatedSVD

    channel_locations = recording.get_channel_locations()

    if bin_mode == "channels":

        dims = ["xyz".index(i) for i in direction]
        channel_locs = channel_locations[:, dims]
        channel_distances = np.linalg.norm(channel_locs[:, np.newaxis] - channel_locs[np.newaxis, :], axis=2)

        loop = []
        for channel_ind in range(channel_locations.shape[0]):
            local_chans = np.flatnonzero(channel_distances[channel_ind, :] <= neighbors_radius_um)
            neighbors_indices = np.flatnonzero(np.isin(peaks["channel_index"], local_chans))
            target_local_inds = np.flatnonzero(peaks["channel_index"][neighbors_indices] == channel_ind)
            loop.append((local_chans, neighbors_indices, target_local_inds))

    elif bin_mode == "vertical_bins":

        assert len(direction) == 1
        dim = "xyz".index(direction)
        channel_depth = channel_locations[:, dim]

        if peak_locations is None:
            # we use the max channel location instead
            peak_depths = channel_depth[peaks["channel_index"]]
        else:
            peak_depths = peak_locations[direction]

        loc0 = min(channel_depth)
        loc1 = max(channel_depth)
        eps = bin_um / 10.0
        bins = np.arange(loc0 - bin_um / 2, loc1 + bin_um / 2 + eps, bin_um)

        bins[0] = -np.inf
        bins[-1] = np.inf

        loop = []
        for b in range(bins.size - 1):

            # limits for peaks
            l0, l1 = bins[b], bins[b + 1]
            # mask = (peak_depths> l0) & (peak_depths<= l1)

            # limits for features
            b0, b1 = l0 - bin_um, l1 + bin_um
            local_chans = np.flatnonzero((channel_locations[:, dim] >= (b0)) & (channel_locations[:, dim] <= (b1)))

            # print(l0, l1, b0, b1)

            mask = (peak_depths >= b0) & (peak_depths < b1)
            neighbors_indices = np.flatnonzero(mask)
            # print(neighbors_indices.size)

            local_depths = peak_depths[neighbors_indices]

            target_mask = (local_depths >= l0) & (local_depths < l1)
            target_local_inds = np.flatnonzero(target_mask)
            # target_indices = neighbors_indices[target_local_inds]

            loop.append((local_chans, neighbors_indices, target_local_inds))

    else:
        raise ValueError("create_graph_from_peak_features : wrong bin_mode")

    if progress_bar:
        loop = tqdm(loop, desc=f"Build distance graph over {bin_mode}")

    local_graphs = []
    row_indices = []
    for local_chans, neighbors_indices, target_local_inds in loop:
        target_indices = neighbors_indices[target_local_inds]
        row_indices.append(target_indices)

        if target_indices.size == 0:
            continue

        local_feats, dont_have_channels = aggregate_sparse_features(
            peaks, neighbors_indices, peak_features, sparse_mask, local_chans
        )

        # if np.sum(dont_have_channels) > 0:
        #     print("dont_have_channels", np.sum(dont_have_channels), "for n=", neighbors_indices.size, "bin", b0, b1)

        flatten_feat = local_feats.reshape(local_feats.shape[0], -1)

        if apply_local_svd:
            if isinstance(n_components, int):
                n_components = min(n_components, flatten_feat.shape[1])
                tsvd = TruncatedSVD(n_components)
                flatten_feat = tsvd.fit_transform(flatten_feat)

            elif isinstance(n_components, float):
                assert 0 < n_components < 1, "n_components should be in ]0, 1["
                tsvd = TruncatedSVD(flatten_feat.shape[1])
                flatten_feat = tsvd.fit_transform(flatten_feat)
                n_explain = np.sum(np.cumsum(tsvd.explained_variance_ratio_) <= n_components) + 1
                flatten_feat = flatten_feat[:, :n_explain]

        if sparse_mode == "connected_to_all_neighbors":
            local_dists = cdist(flatten_feat[target_local_inds], flatten_feat)

            if normed_distances:
                norm = np.linalg.norm(flatten_feat, axis=1)
                local_dists /= norm[target_local_inds, None] + norm[None, :]

            data = local_dists.flatten().astype("float32")
            indptr = np.arange(0, local_dists.size + 1, local_dists.shape[1])
            indices = np.concatenate([neighbors_indices] * target_indices.size)
            local_graph = scipy.sparse.csr_matrix((data, indices, indptr), shape=(target_indices.size, peaks.size))
            local_graphs.append(local_graph)

        elif sparse_mode == "knn":
            # nn_tree = NearestNeighbors(n_neighbors=min(n_neighbors, target_local_inds.size), metric="minkowski", p=2) # euclidean
            nn_tree = NearestNeighbors(n_neighbors=min(n_neighbors, target_local_inds.size))  # euclidean
            nn_tree.fit(flatten_feat)
            local_sparse_dist = nn_tree.kneighbors_graph(flatten_feat[target_local_inds], mode="distance")

            # t1 = time.perf_counter()
            # print("knn", t1-t0)

            # remap to all columns
            # t0 = time.perf_counter()
            data = local_sparse_dist.data.astype("float32")
            indptr = local_sparse_dist.indptr
            if normed_distances:
                for i in range(local_sparse_dist.shape[0]):
                    src = flatten_feat[target_local_inds[i]]
                    a, b = indptr[i], indptr[i + 1]
                    tgt = flatten_feat[local_sparse_dist.indices[a:b]]
                    norm = np.linalg.norm(src) + np.linalg.norm(tgt, axis=1)
                    data[a:b] /= norm
            indices = neighbors_indices[local_sparse_dist.indices]
            local_graph = scipy.sparse.csr_matrix((data, indices, indptr), shape=(target_indices.size, peaks.size))

            # t1 = time.perf_counter()
            # print("make local sparse csr", t1-t0)

            local_graphs.append(local_graph)

        else:
            raise ValueError("create_graph_from_peak_features() wrong mode")

    # stack all local distances in a big sparse one
    if len(local_graphs) > 0:
        distances = scipy.sparse.vstack(local_graphs)
        row_indices = np.concatenate(row_indices)
        # print(np.unique(np.diff(row_indices)))
        row_order = np.argsort(row_indices)
        # print(np.unique(np.diff(row_indices[row_order])))

        # t0 = time.perf_counter()
        distances = distances[row_order]
        # t1 = time.perf_counter()
        # print("row_order", t1 - t0)

        if sparse_mode == "knn":
            # t0 = time.perf_counter()
            distances = scipy.sparse.csr_matrix(distances)
            # t1 = time.perf_counter()
            # print("final csr", t1 - t0)

        if enforce_diagonal_to_zero:
            ind0, ind1 = distances.tocoo().coords
            distances.data[ind0 == ind1] = 0.0

    else:
        distances = scipy.sparse.csr_matrix(shape=(peaks.size, peaks.size), dtype="float32")

    if ensure_symetric:
        # because of the way the graph is done the distance matrix could be not symetric
        # this trick force the symetry
        distances = distances.maximum(distances.T)

    return distances
