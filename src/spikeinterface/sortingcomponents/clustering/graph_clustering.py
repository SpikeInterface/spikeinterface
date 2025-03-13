from pathlib import Path

import numpy as np


from spikeinterface.sortingcomponents.clustering.peak_svd import extract_peaks_svd
from spikeinterface.sortingcomponents.clustering.graph_tools import create_graph_from_peak_features


class GraphClustering:
    """
    Simple clustering by constructing a global sparse graph using local slinding bins along the probe.

    The edge of the graph is constructed using local distance bewteen svd on waveforms.

    Then a classic algorithm like louvain or hdbscan is used.
    """

    _default_params = {
        "radius_um": 180.,
        "bin_um": 60.,
        "motion": None,
        "seed": None,
        "n_neighbors": 100,
        # "clustering_method": "leidenalg",
        "clustering_method": "sknetwork-leiden",
    }

    @classmethod
    def main_function(cls, recording, peaks, params, job_kwargs=dict()):

        radius_um = params["radius_um"]
        bin_um = params["bin_um"]
        motion = params["motion"]
        seed = params["seed"]
        n_neighbors = params["n_neighbors"]
        clustering_method = params["clustering_method"]
        normed_distances = params["normed_distances"]

        motion_aware = motion is not None

        assert radius_um >= bin_um * 3

        

        peaks_svd, sparse_mask, _ = extract_peaks_svd(
            recording, peaks,
            radius_um=radius_um,
            motion_aware=motion_aware,
            motion=None,
        )
        # print(peaks_svd.shape)



        channel_locations = recording.get_channel_locations()
        channel_depth = channel_locations[:, 1]
        peak_depths = channel_depth[peaks["channel_index"]]

        # order peaks by depth
        order = np.argsort(peak_depths)
        ordered_peaks = peaks[order]
        ordered_peaks_svd = peaks_svd[order]

        # TODO : try to use real peak location

        distances = create_graph_from_peak_features(
            recording,
            ordered_peaks,
            ordered_peaks_svd,
            sparse_mask,
            peak_locations=None,
            bin_um=bin_um,
            dim=1,
            # mode="full_connected_bin",
            mode="knn",
            direction="y",
            n_neighbors=n_neighbors,
            apply_local_svd=True,
            # apply_local_svd=False,
            n_components=8,
            normed_distances=True,
        )

        # print(distances)
        # print(distances.shape)
        # print("sparsity: ", distances.indices.size / (distances.shape[0]**2))        

        distances_bool = distances.copy()
        distances_bool.data[:] = 1

        print("clustering_method", clustering_method)

        if clustering_method == "networkx-louvain":
            # using networkx : very slow (possible backend with cude  backend="cugraph",)
            import networkx as nx
            G = nx.Graph(distances_bool)
            communities = nx.community.louvain_communities(G, seed=seed)
            peak_labels = np.zeros(ordered_peaks.size, dtype=int)
            peak_labels[:] = -1
            k = 0
            for community in communities:
                if len(community) == 1:
                    continue
                peak_labels[list(community)] = k
                k += 1
        
        elif clustering_method == "sknetwork-louvain":
            from sknetwork.clustering import Louvain
            classifier = Louvain()
            peak_labels = classifier.fit_predict(distances_bool)
            _remove_small_cluster(peak_labels, min_size=1)

        elif clustering_method == "sknetwork-leiden":
            from sknetwork.clustering import Leiden
            classifier = Leiden()
            peak_labels = classifier.fit_predict(distances_bool)
            _remove_small_cluster(peak_labels, min_size=1)

        elif clustering_method == "leidenalg":
            import leidenalg
            import igraph
            graph = igraph.Graph.Weighted_Adjacency(distances_bool.tocoo(), mode='directed',)
            clusters = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition)
            peak_labels = np.array(clusters.membership)
            _remove_small_cluster(peak_labels, min_size=1)

        elif clustering_method == "hdbscan":
            from hdbscan import HDBSCAN
            # from fast_hdbscan import HDBSCAN

            import scipy.sparse

            
            # need to make subgraph
            n_graph, connected_labels = scipy.sparse.csgraph.connected_components(distances)

            # print(np.unique(connected_labels))
            # print("n_graph", n_graph)
            peak_labels = np.zeros(ordered_peaks.size, dtype='int64')
            peak_labels[:] = -1

            label_count = 0
            for g in range(n_graph):
                rows = np.flatnonzero(connected_labels == g)
                if len(rows) == 1:
                    continue
                
                local_dist = distances[rows, :][:, rows]

                has_neibhor = np.array(np.sum(local_dist>0, axis=1) > 1)
                has_neibhor = has_neibhor[:, 0]

                rows = rows[has_neibhor]
                local_dist = distances[rows, :][:, rows]

                # make symetric
                local_dist = local_dist.maximum(local_dist.T)

                # # print(local_dist)
                # res = hdbscan.hdbscan(local_dist, metric="precomputed")
                # print(res[0])
                # fig, ax = plt.subplots()
                # ax.matshow(local_dist)
                max_dist=(local_dist.max() - local_dist.min()) * 1000

                clusterer = HDBSCAN(
                    metric="precomputed",
                    min_cluster_size=50,
                    cluster_selection_method= "eom",
                    allow_single_cluster= True,
                    metric_params=dict(
                        max_distance=max_dist,
                    ),
                    core_dist_n_jobs=-1,
                )
                local_labels = clusterer.fit_predict(local_dist)

                mask = local_labels>=0
                if np.sum(mask):

                    peak_labels[rows[mask]] = local_labels[mask] + label_count

                    label_count += max(np.max(local_labels), 0)


        else:
            raise ValueError("GraphClustering : wrong clustering_method")

        labels_set = np.unique(peak_labels)
        labels_set = labels_set[labels_set >= 0]

        # we need to reorder labels
        reverse_order = np.argsort(order)
        peak_labels = peak_labels[reverse_order]
        
        return labels_set, peak_labels



def _remove_small_cluster(peak_labels, min_size=1):
    for k in np.unique(peak_labels):
        inds = np.flatnonzero(peak_labels == k)
        if inds.size <= min_size:
            peak_labels[inds] = -1            

