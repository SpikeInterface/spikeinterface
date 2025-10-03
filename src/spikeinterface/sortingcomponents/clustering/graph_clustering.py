from pathlib import Path

import numpy as np


from spikeinterface.sortingcomponents.waveforms.peak_svd import extract_peaks_svd
from spikeinterface.sortingcomponents.clustering.graph_tools import create_graph_from_peak_features


class GraphClustering:
    """
    Simple clustering by constructing a global sparse graph using local slinding bins along the probe.

    The edge of the graph is constructed using local distance bewteen svd on waveforms.

    Then a classic algorithm like louvain or hdbscan is used.
    """

    name = "graph-clustering"
    need_noise_levels = False
    _default_params = {
        "peaks_svd": {"n_components": 5, "ms_before": 0.5, "ms_after": 1.5, "radius_um": 100.0, "motion": None},
        "seed": None,
        "graph_kwargs": dict(
            bin_mode="channels",
            neighbors_radius_um=50.0,
            # bin_mode="vertical_bins",
            # bin_um=30.,
            # direction="y",
            normed_distances=True,
            # n_neighbors=15,
            n_neighbors=5,
            # n_components=0.8,
            n_components=10,
            sparse_mode="knn",
            # sparse_mode="connected_to_all_neighbors"
            apply_local_svd=True,
            enforce_diagonal_to_zero=True,
        ),
        "clusterer": dict(
            method="sknetwork-louvain",
            # min_samples=1,
            # core_dist_n_jobs=-1,
            # min_cluster_size=20,
            # cluster_selection_method='leaf',
            # allow_single_cluster=True,
            # cluster_selection_epsilon=0.1
        ),
        "debug_folder": None,
        "verbose": True,
    }

    params_doc = """
        peaks_svd : params for peak SVD features extraction.
        See spikeinterface.sortingcomponents.waveforms.peak_svd.extract_peaks_svd
                        for more details.,
        seed : Random seed for reproducibility.,
        merge_from_templates : params for the merging step based on templates. See
                 spikeinterface.sortingcomponents.clustering.merging_tools.merge_peak_labels_from_templates
                 for more details.,
        merge_from_features : params for the merging step based on features. See
                    spikeinterface.sortingcomponents.clustering.merging_tools.merge_peak_labels_from_features
                    for more details.,
        debug_folder : If not None, a folder path where to save debug information.,
        verbose : If True, print information during the process.
    """

    @classmethod
    def main_function(cls, recording, peaks, params, job_kwargs=dict()):

        peaks_svd = params["peaks_svd"].copy()
        radius_um = peaks_svd["radius_um"]
        motion = peaks_svd["motion"]
        seed = params["seed"]
        verbose = params["verbose"]
        clustering_kwargs = params["clusterer"].copy()
        graph_kwargs = params["graph_kwargs"].copy()

        motion_aware = motion is not None
        peaks_svd.update(motion_aware=motion_aware)

        if seed is not None:
            peaks_svd.update(seed=seed)

        if graph_kwargs["bin_mode"] == "channels":
            assert radius_um >= graph_kwargs["neighbors_radius_um"] * 2
        elif graph_kwargs["bin_mode"] == "vertical_bins":
            assert radius_um >= graph_kwargs["bin_um"] * 3

        peaks_svd, sparse_mask, svd_model = extract_peaks_svd(
            recording,
            peaks,
            job_kwargs=job_kwargs,
            **peaks_svd,
        )

        # some method need a symetric matrix
        clustering_method = clustering_kwargs.pop("method")
        assert clustering_method in [
            "networkx-louvain",
            "sknetwork-louvain",
            "sknetwork-leiden",
            "leidenalg",
            "hdbscan",
        ]

        ensure_symetric = clustering_method in ("hdbscan",)

        distances = create_graph_from_peak_features(
            recording,
            peaks,
            peaks_svd,
            sparse_mask,
            ensure_symetric=ensure_symetric,
            **graph_kwargs,
        )

        if clustering_method == "networkx-louvain":
            # using networkx : very slow (possible backend with cude  backend="cugraph",)
            import networkx as nx

            distances_bool = distances.copy()
            distances_bool.data[:] = 1
            G = nx.Graph(distances_bool)
            communities = nx.community.louvain_communities(G, seed=seed, **clustering_kwargs)
            peak_labels = np.zeros(peaks.size, dtype=int)
            peak_labels[:] = -1
            k = 0
            for community in communities:
                if len(community) == 1:
                    continue
                peak_labels[list(community)] = k
                k += 1

        elif clustering_method == "sknetwork-louvain":
            from sknetwork.clustering import Louvain

            classifier = Louvain(**clustering_kwargs)
            distances_bool = distances.copy()
            distances_bool.data[:] = 1
            peak_labels = classifier.fit_predict(distances_bool)
            _remove_small_cluster(peak_labels, min_size=1)

        elif clustering_method == "sknetwork-leiden":
            from sknetwork.clustering import Leiden

            classifier = Leiden(**clustering_kwargs)
            distances_bool = distances.copy()
            distances_bool.data[:] = 1
            peak_labels = classifier.fit_predict(distances_bool)
            _remove_small_cluster(peak_labels, min_size=1)

        elif clustering_method == "leidenalg":
            import leidenalg
            import igraph

            adjacency = distances.copy()
            adjacency.data = 1.0 - adjacency.data
            graph = igraph.Graph.Weighted_Adjacency(
                adjacency.tocoo(),
                mode="directed",
            )
            clusters = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition)
            peak_labels = np.array(clusters.membership)
            _remove_small_cluster(peak_labels, min_size=1)

        elif clustering_method == "hdbscan":
            from hdbscan import HDBSCAN
            import scipy.sparse

            n_graph, connected_labels = scipy.sparse.csgraph.connected_components(distances, directed=False)
            peak_labels = np.zeros(peaks.size, dtype="int64")
            peak_labels[:] = -1

            label_count = 0
            for g in range(n_graph):
                connected_nodes = np.flatnonzero(connected_labels == g)
                if len(connected_nodes) == 1:
                    continue

                local_dist = distances[connected_nodes, :].tocsc()[:, connected_nodes].tocsr()

                clusterer = HDBSCAN(metric="precomputed", **clustering_kwargs)
                local_labels = clusterer.fit_predict(local_dist)

                valid_clusters = np.flatnonzero(local_labels >= 0)
                if valid_clusters.size:
                    peak_labels[connected_nodes[valid_clusters]] = local_labels[valid_clusters] + label_count
                    label_count += max(np.max(local_labels), 0)

        else:
            raise ValueError("GraphClustering : wrong clustering_method")

        labels_set = np.unique(peak_labels)
        labels_set = labels_set[labels_set >= 0]

        more_outs = dict(
            svd_model=svd_model,
            peaks_svd=peaks_svd,
            peak_svd_sparse_mask=sparse_mask,
        )
        return labels_set, peak_labels, more_outs


def _remove_small_cluster(peak_labels, min_size=1):
    for k in np.unique(peak_labels):
        inds = np.flatnonzero(peak_labels == k)
        if inds.size <= min_size:
            peak_labels[inds] = -1
