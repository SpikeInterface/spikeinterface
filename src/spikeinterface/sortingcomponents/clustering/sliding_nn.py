from __future__ import annotations

try:
    import numba

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

import numpy as np
from sklearn.utils import check_random_state

try:
    from pynndescent import NNDescent

    HAVE_NNDESCENT = True
except ImportError:
    HAVE_NNDESCENT = False

from spikeinterface.core import get_channel_distances
from tqdm.auto import tqdm

try:
    import hdbscan

    HAVE_HDBSCAN = True
except:
    HAVE_HDBSCAN = False
import copy
from scipy.sparse import coo_matrix

try:
    import pymde

    HAVE_PYMDE = True
except ImportError:
    HAVE_PYMDE = False

try:
    import torch

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

import datetime

from spikeinterface.core.waveform_tools import extract_waveforms_to_buffers


class SlidingNNClustering:
    """
    Sliding window nearest neighbor clustering.
    """

    _default_params = {
        "time_window_s": 5,
        "hdbscan_kwargs": {"min_cluster_size": 20, "allow_single_cluster": True},
        "margin_ms": 100,
        "ms_before": 1,
        "ms_after": 1,
        "n_channel_neighbors": 8,
        "n_neighbors": 5,
        "embedding_dim": None,
        "low_memory": True,
        "waveform_mode": "shared_memory",
        "mde_negative_to_positive_samples": 5,
        "mde_device": "cpu",
        "create_embedding": True,
        "cluster_embedding": True,
        "debug": False,
        "tmp_folder": None,
        "verbose": False,
        "tmp_folder": None,
        "job_kwargs": {"n_jobs": -1},
    }

    @classmethod
    def _initialize_folder(cls, recording, peaks, params):
        assert HAVE_NUMBA, "SlidingNN needs numba to work"
        assert HAVE_TORCH, "SlidingNN needs torch to work"
        assert HAVE_NNDESCENT, "SlidingNN needs pynndescent to work"
        assert HAVE_PYMDE, "SlidingNN needs pymde to work"
        assert HAVE_HDBSCAN, "SlidingNN needs hdbscan to work"

        d = params
        tmp_folder = params["tmp_folder"]

        num_chans = recording.channel_ids.size

        # important sparsity is 2 times radius sparsity because closest channel will be 1 time radius
        chan_distances = get_channel_distances(recording)
        sparsity_mask = np.zeros((num_chans, num_chans), dtype="bool")
        for c in range(num_chans):
            (chans,) = np.nonzero(chan_distances[c, :] <= (2 * d["radius_um"]))
            sparsity_mask[c, chans] = True

        # create a new peak vector to extract waveforms
        dtype = [("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")]
        peaks2 = np.zeros(peaks.size, dtype=dtype)
        peaks2["sample_index"] = peaks["sample_index"]
        peaks2["unit_index"] = peaks["channel_index"]
        peaks2["segment_index"] = peaks["segment_index"]

        fs = recording.get_sampling_frequency()
        dtype = recording.get_dtype()

        nbefore = int(d["ms_before"] * fs / 1000.0)
        nafter = int(d["ms_after"] * fs / 1000.0)

        if tmp_folder is None:
            wf_folder = None
        else:
            wf_folder = tmp_folder / "waveforms_pre_cluster"
            wf_folder.mkdir()

        ids = np.arange(num_chans, dtype="int64")
        wfs_arrays = extract_waveforms_to_buffers(
            recording,
            peaks2,
            ids,
            nbefore,
            nafter,
            mode=d["waveform_mode"],
            return_scaled=False,
            folder=wf_folder,
            dtype=dtype,
            sparsity_mask=sparsity_mask,
            copy=(d["waveform_mode"] == "shared_memory"),
            **d["job_kwargs"],
        )

        return wfs_arrays, sparsity_mask

    @classmethod
    def main_function(cls, recording, peaks, params):
        d = params

        # wfs_arrays, sparsity_mask, noise = cls._initialize_folder(recording, peaks, params)

        # prepare neighborhood parameters
        fs = recording.get_sampling_frequency()
        n_frames = recording.get_num_frames()
        duration = n_frames / fs
        time_window_frames = fs * d["time_window_s"]
        margin_frames = int(d["margin_ms"] / 1000 * fs)
        spike_pre_frames = int(d["ms_before"] / 1000 * fs)
        spike_post_frames = int(d["ms_after"] / 1000 * fs)
        n_channels = recording.get_num_channels()
        n_samples = spike_pre_frames + spike_post_frames

        if d["embedding_dim"] is None:
            d["embedding_dim"] = recording.get_num_channels()

        # get channel distances from one another
        channel_distance = get_channel_distances(recording)

        # get nearest neighbors of channels
        channel_neighbors = np.argsort(channel_distance, axis=1)[:, : d["n_channel_neighbors"]]

        # divide the recording into chunks of time_window_s seconds
        n_chunks = int(np.ceil(n_frames / time_window_frames))
        chunk_start_spike_idxs = np.zeros(n_chunks)
        chunk_end_spike_idxs = np.zeros(n_chunks)

        # prepare an array of nn indices and distances of shape (n_spikes, 2, n_neighbors)
        # because the sliding window computes neighbors for each chunk twice, the first
        # the first row are n_neighbors for neighbors corresponding to the first computation
        # and the second row are the n_neighbors corresponding to the second computation. e.g.
        # --[data stream]
        # |A|B| <- W1 (window 1)
        # .  |A|B| <- W2
        # .    |A|B| <- W3
        # in this array, neighbors are given as:
        # [XXXX][W1_B][W2_B][W3_B]
        # [W1_A][W2_A][W3_A][XXXX]
        n_spikes = len(peaks)
        nn_index_array = np.zeros((n_spikes, 2, d["n_neighbors"]), dtype=int) - 1
        nn_distance_array = np.zeros((n_spikes, 2, d["n_neighbors"]), dtype=float)

        # initialize empty array of embeddings
        if d["create_embedding"]:
            embeddings_all = np.zeros((n_spikes, d["embedding_dim"]))
            if d["cluster_embedding"]:
                # create an empty array of clusters and probabilities (from overlapping
                # chunks)
                clusters = np.zeros((len(peaks), 2), dtype=int) - 1
                cluster_probabilities = np.zeros((len(peaks), 2), dtype=float)

        # for each chunk grab spike nearest neighbors
        # TODO: this can be parallelized (although nearest neighbors is already
        #    parallelized, and bandwidth is limited for reading from raw data)
        end_last = -1
        explore = range(n_chunks - 1)
        if d["verbose"]:
            explore = tqdm(explore, desc="chunk")

        for chunk in explore:
            # set the start and end frame to grab for this chunk
            start_frame = int(chunk * time_window_frames)
            end_frame = int((chunk + 2) * time_window_frames)
            if end_frame > n_frames:
                end_frame = n_frames

            if d["verbose"]:
                print("Extracting waveforms: {}".format(datetime.datetime.now()))
            # grab all spikes
            all_spikes, all_chan_idx, peaks_in_chunk_idx = get_chunk_spike_waveforms(
                recording,
                start_frame,
                end_frame,
                peaks,
                channel_neighbors,
                spike_pre_frames=spike_pre_frames,
                spike_post_frames=spike_post_frames,
                n_channel_neighbors=d["n_channel_neighbors"],
                margin_frames=margin_frames,
            )
            chunk_start_spike_idxs[chunk] = peaks_in_chunk_idx[0]
            chunk_end_spike_idxs[chunk] = peaks_in_chunk_idx[-1]
            idx_next = peaks_in_chunk_idx > end_last
            idx_cur = peaks_in_chunk_idx <= end_last

            if d["verbose"]:
                print("Computing nearest neighbors: {}".format(datetime.datetime.now()))
            # grab nearest neighbors
            knn_indices, knn_distances = get_spike_nearest_neighbors(
                all_spikes,
                all_chan_idx=all_chan_idx,
                n_samples=spike_post_frames + spike_pre_frames,
                n_neighbors=d["n_neighbors"],
                n_channel_neighbors=d["n_channel_neighbors"],
                low_memory=d["low_memory"],
                knn_verbose=d["verbose"],
                n_jobs=d["job_kwargs"]["n_jobs"],
            )
            # remove the first nearest neighbor (which should be self)
            knn_distances = knn_distances[:, 1:]
            knn_indices = knn_indices[:, 1:]
            # get the absolute index (spike number)
            knn_indices_abs = peaks_in_chunk_idx[knn_indices]

            # put new neighbors in first row
            nn_index_array[peaks_in_chunk_idx[idx_next], 0] = knn_indices_abs[idx_next]
            # put overlapping neighbors from previous in second row
            nn_index_array[peaks_in_chunk_idx[idx_cur], 1] = knn_indices_abs[idx_cur]

            # repeat for distances
            nn_distance_array[peaks_in_chunk_idx[idx_next], 0] = knn_distances[idx_next]
            nn_distance_array[peaks_in_chunk_idx[idx_cur], 1] = knn_distances[idx_cur]
            # double up on neighbors in the beginning, since we don't have an overlap
            if chunk == 0:
                nn_index_array[peaks_in_chunk_idx, 1] = knn_indices
                nn_distance_array[peaks_in_chunk_idx, 1] = knn_distances
            # double up neighbors in the end, since we only sample these once
            if chunk == n_chunks - 1:
                nn_index_array[peaks_in_chunk_idx[idx_next], 1] = knn_indices[idx_next]
                # repeat for distances
                nn_distance_array[peaks_in_chunk_idx[idx_next], 1] = knn_distances[idx_next]

            # create embedding
            if d["create_embedding"]:
                if d["verbose"]:
                    print("Computing MDE embeddings: {}".format(datetime.datetime.now()))

                chunk_csr = construct_symmetric_graph_from_idx_vals(knn_indices, knn_distances)
                # number of current embeddings
                n_cur = np.sum(idx_cur)

                # if this is the first chunk, embed a new graph
                # otherwise, embed a graph with fixed embedding locations
                # from the previous embedding. This encourages stationary
                # embeddings over time and provides additional info from
                # the graph of the previous chunk into this chunk (through
                # embedding points.
                if chunk == 0:
                    embeddings_chunk = embed_graph(
                        chunk_csr,
                        prev_embeddings=None,
                        prev_idx=None,
                        mde_device=d["mde_device"],
                        embedding_dim=d["embedding_dim"],
                        negative_to_positive_samples=d["mde_negative_to_positive_samples"],
                    )
                else:
                    embeddings_chunk = embed_graph(
                        chunk_csr,
                        prev_embeddings=embeddings_all[peaks_in_chunk_idx[idx_cur]],
                        prev_idx=np.arange(n_cur),
                        mde_device=d["mde_device"],
                        embedding_dim=d["embedding_dim"],
                        negative_to_positive_samples=d["mde_negative_to_positive_samples"],
                    )
                embeddings_all[peaks_in_chunk_idx[idx_next]] = embeddings_chunk[n_cur:]

                # cluster embedding
                if d["cluster_embedding"]:
                    print(
                        "Clustering MDE embeddings (n={}): {}".format(embeddings_chunk.shape, datetime.datetime.now())
                    )
                    # TODO HDBSCAN can be done on GPU with NVIDIA RAPIDS for speed
                    clusterer = hdbscan.HDBSCAN(
                        prediction_data=True,
                        core_dist_n_jobs=d["job_kwargs"]["n_jobs"],
                        **d["hdbscan_kwargs"],
                    ).fit(embeddings_chunk)

                    # set cluster labels and probabilities for this chunk
                    # put new clusters in first row
                    clusters[peaks_in_chunk_idx[peaks_in_chunk_idx > end_last], 0] = clusterer.labels_[
                        peaks_in_chunk_idx > end_last
                    ]
                    # put overlapping neighbors from previous in second row
                    clusters[peaks_in_chunk_idx[peaks_in_chunk_idx <= end_last], 1] = clusterer.labels_[
                        peaks_in_chunk_idx <= end_last
                    ]
                    # repeat for cluster probabilities
                    cluster_probabilities[peaks_in_chunk_idx[peaks_in_chunk_idx > end_last], 0] = (
                        clusterer.probabilities_[peaks_in_chunk_idx > end_last]
                    )
                    # put overlapping neighbors from previous in second row
                    cluster_probabilities[peaks_in_chunk_idx[peaks_in_chunk_idx <= end_last], 1] = (
                        clusterer.probabilities_[peaks_in_chunk_idx <= end_last]
                    )

                    # TODO retrieve templates for each cluster

            end_last = peaks_in_chunk_idx[-1]

        peak_labels = clusters

        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]

        # return (
        #     nn_index_array,
        #     nn_distance_array,
        #     embeddings_all,
        #     clusters,
        #     cluster_probabilities,
        #     chunk_start_spike_idxs,
        #     chunk_end_spike_idxs
        # )

        return labels, peak_labels


if HAVE_NUMBA:

    @numba.jit(nopython=True, fastmath=True, cache=False)
    def sparse_euclidean(x, y, n_samples, n_dense):
        """Euclidean distance metric over sparse vectors, where first n_dense
        elements are indices, and n_samples is the length of the second dimension
        """
        # break out sparse into columns and data
        x_best = x[:n_dense]  # dense indices
        x = x[n_dense:]
        y_best = y[:n_dense]
        y = y[n_dense:]
        result = 0.0

        xi = 0
        for xb in x_best:
            calc = False
            yi = 0
            for yb in y_best:
                if xb == yb:
                    calc = True
                    # calculate euclidean
                    for i in range(n_samples):
                        result += (x[xi * n_samples + i] - y[yi * n_samples + i]) ** 2

                yi += 1
            if calc == False:
                # add x squared
                for i in range(n_samples):
                    result += x[xi * n_samples + i] ** 2
            xi += 1
        yi = 0
        for yb in y_best:
            calc = False
            for xb in x_best:
                if xb == yb:
                    calc = True
            if calc == False:
                # add y squared
                for i in range(n_samples):
                    result += y[yi * n_samples + i] ** 2

            yi += 1
        return np.sqrt(result)


# HACK: this function only exists because I couldn't get the spikeinterface one to work...
def retrieve_padded_trace(recording, start_frame, end_frame, margin_frames, channel_ids=None):
    """Grabs a chunk of recording trace, with padding"""
    n_frames = recording.get_num_frames()
    # get the padding
    _pre = np.max([0, start_frame - margin_frames])
    _post = np.min([n_frames, end_frame + margin_frames])

    traces = recording.get_traces(start_frame=_pre, end_frame=_post, channel_ids=channel_ids)
    # append zeros if this chunk exists near the border
    if _pre < margin_frames:
        traces = np.vstack([np.zeros((margin_frames - _pre, traces.shape[1])), traces])
    if _post < margin_frames:
        traces = np.vstack([traces, np.zeros((margin_frames - _post, traces.shape[1]))])
    return traces


def get_chunk_spike_waveforms(
    recording,
    start_frame,
    end_frame,
    peaks,
    channel_neighbors,
    spike_pre_frames=30,
    spike_post_frames=30,
    n_channel_neighbors=5,
    margin_frames=3000,
):
    """Grabs the spike waveforms for a chunk of a recording."""
    # grab the trace
    traces = retrieve_padded_trace(recording, start_frame, end_frame, margin_frames, channel_ids=None)

    # find the peaks that exist in this sample
    peaks_in_chunk_mask = (peaks["sample_index"] >= start_frame) & (peaks["sample_index"] <= end_frame)

    # get the peaks in this chunk
    peaks_chunk = peaks[peaks_in_chunk_mask]
    # get the index of which peaks are in this chunk
    peaks_in_chunk_idx = np.where(peaks_in_chunk_mask)[0]
    if len(peaks_in_chunk_idx) == 0:
        return None

    # add peaks indices to list

    # prepare an array of spikes to populate (n_spikes * channels * frames)
    all_spikes = np.zeros(
        (
            len(peaks_chunk),
            n_channel_neighbors,
            spike_pre_frames + spike_post_frames,
        )
    )
    # prepare an array of channels
    all_chan_idx = np.zeros((len(peaks_chunk), n_channel_neighbors))

    # for each spike in the sample, add it to the
    for spike_i, (sample_index, channel_index, amplitude, segment_index) in enumerate(peaks_chunk):
        spike_start = sample_index + margin_frames - spike_pre_frames - start_frame
        spike_end = sample_index + margin_frames + spike_post_frames - start_frame
        all_spikes[spike_i] = traces[spike_start:spike_end, channel_neighbors[channel_index]].T
        all_chan_idx[spike_i] = channel_neighbors[channel_index]

    return all_spikes, all_chan_idx, peaks_in_chunk_idx


def get_spike_nearest_neighbors(
    all_spikes,
    n_neighbors,
    low_memory,
    knn_verbose,
    all_chan_idx,
    n_samples=50,
    n_channel_neighbors=8,
    n_jobs=1,
    max_candidates=60,
):
    """Builds a graph of nearest neighbors from sparse spike array.
    TODO: There are potentially faster ANN approaches. e.g.
        https://github.com/facebookresearch/pysparnn
    """

    # helper functions for nearest-neighbors search tree
    def get_n_trees_iters(X):
        n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
        n_iters = max(5, int(round(np.log2(X.shape[0]))))
        return n_trees, n_iters

    def swap_elements(l, idx1, idx2):
        i1 = l[idx1]
        i2 = l[idx2]
        l[idx1] = i2
        l[idx2] = i1
        return l

    # flatten spikes
    all_spikes_flat = np.reshape(all_spikes, (len(all_spikes), np.product(all_spikes.shape[1:])))

    # add channel indices to channel values (for graph construction)
    all_spikes_flat = np.hstack([all_chan_idx, all_spikes_flat])

    # get parameters for NN search tree
    n_trees, n_iters = get_n_trees_iters(all_spikes_flat)

    # we are using the default nearest neighbors approach from UMAP
    #   with a sparse euclidean metric approach over channels
    #   Q: is euclidean the right metric?
    knn_search_index = NNDescent(
        all_spikes_flat,
        metric=sparse_euclidean,
        metric_kwds={"n_samples": n_samples, "n_dense": n_channel_neighbors},
        n_neighbors=n_neighbors + 1,
        random_state=check_random_state(None),
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=max_candidates,
        low_memory=low_memory,
        n_jobs=n_jobs,
        verbose=knn_verbose,
        compressed=False,
    )
    knn_indices, knn_distances = knn_search_index.neighbor_graph

    # BUG: nndescent finding some elements which are not closest to themselves.
    #   they are always second closest to themselves, and the distance is computed to be 0
    #   very small proportion of events
    # HACK: switch back errors
    # computed neighbors where an element is *not* closest to itself
    nn_errors = np.where(knn_indices[:, 0] != np.arange(len(knn_indices)))[0]
    # correct by swapping
    for nn_error in nn_errors:
        correct_match = np.where(knn_indices[nn_error] == nn_error)[0][0]
        knn_indices[nn_error] = swap_elements(knn_indices[nn_error], correct_match, 0)

    return knn_indices, knn_distances


def merge_nn_dicts(peaks, n_neighbors, peaks_in_chunk_idx_list, knn_indices_list, knn_distances_list):
    """Merge together peaks_in_chunk_idx_list and knn_indices_list to build final graph."""

    nn_index_array = np.zeros((len(peaks), n_neighbors * 2), dtype=int) - 1
    nn_distance_array = np.zeros((len(peaks), n_neighbors * 2), dtype=float)
    end_last = -1
    # for each nn graph
    for idxi, (peaks_in_chunk_idx, knn_indices, knn_distances) in enumerate(
        zip(peaks_in_chunk_idx_list, knn_indices_list, knn_distances_list)
    ):
        # put new neighbors in first 5 rows
        nn_index_array[peaks_in_chunk_idx[peaks_in_chunk_idx > end_last], :n_neighbors] = knn_indices[
            peaks_in_chunk_idx > end_last
        ]
        # put overlapping neighbors from previous
        nn_index_array[peaks_in_chunk_idx[peaks_in_chunk_idx <= end_last], n_neighbors:] = knn_indices[
            peaks_in_chunk_idx <= end_last
        ]

        # repeat for distances
        nn_distance_array[peaks_in_chunk_idx[peaks_in_chunk_idx > end_last], :n_neighbors] = knn_distances[
            peaks_in_chunk_idx > end_last
        ]
        nn_distance_array[peaks_in_chunk_idx[peaks_in_chunk_idx <= end_last], n_neighbors:] = knn_distances[
            peaks_in_chunk_idx <= end_last
        ]
        # double up neighbors the beginning, since we only sample these once

        if idxi == 0:
            nn_index_array[peaks_in_chunk_idx, n_neighbors:] = knn_indices
            nn_distance_array[peaks_in_chunk_idx, n_neighbors:] = knn_distances
        # double up neighbors in the end, since we only sample these once
        if idxi == len(peaks_in_chunk_idx_list) - 1:
            nn_index_array[peaks_in_chunk_idx[peaks_in_chunk_idx > end_last], n_neighbors:] = knn_indices[
                peaks_in_chunk_idx > end_last
            ]
            # repeat for distances
            nn_distance_array[peaks_in_chunk_idx[peaks_in_chunk_idx > end_last], n_neighbors:] = knn_distances[
                peaks_in_chunk_idx > end_last
            ]

        end_last = peaks_in_chunk_idx[-1]
    return nn_index_array, nn_distance_array


def construct_symmetric_graph_from_idx_vals(graph_idx, graph_vals):
    rows = graph_idx.flatten()
    cols = np.repeat(np.arange(len(graph_idx)), graph_idx.shape[1])
    rows_ = np.concatenate([rows, cols])
    cols_ = np.concatenate([cols, rows])
    vals = graph_vals.flatten()

    # matrix
    # TODO: rather than weighting at 1, we can compute local distance
    #   using the UMAP algorithm:
    #   https://github.com/lmcinnes/umap/blob/master/umap/umap_.py#L575
    chunk_csr = coo_matrix(
        (np.ones(len(rows_)), (rows_, cols_)),
        shape=(len(graph_idx), len(graph_idx)),
    ).tocsr()
    return chunk_csr


def embed_graph(
    chunk_csr, prev_embeddings=None, prev_idx=None, negative_to_positive_samples=5, embedding_dim=2, mde_device="cuda"
):
    # graph size
    n_items = chunk_csr.shape[1]

    # grate graph as pymde graph
    knn_graph = pymde.Graph(chunk_csr)

    # constrain embeddings
    if prev_embeddings is not None:
        anchor_constraint = pymde.Anchored(
            anchors=torch.tensor(prev_idx, device=mde_device),
            values=torch.tensor(prev_embeddings, dtype=torch.float32, device=mde_device),
        )
    else:
        anchor_constraint = None

    # initialize with quadratic embedding
    quadratic_mde = pymde.MDE(
        n_items=n_items,
        embedding_dim=embedding_dim,
        edges=knn_graph.edges,
        distortion_function=pymde.penalties.Quadratic(knn_graph.weights),
        constraint=anchor_constraint,
        device=mde_device,
    )

    # embed quadratic initialization
    x = quadratic_mde.embed(verbose=True).cpu()

    # get all nearest neighbor edges
    similar_edges = knn_graph.edges

    # sample a set of dissimilar edges
    n_dis = int(len(similar_edges) * negative_to_positive_samples)
    dissimilar_edges = pymde.preprocess.dissimilar_edges(n_items=n_items, num_edges=n_dis, similar_edges=similar_edges)
    # created a list of weights for similar and dissimilar edges
    edges = torch.cat([similar_edges, dissimilar_edges])
    weights = torch.cat([knn_graph.weights, -1.0 * torch.ones(dissimilar_edges.shape[0])])

    # create a distortion penalty
    f = pymde.penalties.PushAndPull(
        weights=weights,
        attractive_penalty=pymde.penalties.Log1p,
        repulsive_penalty=pymde.penalties.Log,
    )

    # prepare MDE object
    std_mde = pymde.MDE(
        n_items=n_items,
        embedding_dim=embedding_dim,
        edges=edges,
        distortion_function=f,
        constraint=anchor_constraint,
        # constraint=pymde.Standardized(),
        device=mde_device,
    )

    # embed MDE
    x = std_mde.embed(quadratic_mde.X, max_iter=400, verbose=True)

    if mde_device == "cuda":
        x = np.array(x.cpu())

    return x
