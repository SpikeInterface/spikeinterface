"""Cluster quality metrics computed from principal components."""

from __future__ import annotations


from copy import deepcopy

import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor

try:
    import scipy.stats
    import scipy.spatial.distance
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import IncrementalPCA
except:
    pass

import warnings

from .misc_metrics import compute_num_spikes, compute_firing_rates

from ..core import get_random_data_chunks, compute_sparsity
from ..core.template_tools import get_template_extremum_channel


_possible_pc_metric_names = [
    "isolation_distance",
    "l_ratio",
    "d_prime",
    "nearest_neighbor",
    "nn_isolation",
    "nn_noise_overlap",
    "silhouette",
]


_default_params = dict(
    nearest_neighbor=dict(
        max_spikes=10000,
        n_neighbors=5,
    ),
    nn_isolation=dict(
        max_spikes=10000, min_spikes=10, min_fr=0.0, n_neighbors=4, n_components=10, radius_um=100, peak_sign="neg"
    ),
    nn_noise_overlap=dict(
        max_spikes=10000, min_spikes=10, min_fr=0.0, n_neighbors=4, n_components=10, radius_um=100, peak_sign="neg"
    ),
    silhouette=dict(method=("simplified",)),
)


def get_quality_pca_metric_list():
    """Get a list of the available PCA-based quality metrics."""
    return deepcopy(_possible_pc_metric_names)


def calculate_pc_metrics(
    sorting_analyzer, metric_names=None, qm_params=None, unit_ids=None, seed=None, n_jobs=1, progress_bar=False
):
    """Calculate principal component derived metrics.

    Parameters
    ----------
    sorting_analyzer: SortingAnalyzer
        A SortingAnalyzer object
    metric_names : list of str, default: None
        The list of PC metrics to compute.
        If not provided, defaults to all PC metrics.
    qm_params : dict or None
        Dictionary with parameters for each PC metric function.
    unit_ids : list of int or None
        List of unit ids to compute metrics for.
    seed : int, default: None
        Random seed value.
    n_jobs : int
        Number of jobs to parallelize metric computations.
    progress_bar : bool
        If True, progress bar is shown.

    Returns
    -------
    pc_metrics : dict
        The computed PC metrics.
    """
    pca_ext = sorting_analyzer.get_extension("principal_components")
    assert pca_ext is not None, "calculate_pc_metrics() need extension 'principal_components'"

    sorting = sorting_analyzer.sorting

    if metric_names is None:
        metric_names = _possible_pc_metric_names
    if qm_params is None:
        qm_params = _default_params

    extremum_channels = get_template_extremum_channel(sorting_analyzer)

    if unit_ids is None:
        unit_ids = sorting_analyzer.unit_ids
    channel_ids = sorting_analyzer.channel_ids

    # create output dict of dict  pc_metrics['metric_name'][unit_id]
    pc_metrics = {k: {} for k in metric_names}
    if "nearest_neighbor" in metric_names:
        pc_metrics.pop("nearest_neighbor")
        pc_metrics["nn_hit_rate"] = {}
        pc_metrics["nn_miss_rate"] = {}

    if "nn_isolation" in metric_names:
        pc_metrics["nn_unit_id"] = {}

    # Compute nspikes and firing rate outside of main loop for speed
    if any([n in metric_names for n in ["nn_isolation", "nn_noise_overlap"]]):
        n_spikes_all_units = compute_num_spikes(sorting_analyzer, unit_ids=unit_ids)
        fr_all_units = compute_firing_rates(sorting_analyzer, unit_ids=unit_ids)
    else:
        n_spikes_all_units = None
        fr_all_units = None

    run_in_parallel = n_jobs > 1

    if run_in_parallel:
        parallel_functions = []

    # this get dense projection for selected unit_ids
    dense_projections, spike_unit_indices = pca_ext.get_some_projections(channel_ids=None, unit_ids=unit_ids)
    all_labels = sorting.unit_ids[spike_unit_indices]

    items = []
    for unit_id in unit_ids:
        if sorting_analyzer.is_sparse():
            neighbor_channel_ids = sorting_analyzer.sparsity.unit_id_to_channel_ids[unit_id]
            neighbor_unit_ids = [
                other_unit for other_unit in unit_ids if extremum_channels[other_unit] in neighbor_channel_ids
            ]
        else:
            neighbor_channel_ids = channel_ids
            neighbor_unit_ids = unit_ids
        neighbor_channel_indices = sorting_analyzer.channel_ids_to_indices(neighbor_channel_ids)

        labels = all_labels[np.isin(all_labels, neighbor_unit_ids)]
        pcs = dense_projections[np.isin(all_labels, neighbor_unit_ids)][:, :, neighbor_channel_indices]
        pcs_flat = pcs.reshape(pcs.shape[0], -1)

        func_args = (
            pcs_flat,
            labels,
            metric_names,
            unit_id,
            unit_ids,
            qm_params,
            seed,
            n_spikes_all_units,
            fr_all_units,
        )
        items.append(func_args)

    if not run_in_parallel:
        units_loop = enumerate(unit_ids)
        if progress_bar:
            units_loop = tqdm(units_loop, desc="calculate_pc_metrics", total=len(unit_ids))

        for unit_ind, unit_id in units_loop:
            pca_metrics_unit = pca_metrics_one_unit(items[unit_ind])
            for metric_name, metric in pca_metrics_unit.items():
                pc_metrics[metric_name][unit_id] = metric
    else:
        with ProcessPoolExecutor(n_jobs) as executor:
            results = executor.map(pca_metrics_one_unit, items)
            if progress_bar:
                results = tqdm(results, total=len(unit_ids), desc="calculate_pc_metrics")

            for ui, pca_metrics_unit in enumerate(results):
                unit_id = unit_ids[ui]
                for metric_name, metric in pca_metrics_unit.items():
                    pc_metrics[metric_name][unit_id] = metric

    return pc_metrics


#################################################################
# Code from spikemetrics


def mahalanobis_metrics(all_pcs, all_labels, this_unit_id):
    """Calculates isolation distance and L-ratio (metrics computed from Mahalanobis distance)

    Parameters
    ----------
    all_pcs : 2d array
        The PCs for all spikes, organized as [num_spikes, PCs].
    all_labels : 1d array
        The cluster labels for all spikes. Must have length of number of spikes.
    this_unit_id : int
        The ID for the unit to calculate these metrics for.

    Returns
    -------
    isolation_distance : float
        Isolation distance of this unit.
    l_ratio : float
        L-ratio for this unit.

    References
    ----------
    Based on metrics described in [Schmitzer-Torbert]_
    """

    pcs_for_this_unit = all_pcs[all_labels == this_unit_id, :]
    pcs_for_other_units = all_pcs[all_labels != this_unit_id, :]

    mean_value = np.expand_dims(np.mean(pcs_for_this_unit, 0), 0)

    try:
        VI = np.linalg.inv(np.cov(pcs_for_this_unit.T))
    except np.linalg.linalg.LinAlgError:
        # case of singular matrix
        return np.nan, np.nan

    mahalanobis_other = np.sort(scipy.spatial.distance.cdist(mean_value, pcs_for_other_units, "mahalanobis", VI=VI)[0])

    mahalanobis_self = np.sort(scipy.spatial.distance.cdist(mean_value, pcs_for_this_unit, "mahalanobis", VI=VI)[0])

    # number of spikes
    n = np.min([pcs_for_this_unit.shape[0], pcs_for_other_units.shape[0]])

    if n >= 2:
        dof = pcs_for_this_unit.shape[1]  # number of features
        l_ratio = np.sum(1 - scipy.stats.chi2.cdf(pow(mahalanobis_other, 2), dof)) / mahalanobis_self.shape[0]
        isolation_distance = pow(mahalanobis_other[n - 1], 2)
        # if math.isnan(l_ratio):
        #     print("NaN detected", mahalanobis_other, VI)
    else:
        l_ratio = np.nan
        isolation_distance = np.nan

    return isolation_distance, l_ratio


def lda_metrics(all_pcs, all_labels, this_unit_id):
    """Calculates d-prime based on Linear Discriminant Analysis.

    Parameters
    ----------
    all_pcs : 2d array
        The PCs for all spikes, organized as [num_spikes, PCs].
    all_labels : 1d array
        The cluster labels for all spikes. Must have length of number of spikes.
    this_unit_id : int
        The ID for the unit to calculate these metrics for.

    Returns
    -------
    d_prime : float
        D prime measure of this unit.

    References
    ----------
    Based on metric described in [Hill]_
    """

    X = all_pcs

    y = np.zeros((X.shape[0],), dtype="bool")
    y[all_labels == this_unit_id] = True

    lda = LinearDiscriminantAnalysis(n_components=1)

    X_flda = lda.fit_transform(X, y)

    flda_this_cluster = X_flda[np.where(y)[0]]
    flda_other_cluster = X_flda[np.where(np.invert(y))[0]]

    d_prime = (np.mean(flda_this_cluster) - np.mean(flda_other_cluster)) / np.sqrt(
        0.5 * (np.std(flda_this_cluster) ** 2 + np.std(flda_other_cluster) ** 2)
    )

    return d_prime


def nearest_neighbors_metrics(all_pcs, all_labels, this_unit_id, max_spikes, n_neighbors):
    """
    Calculates unit contamination based on NearestNeighbors search in PCA space.

    Parameters
    ----------
    all_pcs : 2d array
        The PCs for all spikes, organized as [num_spikes, PCs].
    all_labels : 1d array
        The cluster labels for all spikes. Must have length of number of spikes.
    this_unit_id : int
        The ID for the unit to calculate these metrics for.
    max_spikes : int
        The number of spikes to use, per cluster.
        Note that the calculation can be very slow when this number is >20000.
    n_neighbors : int
        The number of neighbors to use.

    Returns
    -------
    hit_rate : float
        Fraction of neighbors for target cluster that are also in target cluster.
    miss_rate : float
        Fraction of neighbors outside target cluster that are in target cluster.

    Notes
    -----
    A is a (hopefully) representative subset of cluster X

    .. math::

        NN_hit(X) = 1/k \\sum_i=1^k |{{x in A such that ith closest neighbor is in X}}| / \\|A\\|

    References
    ----------
    Based on metrics described in [Chung]_
    """

    total_spikes = all_pcs.shape[0]
    ratio = max_spikes / total_spikes

    # if no other units in the vicinity, return best possible option
    if len(np.unique(all_labels)) == 1:
        warnings.warn(f"No other units found in the vicinity of {this_unit}. Setting nn_hit_rate=1 and nn_miss_rate=0")
        return 1.0, 0.0

    this_unit = all_labels == this_unit_id
    this_unit_pcs = all_pcs[this_unit, :]
    other_units_pcs = all_pcs[np.invert(this_unit), :]
    X = np.concatenate((this_unit_pcs, other_units_pcs), 0)

    num_obs_this_unit = np.sum(this_unit)

    if ratio < 1:
        inds = np.arange(0, X.shape[0] - 1, 1 / ratio).astype("int")
        X = X[inds, :]
        num_obs_this_unit = int(num_obs_this_unit * ratio)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(X)
    distances, indices = nbrs.kneighbors(X)

    this_cluster_nearest = indices[:num_obs_this_unit, 1:].flatten()
    other_cluster_nearest = indices[num_obs_this_unit:, 1:].flatten()

    hit_rate = np.mean(this_cluster_nearest < num_obs_this_unit)
    miss_rate = np.mean(other_cluster_nearest < num_obs_this_unit)

    return hit_rate, miss_rate


def nearest_neighbors_isolation(
    sorting_analyzer,
    this_unit_id: int | str,
    n_spikes_all_units: dict = None,
    fr_all_units: dict = None,
    max_spikes: int = 1000,
    min_spikes: int = 10,
    min_fr: float = 0.0,
    n_neighbors: int = 5,
    n_components: int = 10,
    radius_um: float = 100,
    peak_sign: str = "neg",
    min_spatial_overlap: float = 0.5,
    seed=None,
):
    """Calculates unit isolation based on NearestNeighbors search in PCA space.

    Parameters
    ----------
    sorting_analyzer: SortingAnalyzer
        A SortingAnalyzer object
    this_unit_id : int | str
        The ID for the unit to calculate these metrics for.
    n_spikes_all_units: dict, default: None
        Dictionary of the form ``{<unit_id>: <n_spikes>}`` for the waveform extractor.
        Recomputed if None.
    fr_all_units: dict, default: None
        Dictionary of the form ``{<unit_id>: <firing_rate>}`` for the waveform extractor.
        Recomputed if None.
    max_spikes : int, default: 1000
        Max number of spikes to use per unit.
    min_spikes : int, default: 10
        Min number of spikes a unit must have to go through with metric computation.
        Units with spikes < min_spikes gets numpy.NaN as the quality metric,
        and are ignored when selecting other units' neighbors.
    min_fr : float, default: 0.0
        Min firing rate a unit must have to go through with metric computation.
        Units with firing rate < min_fr gets numpy.NaN as the quality metric,
        and are ignored when selecting other units' neighbors.
    n_neighbors : int, default: 5
        Number of neighbors to check membership of.
    n_components : int, default: 10
        The number of PC components to use to project the snippets to.
    radius_um : float, default: 100
        The radius, in um, that channels need to be within the peak channel to be included.
    peak_sign: "neg" | "pos" | "both", default: "neg"
        The peak_sign used to compute sparsity and neighbor units. Used if sorting_analyzer
        is not sparse already.
    min_spatial_overlap : float, default: 100
        In case sorting_analyzer is sparse, other units are selected if they share at least
        `min_spatial_overlap` times `n_target_unit_channels` with the target unit
    seed : int, default: None
        Seed for random subsampling of spikes.

    Returns
    -------
    nn_isolation : float
        The calculation nearest neighbor isolation metric for `this_unit_id`.
        If the unit has fewer than `min_spikes`, returns numpy.NaN instead.
    nn_unit_id : np.int16
        Id of the "nearest neighbor" unit (unit with lowest isolation score from `this_unit_id`)

    Notes
    -----
    The overall logic of this approach is:

    #. Choose a cluster
    #. Compute the isolation score with every other cluster
    #. Isolation score is defined as the min of 2. (i.e. 'worst-case measure')

    The implementation of this approach is:

    Let A and B be two clusters from sorting.

    We set \\|A\\| = \\|B\\|:

        * | If max_spikes < \\|A\\| and max_spikes < \\|B\\|:
          |     Then randomly subsample max_spikes samples from A and B.
        * | If max_spikes > min(\\|A\\|, \\|B\\|) (e.g. \\|A\\| > max_spikes > \\|B\\|):
          |     Then randomly subsample min(\\|A\\|, \\|B\\|) samples from A and B.

    This is because the metric is affected by the size of the clusters being compared
    independently of how well-isolated they are.

    We also restrict the waveforms to channels with significant signal.

    See docstring for `_compute_isolation` for the definition of isolation score.

    References
    ----------
    Based on isolation metric described in [Chung]_
    """
    rng = np.random.default_rng(seed=seed)

    waveforms_ext = sorting_analyzer.get_extension("waveforms")
    assert waveforms_ext is not None, "nearest_neighbors_isolation() need extension 'waveforms'"

    sorting = sorting_analyzer.sorting
    all_units_ids = sorting.get_unit_ids()
    if n_spikes_all_units is None:
        n_spikes_all_units = compute_num_spikes(sorting_analyzer)
    if fr_all_units is None:
        fr_all_units = compute_firing_rates(sorting_analyzer)

    # if target unit has fewer than `min_spikes` spikes, print out a warning and return NaN
    if n_spikes_all_units[this_unit_id] < min_spikes:
        warnings.warn(
            f"Unit {this_unit_id} has fewer spikes than specified by `min_spikes` "
            f"({min_spikes}); returning NaN as the quality metric..."
        )
        return np.nan, np.nan
    elif fr_all_units[this_unit_id] < min_fr:
        warnings.warn(
            f"Unit {this_unit_id} has a firing rate below the specified `min_fr` "
            f"({min_fr} Hz); returning NaN as the quality metric..."
        )
        return np.nan, np.nan
    else:
        # first remove the units with too few spikes
        unit_ids_to_keep = np.array(
            [
                unit
                for unit in all_units_ids
                if (n_spikes_all_units[unit] >= min_spikes and fr_all_units[unit] >= min_fr)
            ]
        )
        sorting = sorting.select_units(unit_ids=unit_ids_to_keep)

        all_units_ids = sorting.get_unit_ids()
        other_units_ids = np.setdiff1d(all_units_ids, this_unit_id)

        # get waveforms of target unit
        # waveforms_target_unit = sorting_analyzer.get_waveforms(unit_id=this_unit_id)
        waveforms_target_unit = waveforms_ext.get_waveforms_one_unit(unit_id=this_unit_id, force_dense=False)

        n_spikes_target_unit = waveforms_target_unit.shape[0]

        # find units whose signal channels (i.e. channels inside some radius around
        # the channel with largest amplitude) overlap with signal channels of the target unit
        if sorting_analyzer.is_sparse():
            sparsity = sorting_analyzer.sparsity
        else:
            sparsity = compute_sparsity(sorting_analyzer, method="radius", peak_sign=peak_sign, radius_um=radius_um)
        closest_chans_target_unit = sparsity.unit_id_to_channel_indices[this_unit_id]
        n_channels_target_unit = len(closest_chans_target_unit)
        # select other units that have a minimum spatial overlap with target unit
        other_units_ids = [
            unit_id
            for unit_id in other_units_ids
            if np.sum(np.isin(sparsity.unit_id_to_channel_indices[unit_id], closest_chans_target_unit))
            >= (n_channels_target_unit * min_spatial_overlap)
        ]

        # if no unit is within neighborhood of target unit, then just say isolation is 1 (best possible)
        if not other_units_ids:
            nn_isolation = 1
            nn_unit_id = np.nan
        # if there are units to compare, then compute isolation with each
        else:
            isolation = np.zeros(
                len(other_units_ids),
            )
            for other_unit_id in other_units_ids:
                # waveforms_other_unit = sorting_analyzer.get_waveforms(unit_id=other_unit_id)
                waveforms_other_unit = waveforms_ext.get_waveforms_one_unit(unit_id=other_unit_id, force_dense=False)

                n_spikes_other_unit = waveforms_other_unit.shape[0]
                closest_chans_other_unit = sparsity.unit_id_to_channel_indices[other_unit_id]
                n_snippets = np.min([n_spikes_target_unit, n_spikes_other_unit, max_spikes])

                # make the two clusters equal in terms of: number of spikes & channels with signal
                waveforms_target_unit_idx = rng.choice(n_spikes_target_unit, size=n_snippets, replace=False)
                waveforms_target_unit_sampled = waveforms_target_unit[waveforms_target_unit_idx]
                waveforms_other_unit_idx = rng.choice(n_spikes_other_unit, size=n_snippets, replace=False)
                waveforms_other_unit_sampled = waveforms_other_unit[waveforms_other_unit_idx]

                # project this unit and other unit waveforms on common subspace
                common_channel_idxs = np.intersect1d(closest_chans_target_unit, closest_chans_other_unit)
                if sorting_analyzer.is_sparse():
                    # in this case, waveforms are sparse so we need to do some smart indexing
                    waveforms_target_unit_sampled = waveforms_target_unit_sampled[
                        :, :, np.isin(closest_chans_target_unit, common_channel_idxs)
                    ]
                    waveforms_other_unit_sampled = waveforms_other_unit_sampled[
                        :, :, np.isin(closest_chans_other_unit, common_channel_idxs)
                    ]
                else:
                    waveforms_target_unit_sampled = waveforms_target_unit_sampled[:, :, common_channel_idxs]
                    waveforms_other_unit_sampled = waveforms_other_unit_sampled[:, :, common_channel_idxs]

                # compute principal components after concatenation
                all_snippets = np.concatenate(
                    [
                        waveforms_target_unit_sampled.reshape((n_snippets, -1)),
                        waveforms_other_unit_sampled.reshape((n_snippets, -1)),
                    ],
                    axis=0,
                )
                pca = IncrementalPCA(n_components=n_components)
                pca.partial_fit(all_snippets)
                projected_snippets = pca.transform(all_snippets)

                # compute isolation
                isolation[other_unit_id == other_units_ids] = _compute_isolation(
                    projected_snippets[:n_snippets, :], projected_snippets[n_snippets:, :], n_neighbors
                )
            # isolation metric is the minimum of the pairwise isolations
            # nn_unit_id is the unit with lowest isolation score
            nn_isolation = np.min(isolation)
            nn_unit_id = other_units_ids[np.argmin(isolation)]

        return nn_isolation, nn_unit_id


def nearest_neighbors_noise_overlap(
    sorting_analyzer,
    this_unit_id: int | str,
    n_spikes_all_units: dict = None,
    fr_all_units: dict = None,
    max_spikes: int = 1000,
    min_spikes: int = 10,
    min_fr: float = 0.0,
    n_neighbors: int = 5,
    n_components: int = 10,
    radius_um: float = 100,
    peak_sign: str = "neg",
    seed=None,
):
    """Calculates unit noise overlap based on NearestNeighbors search in PCA space.

    Parameters
    ----------
    sorting_analyzer: SortingAnalyzer
        A SortingAnalyzer object
    this_unit_id : int | str
        The ID of the unit to calculate this metric on.
    n_spikes_all_units: dict, default: None
        Dictionary of the form ``{<unit_id>: <n_spikes>}`` for the waveform extractor.
        Recomputed if None.
    fr_all_units: dict, default: None
        Dictionary of the form ``{<unit_id>: <firing_rate>}`` for the waveform extractor.
        Recomputed if None.
    max_spikes : int, default: 1000
        The max number of spikes to use per cluster.
    min_spikes : int, default: 10
        Min number of spikes a unit must have to go through with metric computation.
        Units with spikes < min_spikes gets numpy.NaN as the quality metric.
    min_fr : float, default: 0.0
        Min firing rate a unit must have to go through with metric computation.
        Units with firing rate < min_fr gets numpy.NaN as the quality metric.
    n_neighbors : int, default: 5
        The number of neighbors to check membership.
    n_components : int, default: 10
        The number of PC components to use to project the snippets to.
    radius_um : float, default: 100
        The radius, in um, that channels need to be within the peak channel to be included.
    peak_sign: "neg" | "pos" | "both", default: "neg"
        The peak_sign used to compute sparsity and neighbor units. Used if sorting_analyzer
        is not sparse already.
    seed : int, default: 0
        Random seed for subsampling spikes.

    Returns
    -------
    nn_noise_overlap : float
        The computed nearest neighbor noise estimate.
        If the unit has fewer than `min_spikes`, returns numpy.NaN instead.

    Notes
    -----
    The general logic of this measure is:

    1. Generate a noise cluster by randomly sampling voltage snippets from recording.
    2. Subtract projection onto the weighted average of noise snippets
       of both the target and noise clusters to correct for bias in sampling.
    3. Compute the isolation score between the noise cluster and the target cluster.

    As with nn_isolation, the clusters that are compared (target and noise clusters)
    have the same number of spikes.

    See docstring for `_compute_isolation` for the definition of isolation score.

    References
    ----------
    Based on noise overlap metric described in [Chung]_
    """
    rng = np.random.default_rng(seed=seed)

    waveforms_ext = sorting_analyzer.get_extension("waveforms")
    assert waveforms_ext is not None, "nearest_neighbors_isolation() need extension 'waveforms'"

    templates_ext = sorting_analyzer.get_extension("templates")
    assert templates_ext is not None, "nearest_neighbors_isolation() need extension 'templates'"

    if n_spikes_all_units is None:
        n_spikes_all_units = compute_num_spikes(sorting_analyzer)
    if fr_all_units is None:
        fr_all_units = compute_firing_rates(sorting_analyzer)

    # if target unit has fewer than `min_spikes` spikes, print out a warning and return NaN
    if n_spikes_all_units[this_unit_id] < min_spikes:
        warnings.warn(
            f"Unit {this_unit_id} has fewer spikes than specified by `min_spikes` "
            f"({min_spikes}); returning NaN as the quality metric..."
        )
        return np.nan
    elif fr_all_units[this_unit_id] < min_fr:
        warnings.warn(
            f"Unit {this_unit_id} has a firing rate below the specified `min_fr` "
            f"({min_fr} Hz); returning NaN as the quality metric...",
        )
        return np.nan
    else:
        # get random snippets from the recording to create a noise cluster
        nsamples = waveforms_ext.nbefore + waveforms_ext.nafter
        recording = sorting_analyzer.recording
        noise_cluster = get_random_data_chunks(
            recording,
            return_scaled=sorting_analyzer.return_scaled,
            num_chunks_per_segment=max_spikes,
            chunk_size=nsamples,
            seed=seed,
        )
        noise_cluster = np.reshape(noise_cluster, (max_spikes, nsamples, -1))

        # get waveforms for target cluster
        # waveforms = sorting_analyzer.get_waveforms(unit_id=this_unit_id).copy()
        waveforms = waveforms_ext.get_waveforms_one_unit(unit_id=this_unit_id, force_dense=False).copy()

        # adjust the size of the target and noise clusters to be equal
        if waveforms.shape[0] > max_spikes:
            wf_ind = rng.choice(waveforms.shape[0], max_spikes, replace=False)
            waveforms = waveforms[wf_ind]
            n_snippets = max_spikes
        elif waveforms.shape[0] < max_spikes:
            noise_ind = rng.choice(noise_cluster.shape[0], waveforms.shape[0], replace=False)
            noise_cluster = noise_cluster[noise_ind]
            n_snippets = waveforms.shape[0]
        else:
            n_snippets = max_spikes

        # restrict to channels with significant signal
        if sorting_analyzer.is_sparse():
            sparsity = sorting_analyzer.sparsity
        else:
            sparsity = compute_sparsity(sorting_analyzer, method="radius", peak_sign=peak_sign, radius_um=radius_um)
        noise_cluster = noise_cluster[:, :, sparsity.unit_id_to_channel_indices[this_unit_id]]

        # compute weighted noise snippet (Z)
        # median_waveform = sorting_analyzer.get_template(unit_id=this_unit_id, mode="median")
        all_templates = templates_ext.get_data(operator="median")
        this_unit_index = sorting_analyzer.sorting.id_to_index(this_unit_id)
        median_waveform = all_templates[this_unit_index, :, :]

        # in case sorting_analyzer is sparse, waveforms and templates are already sparse
        if not sorting_analyzer.is_sparse():
            # @alessio : this next line is suspicious because the waveforms is already sparse no ? Am i wrong ?
            waveforms = waveforms[:, :, sparsity.unit_id_to_channel_indices[this_unit_id]]
            median_waveform = median_waveform[:, sparsity.unit_id_to_channel_indices[this_unit_id]]

        tmax, chmax = np.unravel_index(np.argmax(np.abs(median_waveform)), median_waveform.shape)
        weights = [noise_clip[tmax, chmax] for noise_clip in noise_cluster]
        weights = np.asarray(weights)
        weights = weights / np.sum(weights)
        weighted_noise_snippet = np.sum(weights * noise_cluster.swapaxes(0, 2), axis=2).swapaxes(0, 1)

        # subtract projection onto weighted noise snippet
        for snippet in range(n_snippets):
            waveforms[snippet, :, :] = _subtract_clip_component(waveforms[snippet, :, :], weighted_noise_snippet)
            noise_cluster[snippet, :, :] = _subtract_clip_component(
                noise_cluster[snippet, :, :], weighted_noise_snippet
            )

        # compute principal components after concatenation
        all_snippets = np.concatenate(
            [waveforms.reshape((n_snippets, -1)), noise_cluster.reshape((n_snippets, -1))], axis=0
        )
        pca = IncrementalPCA(n_components=n_components)
        pca.partial_fit(all_snippets)
        projected_snippets = pca.transform(all_snippets)

        # compute overlap
        nn_noise_overlap = 1 - _compute_isolation(
            projected_snippets[:n_snippets, :], projected_snippets[n_snippets:, :], n_neighbors
        )

        return nn_noise_overlap


def simplified_silhouette_score(all_pcs, all_labels, this_unit_id):
    """Calculates the simplified silhouette score for each cluster. The value ranges
    from -1 (bad clustering) to 1 (good clustering). The simplified silhoutte score
    utilizes the centroids for distance calculations rather than pairwise calculations.

    Parameters
    ----------
    all_pcs : 2d array
        The PCs for all spikes, organized as [num_spikes, PCs].
    all_labels : 1d array
        The cluster labels for all spikes. Must have length of number of spikes.
    this_unit_id : int
        The ID for the unit to calculate this metric for.

    Returns
    -------
    unit_silhouette_score : float
        Simplified Silhouette Score for this unit

    References
    ----------
    Based on simplified silhouette score suggested by [Hruschka]_
    """

    pcs_for_this_unit = all_pcs[all_labels == this_unit_id, :]
    centroid_for_this_unit = np.expand_dims(np.mean(pcs_for_this_unit, 0), 0)
    distances_for_this_unit = scipy.spatial.distance.cdist(centroid_for_this_unit, pcs_for_this_unit)
    distance = np.inf

    # find centroid of other cluster and measure distances to that rather than pairwise
    # if less than current minimum distance update
    for label in np.unique(all_labels):
        if label != this_unit_id:
            pcs_for_other_cluster = all_pcs[all_labels == label, :]
            centroid_for_other_cluster = np.expand_dims(np.mean(pcs_for_other_cluster, 0), 0)
            distances_for_other_cluster = scipy.spatial.distance.cdist(centroid_for_other_cluster, pcs_for_this_unit)
            mean_distance_for_other_cluster = np.mean(distances_for_other_cluster)
            if mean_distance_for_other_cluster < distance:
                distance = mean_distance_for_other_cluster
                distances_for_minimum_cluster = distances_for_other_cluster

    sil_distances = (distances_for_minimum_cluster - distances_for_this_unit) / np.maximum(
        distances_for_minimum_cluster, distances_for_this_unit
    )

    unit_silhouette_score = np.mean(sil_distances)
    return unit_silhouette_score


def silhouette_score(all_pcs, all_labels, this_unit_id):
    """Calculates the silhouette score which is a marker of cluster quality ranging from
    -1 (bad clustering) to 1 (good clustering). Distances are all calculated as pairwise
    comparisons of all data points.

    Parameters
    ----------
    all_pcs : 2d array
        The PCs for all spikes, organized as [num_spikes, PCs].
    all_labels : 1d array
        The cluster labels for all spikes. Must have length of number of spikes.
    this_unit_id : int
        The ID for the unit to calculate this metric for.

    Returns
    -------
    unit_silhouette_score : float
        Silhouette Score for this unit

    References
    ----------
    Based on [Rousseeuw]_
    """

    pcs_for_this_unit = all_pcs[all_labels == this_unit_id, :]
    distances_for_this_unit = scipy.spatial.distance.cdist(pcs_for_this_unit, pcs_for_this_unit)
    distance = np.inf

    # iterate through all other clusters and do pairwise distance comparisons
    # if current cluster distances < current mimimum update
    for label in np.unique(all_labels):
        if label != this_unit_id:
            pcs_for_other_cluster = all_pcs[all_labels == label, :]
            distances_for_other_cluster = scipy.spatial.distance.cdist(pcs_for_other_cluster, pcs_for_this_unit)
            mean_distance_for_other_cluster = np.mean(distances_for_other_cluster)
            if mean_distance_for_other_cluster < distance:
                distance = mean_distance_for_other_cluster
                distances_for_minimum_cluster = distances_for_other_cluster

    sil_distances = (distances_for_minimum_cluster - distances_for_this_unit) / np.maximum(
        distances_for_minimum_cluster, distances_for_this_unit
    )

    unit_silhouette_score = np.mean(sil_distances)
    return unit_silhouette_score


def _subtract_clip_component(clip1, component):
    V1 = clip1.flatten()
    V2 = component.flatten()
    V1 = V1 - V2 * np.dot(V1, V2) / np.dot(V2, V2)

    return V1.reshape(clip1.shape)


def _compute_isolation(pcs_target_unit, pcs_other_unit, n_neighbors: int):
    """
    Computes the isolation score used for nn_isolation and nn_noise_overlap

    Parameters
    ----------
    pcs_target_unit : 2d array
        PCA projection of the spikes in the target cluster, as [n_spikes, n_components].
    pcs_other_unit : 2d array
        PCA projection of the spikes in the other cluster, as [n_spikes, n_components].
    n_neighbors : int
        The number of nearest neighbors to check membership of.

    Returns
    -------
    isolation : float
        The computed isolation score.

    Notes
    -----
    Definition of isolation score:
        Isolation(A, B) = 1/k \\sum_{j=1}^k |{x \\in A U B: \rho(x)=\rho(jth nearest neighbor of x)}| / |A U B|
            where \rho(x) is the cluster x belongs to (in this case, either A or B)
    Note that this definition implies that the isolation score:
        (1) ranges from 0 to 1; and
        (2) is symmetric, i.e. Isolation(A, B) = Isolation(B, A)
    """

    # get lengths
    n_spikes_target = pcs_target_unit.shape[0]
    n_spikes_other = pcs_other_unit.shape[0]

    # concatenate
    pcs_concat = np.concatenate((pcs_target_unit, pcs_other_unit), axis=0)
    label_concat = np.concatenate((np.zeros(n_spikes_target), np.ones(n_spikes_other)))

    # if n_neighbors is greater than the number of spikes in both clusters, set it to max possible
    if n_neighbors > len(label_concat):
        n_neighbors_adjusted = len(label_concat) - 1
    else:
        n_neighbors_adjusted = n_neighbors

    _, membership_ind = (
        NearestNeighbors(n_neighbors=n_neighbors_adjusted, algorithm="auto").fit(pcs_concat).kneighbors()
    )

    target_nn_in_target = np.sum(label_concat[membership_ind[:n_spikes_target]] == 0)
    other_nn_in_other = np.sum(label_concat[membership_ind[n_spikes_target:]] == 1)

    isolation = (target_nn_in_target + other_nn_in_other) / (n_spikes_target + n_spikes_other) / n_neighbors_adjusted

    return isolation


def pca_metrics_one_unit(args):
    (
        pcs_flat,
        labels,
        metric_names,
        unit_id,
        unit_ids,
        qm_params,
        seed,
        # we_folder,
        n_spikes_all_units,
        fr_all_units,
    ) = args

    # if "nn_isolation" in metric_names or "nn_noise_overlap" in metric_names:
    #     we = load_waveforms(we_folder)

    pc_metrics = {}
    # metrics
    if "isolation_distance" in metric_names or "l_ratio" in metric_names:
        try:
            isolation_distance, l_ratio = mahalanobis_metrics(pcs_flat, labels, unit_id)
        except:
            isolation_distance = np.nan
            l_ratio = np.nan
        if "isolation_distance" in metric_names:
            pc_metrics["isolation_distance"] = isolation_distance
        if "l_ratio" in metric_names:
            pc_metrics["l_ratio"] = l_ratio

    if "d_prime" in metric_names:
        if len(unit_ids) == 1:
            d_prime = np.nan
        else:
            try:
                d_prime = lda_metrics(pcs_flat, labels, unit_id)
            except:
                d_prime = np.nan
        pc_metrics["d_prime"] = d_prime

    if "nearest_neighbor" in metric_names:
        try:
            nn_hit_rate, nn_miss_rate = nearest_neighbors_metrics(
                pcs_flat, labels, unit_id, **qm_params["nearest_neighbor"]
            )
        except:
            nn_hit_rate = np.nan
            nn_miss_rate = np.nan
        pc_metrics["nn_hit_rate"] = nn_hit_rate
        pc_metrics["nn_miss_rate"] = nn_miss_rate

    if "nn_isolation" in metric_names:
        try:
            nn_isolation, nn_unit_id = nearest_neighbors_isolation(
                we,
                unit_id,
                seed=seed,
                n_spikes_all_units=n_spikes_all_units,
                fr_all_units=fr_all_units,
                **qm_params["nn_isolation"],
            )
        except:
            nn_isolation = np.nan
            nn_unit_id = np.nan
        pc_metrics["nn_isolation"] = nn_isolation
        pc_metrics["nn_unit_id"] = nn_unit_id

    if "nn_noise_overlap" in metric_names:
        try:
            nn_noise_overlap = nearest_neighbors_noise_overlap(
                we,
                unit_id,
                n_spikes_all_units=n_spikes_all_units,
                fr_all_units=fr_all_units,
                seed=seed,
                **qm_params["nn_noise_overlap"],
            )
        except:
            nn_noise_overlap = np.nan
        pc_metrics["nn_noise_overlap"] = nn_noise_overlap

    if "silhouette" in metric_names:
        silhouette_method = qm_params["silhouette"]["method"]
        if "simplified" in silhouette_method:
            try:
                unit_silhouette_score = simplified_silhouette_score(pcs_flat, labels, unit_id)
            except:
                unit_silhouette_score = np.nan
            pc_metrics["silhouette"] = unit_silhouette_score
        if "full" in silhouette_method:
            try:
                unit_silhouette_score = silhouette_score(pcs_flat, labels, unit_id)
            except:
                unit_silhouette_score = np.nan
            pc_metrics["silhouette_full"] = unit_silhouette_score

    return pc_metrics
