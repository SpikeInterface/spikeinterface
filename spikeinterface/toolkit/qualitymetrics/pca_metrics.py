from collections import namedtuple

import numpy as np
import scipy.stats
import scipy.spatial.distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors

from ..postprocessing import WaveformPrincipalComponent

_possible_pc_metric_names = ['isolation_distance', 'l_ratio', 'd_prime', 'nearest_neighbor']


def calculate_pc_metrics(pca, metric_names=None, max_spikes_for_nn=10000, n_neighbors=4, ):
    if metric_names is None:
        metric_names = _possible_pc_metric_names
    # print('metric_names', metric_names)

    assert isinstance(pca, WaveformPrincipalComponent)
    we = pca.waveform_extractor

    unit_ids = we.sorting.unit_ids
    channel_ids = we.recording.channel_ids

    # create output dict of dict  pc_metrics['metric_name'][unit_id]
    pc_metrics = {k: {} for k in metric_names}
    if 'nearest_neighbor' in metric_names:
        pc_metrics.pop('nearest_neighbor')
        pc_metrics['nn_hit_rate'] = {}
        pc_metrics['nn_miss_rate'] = {}

    for unit_id in unit_ids:
        # @alessio @ cole: please propose something here
        # TODO : make clear neiborhood for channel_ids and unit_ids
        # DEBUG : for debug we take all other channels and units
        neighbor_unit_ids = unit_ids
        neighbor_channel_ids = channel_ids
        # END DEBUG

        labels, pcs = pca.get_all_components(
            channel_ids=neighbor_channel_ids, unit_ids=neighbor_unit_ids)

        pcs_flat = pcs.reshape(pcs.shape[0], -1)

        # metrics
        if 'isolation_distance' in metric_names or 'l_ratio' in metric_names:
            isolation_distance, l_ratio = mahalanobis_metrics(pcs_flat, labels, unit_id)
            if 'isolation_distance' in metric_names:
                pc_metrics['isolation_distance'][unit_id] = isolation_distance
            if 'l_ratio' in metric_names:
                pc_metrics['l_ratio'][unit_id] = l_ratio

        if 'd_prime' in metric_names:
            d_prime = lda_metrics(pcs_flat, labels, unit_id)
            pc_metrics['d_prime'][unit_id] = d_prime

        if 'nearest_neighbor' in metric_names:
            nn_hit_rate, nn_miss_rate = nearest_neighbors_metrics(pcs_flat, labels, unit_id,
                                                                  max_spikes_for_nn, n_neighbors)
            pc_metrics['nn_hit_rate'][unit_id] = nn_hit_rate
            pc_metrics['nn_miss_rate'][unit_id] = nn_miss_rate

    return pc_metrics


#################################################################
# Code from spikemetrics

def mahalanobis_metrics(all_pcs, all_labels, this_unit_id):
    """
    Calculates isolation distance and L-ratio (metrics computed from Mahalanobis distance)

    Based on metrics described in Schmitzer-Torbert et al. (2005) Neurosci 131: 1-11

    Inputs:
    -------
    all_pcs : numpy.ndarray (num_spikes x PCs)
        2D array of PCs for all spikes
    all_labels : numpy.ndarray (num_spikes x 0)
        1D array of cluster labels for all spikes
    this_unit_id : Int
        number corresponding to unit for which these metrics will be calculated

    Outputs:
    --------
    isolation_distance : float
        Isolation distance of this unit
    l_ratio : float
        L-ratio for this unit

    """

    pcs_for_this_unit = all_pcs[all_labels == this_unit_id, :]
    pcs_for_other_units = all_pcs[all_labels != this_unit_id, :]

    mean_value = np.expand_dims(np.mean(pcs_for_this_unit, 0), 0)

    try:
        VI = np.linalg.inv(np.cov(pcs_for_this_unit.T))
    except np.linalg.linalg.LinAlgError:
        # case of singular matrix
        return np.nan, np.nan

    mahalanobis_other = np.sort(scipy.spatial.distance.cdist(mean_value,
                                                             pcs_for_other_units,
                                                             'mahalanobis', VI=VI)[0])

    mahalanobis_self = np.sort(scipy.spatial.distance.cdist(mean_value,
                                                            pcs_for_this_unit,
                                                            'mahalanobis', VI=VI)[0])

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
    """ Calculates d-prime based on Linear Discriminant Analysis

    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    Inputs:
    -------
    all_pcs : numpy.ndarray (num_spikes x PCs)
        2D array of PCs for all spikes
    all_labels : numpy.ndarray (num_spikes x 0)
        1D array of cluster labels for all spikes
    this_unit_id : Int
        number corresponding to unit for which these metrics will be calculated

    Outputs:
    --------
    d_prime : float
        Isolation distance of this unit
    l_ratio : float
        L-ratio for this unit

    """

    X = all_pcs

    y = np.zeros((X.shape[0],), dtype='bool')
    y[all_labels == this_unit_id] = True

    lda = LinearDiscriminantAnalysis(n_components=1)

    X_flda = lda.fit_transform(X, y)

    flda_this_cluster = X_flda[np.where(y)[0]]
    flda_other_cluster = X_flda[np.where(np.invert(y))[0]]

    d_prime = (np.mean(flda_this_cluster) - np.mean(flda_other_cluster)) / np.sqrt(
        0.5 * (np.std(flda_this_cluster) ** 2 + np.std(flda_other_cluster) ** 2))

    return d_prime


def nearest_neighbors_metrics(all_pcs, all_labels, this_unit_id, max_spikes_for_nn, n_neighbors):
    """ Calculates unit contamination based on NearestNeighbors search in PCA space

    Based on metrics described in Chung, Magland et al. (2017) Neuron 95: 1381-1394

    A is a (hopefully) representative subset of cluster X
    NN_hit(X) = 1/k \sum_i=1^k |{x in A such that ith closest neighbor is in X}| / |A|

    Inputs:
    -------
    all_pcs : numpy.ndarray (num_spikes x PCs)
        2D array of PCs for all spikes
    all_labels : numpy.ndarray (num_spikes x 0)
        1D array of cluster labels for all spikes
    this_unit_id : Int
        number corresponding to unit for which these metrics will be calculated
    max_spikes_for_nn : Int
        number of spikes to use (calculation can be very slow when this number is >20000)
    n_neighbors : Int
        number of neighbors to use

    Outputs:
    --------
    hit_rate : float
        Fraction of neighbors for target cluster that are also in target cluster
    miss_rate : float
        Fraction of neighbors outside target cluster that are in target cluster

    """

    total_spikes = all_pcs.shape[0]
    ratio = max_spikes_for_nn / total_spikes
    this_unit = all_labels == this_unit_id

    X = np.concatenate((all_pcs[this_unit, :], all_pcs[np.invert(this_unit), :]), 0)

    n = np.sum(this_unit)

    if ratio < 1:
        inds = np.arange(0, X.shape[0] - 1, 1 / ratio).astype('int')
        X = X[inds, :]
        n = int(n * ratio)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    this_cluster_inds = np.arange(n)

    this_cluster_nearest = indices[:n, 1:].flatten()
    other_cluster_nearest = indices[n:, 1:].flatten()

    hit_rate = np.mean(this_cluster_nearest < n)
    miss_rate = np.mean(other_cluster_nearest < n)

    return hit_rate, miss_rate
