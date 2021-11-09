from collections import namedtuple

import numpy as np
import scipy.stats
import scipy.spatial.distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import IncrementalPCA
import spikeinterface as si
from ..utils import get_random_data_chunks
from ..postprocessing import get_template_channel_sparsity

from ..postprocessing import WaveformPrincipalComponent

_possible_pc_metric_names = ['isolation_distance', 'l_ratio', 'd_prime',
                             'nearest_neighbor', 'nn_isolation', 'nn_noise_overlap']


def calculate_pc_metrics(pca, metric_names=None, max_spikes_for_nn=10000, 
                         n_neighbors=4, n_components=10, radius_um=100, seed=0):
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
            
        if 'nn_isolation' in metric_names:
            nn_isolation = nearest_neighbors_isolation(we, unit_id, max_spikes_for_nn,
                                                       n_neighbors, n_components,
                                                       radius_um, seed)
            pc_metrics['nn_isolation'][unit_id] = nn_isolation

        if 'nn_noise_overlap' in metric_names:
            nn_noise_overlap = nearest_neighbors_noise_overlap(we, unit_id, max_spikes_for_nn,
                                                               n_neighbors, n_components,
                                                               radius_um, seed)
            pc_metrics['nn_noise_overlap'][unit_id] = nn_noise_overlap
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

def nearest_neighbors_isolation(waveform_extractor: si.WaveformExtractor, this_unit_id: int,
                                max_spikes_for_nn: int=1000, n_neighbors: int=5, 
                                n_components: int=10, radius_um: float=100, seed: int=0):
    """Calculates unit isolation based on NearestNeighbors search in PCA space

    Based on isolation metric described in Chung et al. (2017) Neuron 95: 1381-1394.

    Rough logic:
    ------------
    1) Choose a cluster
    2) Compute the isolation score with every other cluster
    3) Isolation score is defined as the min of (2) (i.e. 'worst-case measure')
    
    Implementation details:
    -----------------------
    Let A and B be two clusters from sorting. 
    
    We set |A| = |B|:
        If max_spikes_for_nn < |A| and max_spikes_for_nn < |B|, then randomly subsample max_spikes_for_nn samples from A and B.
        If max_spikes_for_nn > min(|A|, |B|) (e.g. |A| > max_spikes_for_nn > |B|), then randomly subsample min(|A|, |B|) samples from A and B.
        This is because the metric is affected by the size of the clusters being compared independently of how well-isolated they are.
    
    We also restrict the waveforms to channels with significant signal
    
    See docstring for `_compute_isolation` for the definition of isolation score.

    Parameters:
    -----------
    all_pcs: array_like, (num_spikes, PCs)
        2D array of PCs for all spikes
    all_labels: array_like, (num_spikes, )
        1D array of cluster labels for all spikes
    this_unit_id: int
        ID of unit for which this metric will be calculated
    max_spikes_for_nn: int
        max number of spikes to use per cluster
    n_neighbors: int
        number of neighbors to check membership of
    seed: int
        seed for random subsampling of spikes

    Outputs:
    --------
    nn_isolation : float
    """
    
    rng = np.random.default_rng(seed=seed)
    
    all_units_ids = waveform_extractor.sorting.get_unit_ids()
    other_units_ids = np.setdiff1d(all_units_ids, this_unit_id)
    
    # get waveforms for target cluster
    waveforms_target_unit = waveform_extractor.get_waveforms(unit_id=this_unit_id)
    n_spikes_target_unit = waveforms_target_unit.shape[0]
    
    # find units whose closest channels overlap with closest channels of target cluster
    closest_chans_all = get_template_channel_sparsity(waveform_extractor, method='radius',
                                                      outputs='index', peak_sign='both', 
                                                      radius_um=radius_um)
    closest_chans_target_unit = closest_chans_all[this_unit_id]
    other_units_ids = [unit_id for unit_id in other_units_ids if np.any(np.in1d(closest_chans_all[unit_id],
                                                                                closest_chans_target_unit))]
    
    # if no unit is within neighborhood of target unit, then just say isolation is 1 (best possible)
    if not other_units_ids:
        nn_isolation = 1
    # if there are units to compare, then compute isolation with each 
    else: 
        isolation = np.zeros(len(other_units_ids),)
        for other_unit_id in other_units_ids:
            waveforms_other_unit = waveform_extractor.get_waveforms(unit_id=other_unit_id)
            n_spikes_other_unit = waveforms_other_unit.shape[0]

            n_snippets = np.min([n_spikes_target_unit, n_spikes_other_unit, max_spikes_for_nn])

            # make the two clusters equal in terms of:
            # - number of spikes
            # - channels with signal
            waveforms_target_unit_idx = rng.choice(n_spikes_target_unit, size=n_snippets, replace=False)
            waveforms_target_unit_sampled = waveforms_target_unit[waveforms_target_unit_idx]
            waveforms_target_unit_sampled = waveforms_target_unit_sampled[:, :, closest_chans_target_unit]

            waveforms_other_unit_idx = rng.choice(n_spikes_other_unit, size=n_snippets, replace=False)
            waveforms_other_unit_sampled = waveforms_other_unit[waveforms_other_unit_idx]
            waveforms_other_unit_sampled = waveforms_other_unit_sampled[:, :, closest_chans_target_unit]

            # compute principal components after concatenation
            all_snippets = np.concatenate([waveforms_target_unit_sampled.reshape((n_snippets, -1)),
                                           waveforms_other_unit_sampled.reshape((n_snippets, -1))], axis=0)
            pca = IncrementalPCA(n_components=n_components)
            pca.partial_fit(all_snippets)
            projected_snippets = pca.transform(all_snippets)
            # compute isolation
            isolation[other_unit_id==other_units_ids] = _compute_isolation(projected_snippets[:n_snippets,:],
                                                                           projected_snippets[n_snippets:,:],
                                                                           n_neighbors)
        nn_isolation = np.min(isolation)
    return nn_isolation

def nearest_neighbors_noise_overlap(waveform_extractor: si.WaveformExtractor, 
                                    this_unit_id: int, max_spikes_for_nn: int=1000,
                                    n_neighbors: int=5, n_components: int=10,
                                    radius_um: float=100, seed: int=0):
    """Calculates unit noise overlap based on NearestNeighbors search in PCA space.

    Based on noise overlap metric described in Chung et al. (2017) Neuron 95: 1381-1394.

    Rough logic:
    ------------
    1) Generate a noise cluster by randomly sampling voltage snippets from recording.
    2) Subtract projection onto the weighted average of noise snippets
       of both the target and noise clusters to correct for bias in sampling.
    3) Compute the isolation score between the noise cluster and the target cluster.
    
    Implementation details:
    -----------------------
    As with nn_isolation, the clusters that are compared (target and noise clusters)
    have the same number of spikes.
    
    See docstring for `_compute_isolation` for the definition of isolation score.
    
    Parameters:
    -----------
    we: si.WaveformExtractor
    this_unit_id: int
        ID of unit for which this metric will be calculated
    max_spikes_for_nn: int
        max number of spikes to use per cluster
    n_neighbors: int
        number of neighbors to check membership of
    n_components: int
        number of PC components to project the snippets
    radius_um: float
        only the channels within this radius of the peak channel
        are used to compute the metric
    seed: int
        seed for random subsampling of spikes

    Outputs:
    --------
    nn_noise_overlap : float
    """

    # set random seed
    rng = np.random.default_rng(seed=seed)
    
    # get random snippets from the recording to create a noise cluster
    recording = waveform_extractor.recording
    noise_cluster = get_random_data_chunks(recording, return_scaled=waveform_extractor.return_scaled,
                                           num_chunks_per_segment=max_spikes_for_nn,
                                           chunk_size=waveform_extractor.nsamples, seed=seed)
    
    noise_cluster = np.reshape(noise_cluster, (max_spikes_for_nn, waveform_extractor.nsamples, -1))
    
    # get waveforms for target cluster
    waveforms = waveform_extractor.get_waveforms(unit_id=this_unit_id)
    
    # adjust the size of the target and noise clusters to be equal
    if waveforms.shape[0] > max_spikes_for_nn:
        wf_ind = rng.choice(waveforms.shape[0], max_spikes_for_nn, replace=False)
        waveforms = waveforms[wf_ind]
        n_snippets = max_spikes_for_nn
    elif waveforms.shape[0] < max_spikes_for_nn:
        noise_ind = rng.choice(noise_cluster.shape[0], waveforms.shape[0], replace=False)
        noise_cluster = noise_cluster[noise_ind]
        n_snippets = waveforms.shape[0]
    else:
        n_snippets = max_spikes_for_nn

    # restrict to channels with significant signal
    closest_chans_idx = get_template_channel_sparsity(waveform_extractor, method='radius',
                                                      outputs='index', peak_sign='both', radius_um=radius_um)
    waveforms = waveforms[:,:,closest_chans_idx[this_unit_id]]
    noise_cluster = noise_cluster[:,:,closest_chans_idx[this_unit_id]]
    
    # compute weighted noise snippet (Z)
    median_waveform = waveform_extractor.get_template(unit_id=this_unit_id, mode='median')
    median_waveform = median_waveform[:, closest_chans_idx[this_unit_id]]
    tmax, chmax = np.unravel_index(np.argmax(np.abs(median_waveform)), median_waveform.shape)   
    weights = [noise_clip[tmax, chmax] for noise_clip in noise_cluster]
    weights = np.asarray(weights)
    weights = weights / np.sum(weights)
    weighted_noise_snippet = np.sum(weights * noise_cluster.swapaxes(0,2), axis=2).swapaxes(0,1)
    
    # subtract projection onto weighted noise snippet
    for snippet in range(n_snippets):
        waveforms[snippet, :, :] = _subtract_clip_component(waveforms[snippet, :, :], weighted_noise_snippet)
        noise_cluster[snippet, :, :] = _subtract_clip_component(noise_cluster[snippet, :, :], weighted_noise_snippet)
        
    # compute principal components after concatenation
    all_snippets = np.concatenate([waveforms.reshape((n_snippets, -1)), noise_cluster.reshape((n_snippets, -1))], axis=0)
    pca = IncrementalPCA(n_components=n_components)
    pca.partial_fit(all_snippets)
    projected_snippets = pca.transform(all_snippets)
    
    # compute overlap
    nn_noise_overlap = 1 - _compute_isolation(projected_snippets[:n_snippets,:],
                                              projected_snippets[n_snippets:,:],
                                              n_neighbors)
    return nn_noise_overlap

def _subtract_clip_component(clip1, component):
    V1 = clip1.flatten()
    V2 = component.flatten()
    V1 = V1 - V2 * np.dot(V1, V2) / np.dot(V2, V2)
    return V1.reshape(clip1.shape)

def _compute_isolation(pcs_target_unit, pcs_other_unit, n_neighbors:int):
    """Computes the isolation score used for nn_isolation and nn_noise_overlap

    Definition of isolation score:
        Isolation(A, B) = 1/k \sum_{j=1}^k |{x \in A U B: \rho(x)=\rho(jth nearest neighbor of x)}| / |A U B|
            where \rho(x) is the cluster x belongs to (in this case, either A or B)
        Note that this definition implies that the isolation score (1) ranges from 0 to 1; and 
                                                                   (2) is symmetric, i.e. Isolation(A, B) = Isolation(B, A)

    Parameters
    ----------
    pcs_target_unit: np.array, (n_spikes, n_components)
        PCA projection of the spikes in the target cluster
    pcs_other_unit: np.array, (n_spikes, n_components)
        PCA projection of the spikes in the other cluster
    n_neighbors: int
        number of nearest neighbors to check membership of
    """

    # get lengths
    n_spikes_target = pcs_target_unit.shape[0]
    n_spikes_other = pcs_other_unit.shape[0]
    
    # concatenate
    pcs_concat = np.concatenate((pcs_target_unit, pcs_other_unit), axis=0)
    label_concat = np.concatenate((np.zeros(n_spikes_target),np.ones(n_spikes_other)))
        
    # if n_neighbors is greater than the number of spikes in both clusters, then set it to max possible
    if n_neighbors > len(label_concat):
        n_neighbors_adjusted = len(label_concat)-1
    else:
        n_neighbors_adjusted = n_neighbors
    
    _, membership_ind = NearestNeighbors(n_neighbors=n_neighbors_adjusted, algorithm='auto').fit(pcs_concat).kneighbors()
    
    target_nn_in_target = np.sum(label_concat[membership_ind[:n_spikes_target]]==0)
    other_nn_in_other = np.sum(label_concat[membership_ind[n_spikes_target:]]==1) 

    isolation = (target_nn_in_target + other_nn_in_other) / (n_spikes_target+n_spikes_other) / n_neighbors_adjusted
    
    return isolation