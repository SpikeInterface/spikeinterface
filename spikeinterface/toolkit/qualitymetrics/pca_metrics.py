from collections import namedtuple

import numpy as np
import scipy.stats
import scipy.spatial.distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors

from ..postprocessing import WaveformPrincipalComponent

_possible_pc_metric_names = ['isolation_distance', 'l_ratio', 'd_prime', 'nearest_neighbor', 'nn_isolation']


def calculate_pc_metrics(pca, metric_names=None, max_spikes_for_nn=10000, n_neighbors=4, seed=0):
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
            
        if 'nearest_neighbor_isolation' in metric_names:
            nn_isolation = nearest_neighbors_isolation(pcs_flat, labels, unit_id, 
                                                       max_spikes_for_nn, n_neighbors, seed)
            pc_metrics['nn_isolation'][unit_id] = nn_isolation
    
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

def nearest_neighbors_isolation(all_pcs, all_labels, this_unit_id, max_spikes_for_nn, n_neighbors, seed):
    """ Calculates unit isolation based on NearestNeighbors search in PCA space

    Based on isolation metric described in Chung et al. (2017) Neuron 95: 1381-1394.

    Rough logic
    -----------
    1) Choose a cluster
    2) Compute the isolation function with every other cluster
    3) Isolation score is defined as the min of (2)
    
    Implementation
    --------------
    Let A and B be clusters from sorting. 
    
    We set |A| = |B|:
        If max_spikes_for_nn < |A| and max_spikes_for_nn < |B|, then randomly subsample max_spikes_for_nn samples from A and B.
        If max_spikes_for_nn > min(|A|, |B|) (e.g. |A| > max_spikes_for_nn > |B|), then randomly subsample min(|A|, |B|) samples from A and B.
        This is because the metric is affected by the size of the clusters being compared independently of how well-isolated they are.
        
    Isolation function:
        Isolation(A, B) = 1/k \sum_{j=1}^k |{x \in A U B: \rho(x)=\rho(jth nearest neighbor of x)}| / |A U B|
            where \rho(x) is the cluster x belongs to (in this case, either A or B)
        Note that this definition implies that the isolation funciton  (1) ranges from 0 to 1; and 
                                                                       (2) is symmetric, i.e. Isolation(A, B) = Isolation(B, A)

    Parameters:
    -----------
    all_pcs: array_like, (num_spikes, PCs)
        2D array of PCs for all spikes
    all_labels: array_like, (num_spikes, )
        1D array of cluster labels for all spikes
    this_unit_id: int
        ID of unit for which thiss metric will be calculated
    max_spikes_for_nn: int
        max number of spikes to use per cluster
    n_neighbors: int
        number of neighbors to check membership of
    seed: int
        seed for random subsampling of spikes

    Outputs:
    --------
    nearest_neighbor_isolation : float
    
    """
    
    rng = np.random.default_rng(seed=seed)
    
    all_units_ids = np.unique(all_labels)
    other_units_ids = np.setdiff1d(all_units_ids, this_unit_id)

    isolation = np.zeros(len(other_units_ids),)
    # compute isolation with each cluster
    for other_unit_id in other_units_ids:
        n_spikes_target_unit = np.sum(all_labels==this_unit_id)
        pcs_target_unit = all_pcs[all_labels==this_unit_id, :]

        n_spikes_other_unit = np.sum(all_labels==other_unit_id)
        pcs_other_unit = all_pcs[all_labels==other_unit_id]

        spikes_for_nn_actual = np.min([n_spikes_target_unit, n_spikes_other_unit, max_spikes_for_nn])

        if spikes_for_nn_actual < n_spikes_target_unit:
            pcs_target_unit_idx = rng.choice(np.arange(n_spikes_target_unit), size=spikes_for_nn_actual)
            pcs_target_unit = pcs_target_unit[pcs_target_unit_idx]

        if spikes_for_nn_actual < n_spikes_other_unit:
            pcs_other_unit_idx = rng.choice(np.arange(n_spikes_other_unit), size=spikes_for_nn_actual)
            pcs_other_unit = pcs_other_unit[pcs_other_unit_idx]

        pcs_concat = np.concatenate((pcs_target_unit, pcs_other_unit), axis=0)
        label_concat = np.concatenate((np.zeros(spikes_for_nn_actual),np.ones(spikes_for_nn_actual)))
        
        # if n_neighbors is greater than the number of spikes in both clusters, then set it to max possible
        if n_neighbors > len(label_concat):
            n_neighbors_adjusted = len(label_concat)-1
        else:
            n_neighbors_adjusted = n_neighbors
        
        _, membership_ind = NearestNeighbors(n_neighbors=n_neighbors_adjusted, algorithm='auto').fit(pcs_concat).kneighbors()
        
        target_nn_in_target = np.sum(label_concat[membership_ind[:spikes_for_nn_actual]]==0)
        other_nn_in_other = np.sum(label_concat[membership_ind[spikes_for_nn_actual:]]==1) 

        isolation[other_unit_id==other_units_ids] = (target_nn_in_target + other_nn_in_other) / (2*spikes_for_nn_actual) / n_neighbors_adjusted
    
    nearest_neighbor_isolation = np.min(isolation)
    return nearest_neighbor_isolation

def nearest_neighbors_noise_overlap(all_pcs, all_labels, this_unit_id, max_spikes_for_nn, n_neighbors, seed):
    """ Calculates unit noise overlap based on NearestNeighbors search in PCA space

    Based on noise overlap metric described in Chung et al. (2017) Neuron 95: 1381-1394.

    Rough logic
    -----------
    1) Generate a noise cluster by randomly sampling from recording
    2) Compute the isolation function between noise cluster and target cluster
    
    Implementation
    --------------
    Note that the noise cluster must have the same number of spikes as the target cluster
    Let A and B be clusters from sorting. 
    
    See docstring for nearest_neighbors_isolation for definition of isolation function.
    
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
    nearest_neighbor_isolation : float
    
    """

    # set random seed
    rng = np.random.default_rng(seed=seed)
    
    

    n_waveforms_per_unit = np.array([len(wf) for wf in waveforms])
    n_spikes_per_unit = np.array([len(self._metric_data._sorting.get_unit_spike_train(u)) for u in self._metric_data._unit_ids])

    if np.all(n_waveforms_per_unit < max_spikes_per_unit_for_noise_overlap):
        # in this case it means that waveforms have been computed on
        # less spikes than max_spikes_per_unit_for_noise_overlap --> recompute
        kwargs['recompute_info'] = True
        waveforms = st.postprocessing.get_unit_waveforms(
                self._metric_data._recording,
                self._metric_data._sorting,
                unit_ids = self._metric_data._unit_ids,
                # max_spikes_per_unit = max_spikes_per_unit_for_noise_overlap,
                **kwargs)
    elif np.all(n_waveforms_per_unit >= max_spikes_per_unit_for_noise_overlap):
        # waveforms computed on more spikes than needed --> sample
        for i_w, wfs in enumerate(waveforms):
            if len(wfs) > max_spikes_per_unit_for_noise_overlap:
                selecte_idxs = np.random.permutation(len(wfs))[:max_spikes_per_unit_for_noise_overlap]
                waveforms[i_w] = wfs[selecte_idxs]

    # get channel idx and locations
    channel_idx = np.arange(self._metric_data._recording.get_num_channels())
    channel_locations = self._metric_data._channel_locations

    if num_channels_to_compare > len(channel_idx):
        num_channels_to_compare = len(channel_idx)

    # get noise snippets
    min_time = min([self._metric_data._sorting.get_unit_spike_train(unit_id=unit)[0]
                for unit in self._metric_data._sorting.get_unit_ids()])
    max_time = max([self._metric_data._sorting.get_unit_spike_train(unit_id=unit)[-1]
                for unit in self._metric_data._sorting.get_unit_ids()])
    max_spikes = np.max([len(self._metric_data._sorting.get_unit_spike_train(u)) for u in self._metric_data._unit_ids])
    if max_spikes < max_spikes_per_unit_for_noise_overlap:
        max_spikes_per_unit_for_noise_overlap = max_spikes
    times_control = np.random.choice(np.arange(min_time, max_time),
                size=max_spikes_per_unit_for_noise_overlap, replace=False)
    clip_size = waveforms[0].shape[-1]
    # np.array, (n_spikes, n_channels, n_timepoints)
    clips_control_max = np.stack(self._metric_data._recording.get_snippets(snippet_len=clip_size,
                                                                            reference_frames=times_control))

    noise_overlaps = []
    for i_u, unit in enumerate(self._metric_data._unit_ids):
        # show progress bar
        if self._metric_data.verbose:
            printProgressBar(i_u + 1, len(self._metric_data._unit_ids))

        # get spike and noise snippets
        # np.array, (n_spikes, n_channels, n_timepoints)
        clips = waveforms[i_u]
        clips_control = clips_control_max

        # make noise snippets size equal to number of spikes
        if len(clips) < max_spikes_per_unit_for_noise_overlap:
            selected_idxs = np.random.choice(np.arange(max_spikes_per_unit_for_noise_overlap),
                                            size=len(clips), replace=False)
            clips_control = clips_control[selected_idxs]
        else:
            selected_idxs = np.random.choice(np.arange(len(clips)),
                                            size=max_spikes_per_unit_for_noise_overlap,
                                            replace=False)
            clips = clips[selected_idxs]

        num_clips = len(clips)

        # compute weight for correcting noise snippets
        template = np.median(clips, axis=0)
        chmax, tmax = np.unravel_index(np.argmax(np.abs(template)), template.shape)
        max_val = template[chmax, tmax]
        weighted_clips_control = np.zeros(clips_control.shape)
        weights = np.zeros(num_clips)
        for j in range(num_clips):
            clip0 = clips_control[j, :, :]
            val0 = clip0[chmax, tmax]
            weight0 = val0 * max_val
            weights[j] = weight0
            weighted_clips_control[j, :, :] = clip0 * weight0

        noise_template = np.sum(weighted_clips_control, axis=0)
        noise_template = noise_template / np.sum(np.abs(noise_template)) * np.sum(np.abs(template))

        # subtract it out
        for j in range(num_clips):
            clips[j, :, :] = _subtract_clip_component(clips[j, :, :], noise_template)
            clips_control[j, :, :] = _subtract_clip_component(clips_control[j, :, :], noise_template)

        # use only subsets of channels that are closest to peak channel
        channels_to_use = find_neighboring_channels(chmax, channel_idx,
                                num_channels_to_compare, channel_locations)
        channels_to_use = np.sort(channels_to_use)
        clips = clips[:,channels_to_use,:]
        clips_control = clips_control[:,channels_to_use,:]

        all_clips = np.concatenate([clips, clips_control], axis=0)
        num_channels_wfs = all_clips.shape[1]
        num_samples_wfs = all_clips.shape[2]
        all_features = _compute_pca_features(all_clips.reshape((num_clips * 2,
                                                                num_channels_wfs * num_samples_wfs)), num_features)
        num_all_clips=len(all_clips)
        distances, indices = NearestNeighbors(n_neighbors=min(num_knn + 1, num_all_clips - 1), algorithm='auto').fit(
            all_features.T).kneighbors()

        group_id = np.zeros((num_clips * 2))
        group_id[0:num_clips] = 1
        group_id[num_clips:] = 2
        num_match = 0
        total = 0
        for j in range(num_clips * 2):
            for k in range(1, min(num_knn + 1, num_all_clips - 1)):
                ind = indices[j][k]
                if group_id[j] == group_id[ind]:
                    num_match = num_match + 1
                total = total + 1
        pct_match = num_match / total
        noise_overlap = 1 - pct_match
        noise_overlaps.append(noise_overlap)
    noise_overlaps = np.asarray(noise_overlaps)
    if save_property_or_features:
        self.save_property_or_features(self._metric_data._sorting, noise_overlaps, self._metric_name)
    return noise_overlaps

def compute_noise_overlaps(
        sorting,
        recording,
        num_channels_to_compare=NoiseOverlap.params['num_channels_to_compare'],
        num_features=NoiseOverlap.params['num_features'],
        num_knn=NoiseOverlap.params['num_knn'],
        max_spikes_per_unit_for_noise_overlap=NoiseOverlap.params['max_spikes_per_unit_for_noise_overlap'],
        unit_ids=None,
        **kwargs
):
    """
    Computes and returns the noise overlaps in the sorted dataset.
    Noise overlap estimates the fraction of ‘‘noise events’’ in a cluster, i.e., above-threshold events not associated
    with true firings of this or any of the other clustered units. A large noise overlap implies a high false-positive
    rate.

    Implementation from ml_ms4alg. For more information see https://doi.org/10.1016/j.neuron.2017.08.030

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes
    num_features: int
        Number of features to use for PCA
    num_knn: int
        Number of nearest neighbors
    max_spikes_per_unit_for_noise_overlap: int
        Number of waveforms to use for noise overlaps estimation
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    **kwargs: keyword arguments
        Keyword arguments among the following:
            method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes
            peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or
                both ('both' - default)
            frames_before: int
                Frames before peak to compute amplitude
            frames_after: int
                Frames after peak to compute amplitude
            apply_filter: bool
                If True, recording is bandpass-filtered
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz)
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz)
            grouping_property: str
                Property to group channels. E.g. if the recording extractor has the 'group' property and
                'grouping_property' is 'group', then waveforms are computed group-wise.
            ms_before: float
                Time period in ms to cut waveforms before the spike events
            ms_after: float
                Time period in ms to cut waveforms after the spike events
            dtype: dtype
                The numpy dtype of the waveforms
            compute_property_from_recording: bool
                If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding
                property of the recording extractor channel on which the average waveform is the largest
            max_channels_per_waveforms: int or None
                Maximum channels per waveforms to return. If None, all channels are returned
            n_jobs: int
                Number of parallel jobs (default 1)
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            save_property_or_features: bool
                If true, it will save features in the sorting extractor
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit
            seed: int
                Random seed for reproducibility
            verbose: bool
                If True, will be verbose in metric computation

    Returns
    ----------
    noise_overlaps: np.ndarray
        The noise_overlaps of the sorted units.
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    duration_in_frames=None, freq_max=params_dict["freq_max"], unit_ids=unit_ids,
                    verbose=params_dict['verbose'])

    noise_overlap = NoiseOverlap(metric_data=md)
    noise_overlaps = noise_overlap.compute_metric(num_channels_to_compare,
                                                  max_spikes_per_unit_for_noise_overlap,
                                                  num_features, num_knn, **kwargs)
    return noise_overlaps

class NoiseOverlap(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('num_channels_to_compare', 13),
                          ('max_spikes_per_unit_for_noise_overlap', 1000),
                          ('num_features', 10),
                          ('num_knn', 6)])
    curator_name = "ThresholdNoiseOverlaps"

    def __init__(self, metric_data):
        QualityMetric.__init__(self, metric_data, metric_name="noise_overlap")

        if not metric_data.has_recording():
            raise ValueError("MetricData object must have a recording")

    def compute_metric(self, num_channels_to_compare, max_spikes_per_unit_for_noise_overlap,
                        num_features, num_knn, **kwargs):

        # Make sure max_spikes_per_unit_for_noise_overlap is not None
        assert max_spikes_per_unit_for_noise_overlap is not None, "'max_spikes_per_unit_for_noise_overlap' must be an integer."

        # update keyword arg in case it's already specified to something
        kwargs['max_spikes_per_unit'] = max_spikes_per_unit_for_noise_overlap
        params_dict = update_all_param_dicts_with_kwargs(kwargs)
        save_property_or_features = params_dict['save_property_or_features']
        seed = params_dict['seed']

        # set random seed
        if seed is not None:
            np.random.seed(seed)

        # first, get waveform snippets of every unit (at most n spikes)
        # waveforms = List (units,) of np.array (n_spikes, n_channels, n_timepoints)
        waveforms = st.postprocessing.get_unit_waveforms(
            self._metric_data._recording,
            self._metric_data._sorting,
            unit_ids=self._metric_data._unit_ids,
            **kwargs)

        n_waveforms_per_unit = np.array([len(wf) for wf in waveforms])
        n_spikes_per_unit = np.array([len(self._metric_data._sorting.get_unit_spike_train(u)) for u in self._metric_data._unit_ids])

        if np.all(n_waveforms_per_unit < max_spikes_per_unit_for_noise_overlap):
            # in this case it means that waveforms have been computed on
            # less spikes than max_spikes_per_unit_for_noise_overlap --> recompute
            kwargs['recompute_info'] = True
            waveforms = st.postprocessing.get_unit_waveforms(
                    self._metric_data._recording,
                    self._metric_data._sorting,
                    unit_ids = self._metric_data._unit_ids,
                    # max_spikes_per_unit = max_spikes_per_unit_for_noise_overlap,
                    **kwargs)
        elif np.all(n_waveforms_per_unit >= max_spikes_per_unit_for_noise_overlap):
            # waveforms computed on more spikes than needed --> sample
            for i_w, wfs in enumerate(waveforms):
                if len(wfs) > max_spikes_per_unit_for_noise_overlap:
                    selecte_idxs = np.random.permutation(len(wfs))[:max_spikes_per_unit_for_noise_overlap]
                    waveforms[i_w] = wfs[selecte_idxs]

        # get channel idx and locations
        channel_idx = np.arange(self._metric_data._recording.get_num_channels())
        channel_locations = self._metric_data._channel_locations

        if num_channels_to_compare > len(channel_idx):
            num_channels_to_compare = len(channel_idx)

        # get noise snippets
        min_time = min([self._metric_data._sorting.get_unit_spike_train(unit_id=unit)[0]
                    for unit in self._metric_data._sorting.get_unit_ids()])
        max_time = max([self._metric_data._sorting.get_unit_spike_train(unit_id=unit)[-1]
                    for unit in self._metric_data._sorting.get_unit_ids()])
        max_spikes = np.max([len(self._metric_data._sorting.get_unit_spike_train(u)) for u in self._metric_data._unit_ids])
        if max_spikes < max_spikes_per_unit_for_noise_overlap:
            max_spikes_per_unit_for_noise_overlap = max_spikes
        times_control = np.random.choice(np.arange(min_time, max_time),
                    size=max_spikes_per_unit_for_noise_overlap, replace=False)
        clip_size = waveforms[0].shape[-1]
        # np.array, (n_spikes, n_channels, n_timepoints)
        clips_control_max = np.stack(self._metric_data._recording.get_snippets(snippet_len=clip_size,
                                                                               reference_frames=times_control))

        noise_overlaps = []
        for i_u, unit in enumerate(self._metric_data._unit_ids):
            # show progress bar
            if self._metric_data.verbose:
                printProgressBar(i_u + 1, len(self._metric_data._unit_ids))

            # get spike and noise snippets
            # np.array, (n_spikes, n_channels, n_timepoints)
            clips = waveforms[i_u]
            clips_control = clips_control_max

            # make noise snippets size equal to number of spikes
            if len(clips) < max_spikes_per_unit_for_noise_overlap:
                selected_idxs = np.random.choice(np.arange(max_spikes_per_unit_for_noise_overlap),
                                                size=len(clips), replace=False)
                clips_control = clips_control[selected_idxs]
            else:
                selected_idxs = np.random.choice(np.arange(len(clips)),
                                                size=max_spikes_per_unit_for_noise_overlap,
                                                replace=False)
                clips = clips[selected_idxs]

            num_clips = len(clips)

            # compute weight for correcting noise snippets
            template = np.median(clips, axis=0)
            chmax, tmax = np.unravel_index(np.argmax(np.abs(template)), template.shape)
            max_val = template[chmax, tmax]
            weighted_clips_control = np.zeros(clips_control.shape)
            weights = np.zeros(num_clips)
            for j in range(num_clips):
                clip0 = clips_control[j, :, :]
                val0 = clip0[chmax, tmax]
                weight0 = val0 * max_val
                weights[j] = weight0
                weighted_clips_control[j, :, :] = clip0 * weight0

            noise_template = np.sum(weighted_clips_control, axis=0)
            noise_template = noise_template / np.sum(np.abs(noise_template)) * np.sum(np.abs(template))

            # subtract it out
            for j in range(num_clips):
                clips[j, :, :] = _subtract_clip_component(clips[j, :, :], noise_template)
                clips_control[j, :, :] = _subtract_clip_component(clips_control[j, :, :], noise_template)

            # use only subsets of channels that are closest to peak channel
            channels_to_use = find_neighboring_channels(chmax, channel_idx,
                                    num_channels_to_compare, channel_locations)
            channels_to_use = np.sort(channels_to_use)
            clips = clips[:,channels_to_use,:]
            clips_control = clips_control[:,channels_to_use,:]

            all_clips = np.concatenate([clips, clips_control], axis=0)
            num_channels_wfs = all_clips.shape[1]
            num_samples_wfs = all_clips.shape[2]
            all_features = _compute_pca_features(all_clips.reshape((num_clips * 2,
                                                                    num_channels_wfs * num_samples_wfs)), num_features)
            num_all_clips=len(all_clips)
            distances, indices = NearestNeighbors(n_neighbors=min(num_knn + 1, num_all_clips - 1), algorithm='auto').fit(
                all_features.T).kneighbors()

            group_id = np.zeros((num_clips * 2))
            group_id[0:num_clips] = 1
            group_id[num_clips:] = 2
            num_match = 0
            total = 0
            for j in range(num_clips * 2):
                for k in range(1, min(num_knn + 1, num_all_clips - 1)):
                    ind = indices[j][k]
                    if group_id[j] == group_id[ind]:
                        num_match = num_match + 1
                    total = total + 1
            pct_match = num_match / total
            noise_overlap = 1 - pct_match
            noise_overlaps.append(noise_overlap)
        noise_overlaps = np.asarray(noise_overlaps)
        if save_property_or_features:
            self.save_property_or_features(self._metric_data._sorting, noise_overlaps, self._metric_name)
        return noise_overlaps