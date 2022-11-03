from re import template
import numpy as np
import scipy.signal
import scipy.spatial

from ..postprocessing import compute_correlograms, get_template_extremum_channel
from ..qualitymetrics import compute_refrac_period_violations, compute_firing_rate

from .mergeunitssorting import MergeUnitsSorting


def get_potential_auto_merge(waveform_extractor,
                minimum_spikes=1000, maximum_distance_um=150.,
                peak_sign="neg",
                bin_ms=0.25, window_ms=100.,
                corr_diff_thresh=0.16,
                template_diff_thresh=0.25,
                censored_period_ms=0., refractory_period_ms=1.0,
                sigma_smooth_ms = 0.6,
                contamination_threshold=0.2,
                adaptative_window_threshold=0.5,
                num_channels=5,
                num_shift=5,
                firing_contamination_balance=1.5,
                extra_outputs=False,
                ):
    """
    Algorithm to find and check potential merges between units.

    This is taken from lussac version 1 done by Aurelien Wyngaard.
    https://github.com/BarbourLab/lussac/blob/v1.0.0/postprocessing/merge_units.py


    The merge are proposed when:
      * STEP 1:  enougth spikes in each units for coimputing the correlogram
      * STEP 2: each units is not contaminated by checking auto corrlogram
      * STEP 3 : unit location are close enough
      * STEP 4 : the cross correlogram of 2 units is similar to each auto corrleogram
      * STEP 5 the waveform of the 2 units are similar
      * STEP 6 : the "score" is increase.
        The score is a balance between having more spike (increasing the firing) and 
        getting more contamination in the auto correlogram.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor
    minimum_spikes: int 
        Minimum number of spike to be a potential merge.
        Enought to estimate the correlogram
        STEP 1
    maximum_distance_um: float
        Minimum distance between units for considering a merge.
        STEP 3
    peak_sign: "neg"/"pos"/"both"
        Used to estimate the maximum channel of a template
        STEP 3
    bin_ms: float
        Used for computing the correlogram
        STEP 4
    window_ms: float
        Used for computing the correlogram
        STEP 4
    corr_diff_thresh: float
        between 0 and 1
        The threshold on the "correlogram distance metric" for considering a merge.
        STEP 4
    template_diff_thresh: float
        between 0 and 1
        The threshold on the "template distance metric" for considering a merge.
        STEP 5
    censored_period_ms: float
        Used to compute the refractory period violations aka "contaminagtion"
        STEP 2.
    refractory_period_ms: float
        Used to compute the refractory period violations aka "contaminagtion"
        STEP 2.
    sigma_smooth_ms: float
        Parameters to smooth the correlogram in STEP 4
    contamination_threshold: float
        Threshold for not taking in account a unit when too much contaminated (STEP 2)
    adaptative_window_threshold:: float
        Parameter to detect the window size in STEP 4
    num_channels: int
        Number of channel to use for STEP 5
    num_shift: int
        shift to be explored for STEP 5
    firing_contamination_balance: float
        Parameter to control the balance for STEP 6.
        
    Returns
    -------
    potential_merges
        A list of tuple of 2 elements.
        List of pair that could be merged
    outs: optional
        Return only when extra_outputs=True
        A dictionary that contain data for debugging and plotting.
    """
    
    we = waveform_extractor
    sorting = we.sorting
    unit_ids = sorting.unit_ids

    # to get fast computation we will not analyse pairs when:
    #    * not enough spikes for one of theses
    #    * auto correlogram is contaminated
    #    * to far away one from each other
    

    # STEP 1 : 
    n = unit_ids.size
    pair_mask = np.ones((n, n), dtype='bool')
    num_spikes = np.array(list(sorting.get_total_num_spikes().values()))
    to_remove = num_spikes < minimum_spikes
    pair_mask[to_remove, :] = False
    pair_mask[:, to_remove] = False

    # STEP 2 : remove contaminated auto corr
    nb_violations, contaminations = compute_refrac_period_violations(we, refractory_period_ms=refractory_period_ms,
                                    censored_period_ms=censored_period_ms)
    nb_violations = np.array(list(nb_violations.values()))
    contaminations = np.array(list(contaminations.values()))
    to_remove = contaminations > contamination_threshold
    pair_mask[to_remove, :] = False
    pair_mask[:, to_remove] = False

    # STEP 3 : unit positions are estimated roughly with channel
    chan_loc = we.recording.get_channel_locations()
    unit_max_chan = get_template_extremum_channel(we, peak_sign=peak_sign, mode="extremum", outputs="index")
    unit_max_chan = list(unit_max_chan.values())
    unit_locations = chan_loc[unit_max_chan, :]
    unit_distances = scipy.spatial.distance.cdist(unit_locations, unit_locations, metric='euclidean')
    pair_mask = pair_mask & (unit_distances <= maximum_distance_um)
    
    # print('Will check ', np.sum(pair_mask), 'pairs on ', pair_mask.size)
    
    # Step 4 : potential auto merge by correlogram
    correlograms, bins = compute_correlograms(sorting, window_ms=window_ms, bin_ms=bin_ms, method='numba')
    correlograms_smoothed = smooth_correlogram(correlograms, bins, sigma_smooth_ms=sigma_smooth_ms)
    # find correlogram window for each units
    win_sizes = np.zeros(n, dtype=int)
    for unit_ind in range(n):
        auto_corr = correlograms_smoothed[unit_ind, unit_ind, :]
        thresh = np.max(auto_corr) * adaptative_window_threshold
        win_size = get_unit_adaptive_window(auto_corr, thresh)
        win_sizes[unit_ind] = win_size
    correlogram_diff = compute_correlogram_diff(sorting, correlograms_smoothed, bins, win_sizes,
                                    adaptative_window_threshold=adaptative_window_threshold,
                                    pair_mask=pair_mask)
    pair_mask = pair_mask & (correlogram_diff  < corr_diff_thresh)



    # step 5 : check if potential merge with CC also have template similarity
    templates = we.get_all_templates(mode='average')
    templates_diff = compute_templates_diff(sorting, templates, num_channels=num_channels, num_shift=num_shift, pair_mask=pair_mask)
    pair_mask = pair_mask & (templates_diff  < template_diff_thresh)




    # step 6 : validate the potential merges with CC increase the contamination quality metrics
    pair_mask = check_improve_contaminations_score(we, pair_mask, contaminations, 
        firing_contamination_balance, refractory_period_ms, censored_period_ms)


    # create the final list from pair_mask boolean matrix
    ind1, ind2 = np.nonzero(pair_mask)
    potential_merges = list(zip(unit_ids[ind1], unit_ids[ind2]))


    if extra_outputs:
        outs = dict(
            correlograms=correlograms,
            bins=bins,
            correlograms_smoothed=correlograms_smoothed,
            correlogram_diff=correlogram_diff,
            win_sizes=win_sizes,
            templates_diff=templates_diff,
        )
        return potential_merges, outs
    else:
        return potential_merges


def compute_correlogram_diff(sorting, correlograms_smoothed, bins, win_sizes, adaptative_window_threshold=0.5,
            pair_mask=None):
    """
    Original author: Aurelien Wyngaard (lussac)
    
    Parameters
    ----------
    sorting: BaseSorting
        The sorting object
    correlograms_smoothed: array 3d
        The 3d array containing all cross and auto correlograms
        (smoothed by a convolution with a gaussian curve)
    bins: array
        Bins of the correlograms
    win_sized: 
        TODO
    adaptative_window_threshold: float
        TODO
    pair_mask: None or boolean array
        A bool matrix of size (num_units, num_units) to select
        which pair to compute.
    
    Returns
    -------
    corr_diff
    """
    # bin_ms = bins[1] - bins[0] 
    
    unit_ids = sorting.unit_ids
    n = len(unit_ids)
    
    if pair_mask is None:
        pair_mask = np.ones((n, n), dtype='bool')
    
    
    # Index of the middle of the correlograms.
    m = correlograms_smoothed.shape[2] // 2
    
    num_spikes = sorting.get_total_num_spikes()

    corr_diff = np.full((n, n), np.nan, dtype='float64')
    for unit_ind1 in range(n):
        for unit_ind2 in range(unit_ind1 + 1, n):
            if not pair_mask[unit_ind1, unit_ind2]:
                continue

            unit_id1, unit_id2 = unit_ids[unit_ind1], unit_ids[unit_ind2]

            
            num1, num2 = num_spikes[unit_id1], num_spikes[unit_id2]
            # Weighted window (larger unit imposes its window).
            win_size = int(round((num1 * win_sizes[unit_ind1] + num2 * win_sizes[unit_ind2]) / (num1 + num2)))    
            # Plage of indices where correlograms are inside the window.
            corr_inds = np.arange(m - win_size, m + win_size, dtype=int)
            
            # TODO : for Aurelien 
            shift = 0

            auto_corr1 = normalize_correlogram(correlograms_smoothed[unit_ind1, unit_ind1, :])
            auto_corr2 = normalize_correlogram(correlograms_smoothed[unit_ind2, unit_ind2, :])
            cross_corr = normalize_correlogram(correlograms_smoothed[unit_ind1, unit_ind2, :])
            diff1 = np.sum(np.abs(cross_corr[corr_inds - shift] - auto_corr1[corr_inds])) / len(corr_inds)
            diff2 = np.sum(np.abs(cross_corr[corr_inds - shift] - auto_corr2[corr_inds])) / len(corr_inds)
            # Weighted difference (larger unit imposes its difference).
            w_diff = (num1 * diff1 + num2 * diff2) / (num1 + num2)
            corr_diff[unit_ind1, unit_ind2] = w_diff
    
    return corr_diff


def normalize_correlogram(correlogram: np.ndarray):
    """
    Normalizes a correlogram so its mean in time is 1.
    If correlogram is 0 everywhere, stays 0 everywhere.

    Parameters
    ----------
    correlogram (np.ndarray):
        Correlogram to normalize.

    Returns
    -------
    normalized_correlogram (np.ndarray) [time]:
        Normalized correlogram to have a mean of 1.
    """
    mean = np.mean(correlogram)
    return correlogram if mean == 0 else correlogram / mean


def smooth_correlogram(correlograms, bins, sigma_smooth_ms=0.6):
    """
    
    """
    # OLD implementation : smooth correlogram by low pass filter
    # b, a = scipy.signal.butter(N=2, Wn = correlogram_low_pass / (1e3 / bin_ms /2), btype='low')
    # correlograms_smoothed = scipy.signal.filtfilt(b, a, correlograms, axis=2)

    # new implementation smooth by convolution with a Gaussian kernel
    smooth_kernel = np.exp( -bins**2 / ( 2 * sigma_smooth_ms **2))
    smooth_kernel /= np.sum(smooth_kernel)
    smooth_kernel = smooth_kernel[None, None, :]
    correlograms_smoothed = scipy.signal.fftconvolve(correlograms, smooth_kernel, mode='same', axes=2)

    return correlograms_smoothed

def get_unit_adaptive_window(auto_corr: np.ndarray, threshold: float):
    """
    Computes an adaptive window to correlogram (basically corresponds to the first peak).
    Based on a minimum threshold and minimum of second derivative.
    If no peak is found over threshold, recomputes with threshold/2.

    Parameters
    ----------
    auto_corr (np.ndarray) [time]:
        Correlogram used for adaptive window.
    threshold (float):
        Minimum threshold of correlogram (all peaks under this threshold are discarded).

    Returns
    -------
    unit_window (int):
        Index at which the adaptive window has been calculated.
    """
    if np.sum(np.abs(auto_corr)) == 0:
        return 20.0

    derivative_2 = -np.gradient(np.gradient(auto_corr))
    peaks = scipy.signal.find_peaks(derivative_2)[0]

    keep = auto_corr[peaks] >= threshold
    peaks = peaks[keep]
    keep = peaks < (auto_corr.shape[0] // 2)
    peaks = peaks[keep]
 
    if peaks.size == 0:
        # If none of the peaks crossed the threshold, redo with threshold/2.
        return get_unit_adaptive_window(auto_corr, threshold/2)

    # keep the last peak (nearest to center)
    win_size = auto_corr.shape[0] // 2 - peaks[-1]

    return win_size



def compute_templates_diff(sorting, templates, num_channels=5, num_shift=5, pair_mask=None):
    """
    Computes normalilzed template differences.

    Parameters
    ----------
    pair_mask: None or boolean array
        A bool matrix of size (num_units, num_units) to select
        which pair to compute.


    Returns
    -------
    templates_diff

    """
    unit_ids = sorting.unit_ids
    n = len(unit_ids)

    if pair_mask is None:
        pair_mask = np.ones((n, n), dtype='bool')

    templates_diff = np.full((n, n), np.nan, dtype='float64')
    for unit_ind1 in range(n):
        for unit_ind2 in range(unit_ind1 + 1, n):
            if not pair_mask[unit_ind1, unit_ind2]:
                continue
            
            template1 = templates[unit_ind1]
            template2 = templates[unit_ind2]
            # take best channels
            chan_inds = np.argsort(np.max(np.abs(template1 + template2), axis=0))[::-1][:num_channels]
            template1 = template1[:, chan_inds]
            template2 = template2[:, chan_inds]

            num_samples = template1.shape[0]
            norm = np.sum(np.abs(template1)) + np.sum(np.abs(template2))
            all_shift_diff = []
            for shift in range(-num_shift, num_shift+1):
                temp1 = template1[num_shift:num_samples - num_shift, :]
                temp2 = template2[num_shift + shift:num_samples - num_shift + shift, :]
                d = np.sum(np.abs(temp1 - temp2)) / (norm)
                all_shift_diff.append(d)
            templates_diff[unit_ind1, unit_ind2] = np.min(all_shift_diff)

    return templates_diff

class MockWaveformExtractor:
    def __init__(self, recording, sorting):
        self.recording = recording
        self.sorting = sorting

def check_improve_contaminations_score(we, pair_mask, contaminations,
        firing_contamination_balance, refractory_period_ms, censored_period_ms):
    """
    Check that the score is improve afeter a potential merge

    The score is a balance between:
      * contamination decrease
      * firing increase

    Check that the contamination score is improved (decrease)  after 
    a potential merge
    """
    recording = we.recording
    sorting = we.sorting
    pair_mask = pair_mask.copy()

    

    firing_rates = list(compute_firing_rate(we).values())

    inds1, inds2 = np.nonzero(pair_mask)
    for i in range(inds1.size):
        ind1, ind2 =inds1[i], inds2[i]

        c_1 = contaminations[ind1]
        c_2 = contaminations[ind2]

        f_1 = firing_rates[ind1]
        f_2 = firing_rates[ind2]

        # make a merged sorting and tale one unit (unit_id1 is used)
        unit_id1, unit_id2 = sorting.unit_ids[ind1], sorting.unit_ids[ind2]
        sorting_merged = MergeUnitsSorting(sorting, [unit_id1, unit_id2], new_unit_id=unit_id1).select_units([unit_id1])
        # make a lazy fake WaveformExtractor to compute contamination and firing rate
        we_new = MockWaveformExtractor(recording, sorting_merged)
        _, contaminations = compute_refrac_period_violations(we_new, refractory_period_ms=refractory_period_ms,
                                    censored_period_ms=censored_period_ms)
        c_new = contaminations[unit_id1]
        f_new = compute_firing_rate(we)[unit_id1]

        # old and new scores
        k = 1 + firing_contamination_balance
        score_1 = f_1 * ( 1 - k * c_1)
        score_2 = f_2 * ( 1 - k * c_2)
        score_new = f_new * ( 1 - k * c_new)

        if score_new < score_1 or score_new < score_2:
            # the score is not improved
            pair_mask[ind1, ind2] = False

    return pair_mask