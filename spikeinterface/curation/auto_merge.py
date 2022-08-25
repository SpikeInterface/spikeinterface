import numpy as np
import scipy.signal

from ..postprocessing import compute_correlograms

def get_potential_auto_merge(waveform_extractor,
                bin_ms=1., window_ms=50., corr_thresh=0.3,
                
                ):
    """
    Algorithm to find and check potential merges between units.
    
    This is taken from lussac version 1 done Aurelien Wyngaard.
    https://github.com/BarbourLab/lussac/blob/main/postprocessing/merge_units.py
    
    This check:
       * correlograms
       * template_similarity
       * contamination quality mertics
    
    """
    
    we = waveform_extractor
    sorting = we.sorting


    correlograms, bins = compute_correlograms(sorting, window_ms=window_ms, bin_ms=bin_ms, method='numba')
    #~ print(correlograms.shape)
    #~ import matplotlib.pyplot as plt
    
    #~ for i in range(correlograms.shape[0]):
        #~ fig, ax = plt.subplots()
        #~ ax.plot(correlograms[i, i, :])
        #~ plt.show()
    
    corr_diff = compute_correlogram_diff(sorting, correlograms, bins)
    
    print(corr_diff)
    ind1, ind2 = np.nonzero(corr_diff  < corr_thresh)
    potential_merges_cc = list(zip(ind1, ind2))
    print(potential_merges_cc)
    
    
    potential_merges = potential_merges_cc
    
    return potential_merges


def compute_correlogram_diff(sorting, correlograms, bins, pair_mask=None):
    """
    Original Author : Aurelien Wyngaard
    
    Parameters
    ----------
    sorting: BaseSorting
        The sorting object
    correlograms: array 3d
        The 3d array containing all cross and auto corrlogram
    bins: array
        Bins of the correlograms
    pair_mask: None or boolean array
        A bool matrix of size (num_units, num_units) to select
        which pair to compute.
    
    Returns
    -------
    corr_diff
    """
    bin_ms = bins[1] - bins[0] 
    
    unit_ids = sorting.unit_ids
    n = len(unit_ids)
    
    if pair_mask is None:
        pair_mask = np.ones((n, n), dtype='bool')
    
    
    # Index of the middle of the correlograms.
    m = correlograms.shape[2] // 2
    
    num_spikes = sorting.get_total_num_spikes()
    
    # smooth correlogram by low pass filter
    correlogram_low_pass = 300.
    b, a = scipy.signal.butter(N=2, Wn = correlogram_low_pass / (1e3 / bin_ms /2), btype='low')
    correlograms_smooth = scipy.signal.filtfilt(b, a, correlograms, axis=2)
    
    
    #TODO make adaptative window sizes
    win_sizes = np.ones(n, dtype=int) * 20
    
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

            auto_corr1 = normalize_correlogram(correlograms_smooth[unit_ind1, unit_ind1, :])
            auto_corr2 = normalize_correlogram(correlograms_smooth[unit_ind2, unit_ind2, :])
            cross_corr = normalize_correlogram(correlograms_smooth[unit_ind1, unit_ind2, :])
            diff1 = np.sum(np.abs(cross_corr[corr_inds - shift] - auto_corr1[corr_inds])) / len(corr_inds)
            diff2 = np.sum(np.abs(cross_corr[corr_inds - shift] - auto_corr2[corr_inds])) / len(corr_inds)
            # Weighted difference (larger unit imposes its difference).
            w_diff = (num1 * diff1 + num2 * diff2) / (num1 + num2)
            corr_diff[unit_ind1, unit_ind2] = w_diff
            
            
            # debug
            corr_thresh = 0.3
            if w_diff < corr_thresh:
                import matplotlib.pyplot as plt
                bins2 = bins[:-1] + np.mean(np.diff(bins))
                fig, axs = plt.subplots(ncols=3)
                ax = axs[0]
                ax.plot(bins2, correlograms[unit_ind1, unit_ind1, :], color='b')
                ax.plot(bins2, correlograms[unit_ind2, unit_ind2, :], color='r')
                ax.plot(bins2, correlograms_smooth[unit_ind1, unit_ind1, :], color='b')
                ax.plot(bins2, correlograms_smooth[unit_ind2, unit_ind2, :], color='r')
                
                
                ax.set_title(f'{unit_id1} {unit_id2}')
                ax = axs[1]
                ax.plot(bins2, correlograms[unit_ind1, unit_ind2, :], color='g')
                ax = axs[2]
                ax.plot(bins2, auto_corr1, color='b')
                ax.plot(bins2, auto_corr2, color='r')
                ax.axvline(bins2[m - win_size])
                ax.axvline(bins2[m + win_size])
                ax.set_title(f'corr diff {w_diff}')
                plt.show()
    
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



    