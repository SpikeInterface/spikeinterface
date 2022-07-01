import numpy as np
from typing import Union

from spikeinterface import WaveformExtractor, BaseSorting

from ..postprocessing import align_sorting


def remove_redundant_units(sorting_or_waveform_extractor : Union[BaseSorting, WaveformExtractor], 
                           align : bool=True, 
                           delta_time: float=0.4,
                           agreement_threshold : float=0.2,
                           duplicate_threshold : float=0.8,
                           verbose : bool=False):
    """
    Removes redundant or duplicate units by comparing the sorting output with itself.
    
    When a redundant pair is found, the unit with the least number of spikes is removed.
    

    Parameters
    ----------
    sorting_or_waveform_extractor : BaseSorting or WaveformExtractor
        If WaveformExtractor, the spike trains can be optionally realigned using the peak shift in the
        template to improve the matching procedure.
        If BaseSorting, the spike trains are not aligned.
    align : bool, optional
        If True, spike trains are aligned (if a WaveformExtractor is used), by default False
    delta_time : float, optional
        The time in ms to consider matching spikes, by default 0.4
    agreement_threshold : float, optional
        Threshold on the agreement scores to flag possible redundant/duplicate units, by default 0.2
    duplicate_threshold : float, optional
        Final threshold on the portion of coincident events over the number of spikes above which the  
        unit is removed, by default 0.8
    """
    
    if isinstance(sorting_or_waveform_extractor, WaveformExtractor):
        if align:
            sorting_aligned = align_sorting(sorting_or_waveform_extractor)
        else:
            sorting_aligned = sorting_or_waveform_extractor.sorting
        sorting = sorting_or_waveform_extractor.sorting
    else:
        sorting_aligned = sorting_or_waveform_extractor
        sorting = sorting_or_waveform_extractor
        
    remove_unit_ids, _ = find_redundant_units(sorting_aligned, 
                                              delta_time=delta_time,
                                              agreement_threshold=agreement_threshold,
                                              duplicate_threshold=duplicate_threshold)
    
    if verbose:
        print(f"Removing {len(remove_unit_ids)} duplicate units: {remove_unit_ids}")
    
    sorting_removed = sorting.remove_units(remove_unit_ids)
    
    return sorting_removed
    
    
def find_redundant_units(sorting: BaseSorting, 
                         delta_time: float=0.4,
                         agreement_threshold : float=0.2,
                         duplicate_threshold : float=0.8):
    """
    Finds redundant or duplicate units by comparing the sorting output with itself.

    Parameters
    ----------
    sorting : BaseSorting
        The input sorting object
    delta_time : float, optional
        The time in ms to consider matching spikes, by default 0.4
    agreement_threshold : float, optional
        Threshold on the agreement scores to flag possible redundant/duplicate units, by default 0.2
    duplicate_threshold : float, optional
        Final threshold on the portion of coincident events over the number of spikes above which the  
        unit is flagged as duplicate/redundant, by default 0.8

    Returns
    -------
    list
        The list of duplicate units
    list of 2-element lists
        The list of duplicate pairs
    """
    from spikeinterface.comparison import compare_two_sorters
    
    comparison = compare_two_sorters(sorting, sorting,
                                     delta_time=delta_time)
    
    # make agreement triangular and exclude diagonal
    agreement_scores_cleaned = np.tril(comparison.agreement_scores.values, k=-1)
    
    possible_pairs = np.where(agreement_scores_cleaned >= agreement_threshold)
    
    rows, cols = possible_pairs
    redundant_unit_ids = []
    redundant_unit_pairs = []
    for r, c in zip(rows, cols):
        unit_i = sorting.unit_ids[r]
        unit_j = sorting.unit_ids[c]
        
        n_coincidents = comparison.match_event_count.at[unit_i, unit_j]
        event_counts = comparison.event_counts1
        shared = max(n_coincidents / event_counts[unit_i], n_coincidents / event_counts[unit_j])
        
        if shared > duplicate_threshold:
            redundant_unit_ids.append([unit_i, unit_j][np.argmin([event_counts[unit_i], event_counts[unit_j]])])
            redundant_unit_pairs.append([unit_i, unit_j])
            
    return redundant_unit_ids, redundant_unit_pairs
