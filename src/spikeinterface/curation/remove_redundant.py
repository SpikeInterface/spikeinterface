from __future__ import annotations
import numpy as np
from spikeinterface import BaseSorting

from spikeinterface import SortingAnalyzer

from spikeinterface.core.template_tools import get_template_extremum_channel_peak_shift, get_template_amplitudes
from spikeinterface.postprocessing import align_sorting


_remove_strategies = ("minimum_shift", "highest_amplitude", "max_spikes")


def remove_redundant_units(
    sorting_or_sorting_analyzer,
    align=True,
    unit_peak_shifts=None,
    delta_time=0.4,
    agreement_threshold=0.2,
    duplicate_threshold=0.8,
    remove_strategy="minimum_shift",
    peak_sign="neg",
    extra_outputs=False,
) -> BaseSorting:
    """
    Removes redundant or duplicate units by comparing the sorting output with itself.

    When a redundant pair is found, there are several strategies to choose which unit is the best:

       * "minimum_shift"
       * "highest_amplitude"
       * "max_spikes"


    Parameters
    ----------
    sorting_or_sorting_analyzer : BaseSorting or SortingAnalyzer
        If SortingAnalyzer, the spike trains can be optionally realigned using the peak shift in the
        template to improve the matching procedure.
        If BaseSorting, the spike trains are not aligned.
    align : bool, default: False
        If True, spike trains are aligned (if a SortingAnalyzer is used)
    delta_time : float, default: 0.4
        The time in ms to consider matching spikes
    agreement_threshold : float, default: 0.2
        Threshold on the agreement scores to flag possible redundant/duplicate units
    duplicate_threshold : float, default: 0.8
        Final threshold on the portion of coincident events over the number of spikes above which the
        unit is removed
    remove_strategy : "minimum_shift" | "highest_amplitude" | "max_spikes", default: "minimum_shift"
        Which strategy to remove one of the two duplicated units:

            * "minimum_shift" : keep the unit with best peak alignment (minimum shift)
                             If shifts are equal then the "highest_amplitude" is used
            * "highest_amplitude" : keep the unit with the best amplitude on unshifted max.
            * "max_spikes" : keep the unit with more spikes

    peak_sign : "neg" | "pos" | "both", default: "neg"
        Used when remove_strategy="highest_amplitude"
    extra_outputs : bool, default: False
        If True, will return the redundant pairs.
    unit_peak_shifts : dict
        Dictionary mapping the unit_id to the unit's shift (in number of samples).
        A positive shift means the spike train is shifted back in time, while
        a negative shift means the spike train is shifted forward.

    Returns
    -------
    BaseSorting
        Sorting object without redundant units
    """

    if isinstance(sorting_or_sorting_analyzer, SortingAnalyzer):
        sorting = sorting_or_sorting_analyzer.sorting
        sorting_analyzer = sorting_or_sorting_analyzer
    else:
        assert not align, "The 'align' option is only available when a SortingAnalyzer is used as input"
        # other remove strategies rely on sorting analyzer looking at templates
        assert remove_strategy == "max_spikes", "For a Sorting input the remove_strategy must be 'max_spikes'"
        sorting = sorting_or_sorting_analyzer
        sorting_analyzer = None

    if align and unit_peak_shifts is None:
        assert sorting_analyzer is not None, "For align=True must give a SortingAnalyzer or explicit unit_peak_shifts"
        unit_peak_shifts = get_template_extremum_channel_peak_shift(sorting_analyzer)

    if align:
        sorting_aligned = align_sorting(sorting, unit_peak_shifts)
    else:
        sorting_aligned = sorting

    redundant_unit_pairs = find_redundant_units(
        sorting_aligned,
        delta_time=delta_time,
        agreement_threshold=agreement_threshold,
        duplicate_threshold=duplicate_threshold,
    )

    remove_unit_ids = []

    if remove_strategy in ("minimum_shift", "highest_amplitude"):
        # this is the values at spike index !
        peak_values = get_template_amplitudes(sorting_analyzer, peak_sign=peak_sign, mode="at_index")
        peak_values = {unit_id: np.max(np.abs(values)) for unit_id, values in peak_values.items()}

    if remove_strategy == "minimum_shift":
        assert align, "remove_strategy with minimum_shift needs align=True"
        for u1, u2 in redundant_unit_pairs:
            if np.abs(unit_peak_shifts[u1]) > np.abs(unit_peak_shifts[u2]):
                remove_unit_ids.append(u1)
            elif np.abs(unit_peak_shifts[u1]) < np.abs(unit_peak_shifts[u2]):
                remove_unit_ids.append(u2)
            else:
                # equal shift use peak values
                if np.abs(peak_values[u1]) < np.abs(peak_values[u2]):
                    remove_unit_ids.append(u1)
                else:
                    remove_unit_ids.append(u2)
    elif remove_strategy == "highest_amplitude":
        for u1, u2 in redundant_unit_pairs:
            if np.abs(peak_values[u1]) < np.abs(peak_values[u2]):
                remove_unit_ids.append(u1)
            else:
                remove_unit_ids.append(u2)
    elif remove_strategy == "max_spikes":
        num_spikes = sorting.count_num_spikes_per_unit(outputs="dict")
        for u1, u2 in redundant_unit_pairs:
            if num_spikes[u1] < num_spikes[u2]:
                remove_unit_ids.append(u1)
            else:
                remove_unit_ids.append(u2)
    elif remove_strategy == "with_metrics":
        # TODO
        # @aurelien @alessio
        # here sorting_analyzer can implement the choice of the best one given an external metrics table
        # this will be implemented in a futur PR by the first who need it!
        raise NotImplementedError()
    else:
        raise ValueError(f"remove_strategy : {remove_strategy} is not implemented! Options are {_remove_strategies}")

    sorting_clean = sorting.remove_units(remove_unit_ids)

    if extra_outputs:
        return sorting_clean, redundant_unit_pairs
    else:
        return sorting_clean


def find_redundant_units(sorting, delta_time: float = 0.4, agreement_threshold=0.2, duplicate_threshold=0.8):
    """
    Finds redundant or duplicate units by comparing the sorting output with itself.

    Parameters
    ----------
    sorting : BaseSorting
        The input sorting object
    delta_time : float, default: 0.4
        The time in ms to consider matching spikes
    agreement_threshold : float, default: 0.2
        Threshold on the agreement scores to flag possible redundant/duplicate units
    duplicate_threshold : float, default: 0.8
        Final threshold on the portion of coincident events over the number of spikes above which the
        unit is flagged as duplicate/redundant

    Returns
    -------
    list
        The list of duplicate units
    list of 2-element lists
        The list of duplicate pairs
    """
    from spikeinterface.comparison import compare_two_sorters

    comparison = compare_two_sorters(sorting, sorting, delta_time=delta_time)

    # make agreement triangular and exclude diagonal
    agreement_scores_cleaned = np.tril(comparison.agreement_scores.values, k=-1)

    possible_pairs = np.where(agreement_scores_cleaned >= agreement_threshold)

    rows, cols = possible_pairs
    redundant_unit_pairs = []
    for r, c in zip(rows, cols):
        unit_i = sorting.unit_ids[r]
        unit_j = sorting.unit_ids[c]

        n_coincidents = comparison.match_event_count.at[unit_i, unit_j]
        event_counts = comparison.event_counts1
        shared = max(n_coincidents / event_counts[unit_i], n_coincidents / event_counts[unit_j])
        if shared > duplicate_threshold:
            redundant_unit_pairs.append([unit_i, unit_j])

    return redundant_unit_pairs
