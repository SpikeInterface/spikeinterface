import numpy as np

from .basecomparison import BaseTwoSorterComparison
from .comparisontools import make_possible_match, make_best_match, make_hungarian_match


class SymmetricSortingComparison(BaseTwoSorterComparison):
    """
    Class for symmetric comparison of two sorters when no assumption is done.
    """

    def __init__(self, sorting1, sorting2, sorting1_name=None, sorting2_name=None,
                 delta_time=0.4, sampling_frequency=None, match_score=0.5, chance_score=0.1,
                 n_jobs=-1, verbose=False):
        BaseTwoSorterComparison.__init__(self, sorting1, sorting2, sorting1_name=sorting1_name,
                                         sorting2_name=sorting2_name,
                                         delta_time=delta_time, # sampling_frequency=sampling_frequency,
                                         match_score=match_score, chance_score=chance_score,
                                         n_jobs=n_jobs, verbose=verbose)

    def _do_matching(self):
        if self._verbose:
            print("Matching...")

        self.possible_match_12, self.possible_match_21 = make_possible_match(self.agreement_scores, self.chance_score)
        self.best_match_12, self.best_match_21 = make_best_match(self.agreement_scores, self.chance_score)
        self.hungarian_match_12, self.hungarian_match_21 = make_hungarian_match(self.agreement_scores,
                                                                                self.match_score)

    #~ def get_mapped_sorting1(self):
        #~ """
        #~ Returns a MappedSortingExtractor for sorting 1.

        #~ The returned MappedSortingExtractor.get_unit_ids returns the unit_ids of sorting 1.

        #~ The returned MappedSortingExtractor.get_mapped_unit_ids returns the mapped unit_ids
        #~ of sorting 2 to the units of sorting 1 (if units are not mapped they are labeled as -1).

        #~ The returned MappedSortingExtractor.get_unit_spikeTrains returns the the spike trains
        #~ of sorting 2 mapped to the unit_ids of sorting 1.
        #~ """
        #~ return MappedSortingExtractor(self.sorting2, self.hungarian_match_12.to_dict())

    #~ def get_mapped_sorting2(self):
        #~ """
        #~ Returns a MappedSortingExtractor for sorting 2.

        #~ The returned MappedSortingExtractor.get_unit_ids returns the unit_ids of sorting 2.

        #~ The returned MappedSortingExtractor.get_mapped_unit_ids returns the mapped unit_ids
        #~ of sorting 1 to the units of sorting 2 (if units are not mapped they are labeled as -1).

        #~ The returned MappedSortingExtractor.get_unit_spikeTrains returns the the spike trains
        #~ of sorting 1 mapped to the unit_ids of sorting 2.
        #~ """
        #~ return MappedSortingExtractor(self.sorting1, self.hungarian_match_21.to_dict())
    
    def get_matching(self):
        return self.hungarian_match_12, self.hungarian_match_21
    
    def get_matching_event_count(self, unit1, unit2):
        if (unit1 is not None) and (unit2 is not None):
            return self.match_event_count.at[unit1, unit2]
        else:
            raise Exception('get_matching_event_count: unit1 and unit2 must not be None.')

    def get_best_unit_match1(self, unit1):
        return self.best_match_12[unit1]

    def get_best_unit_match2(self, unit2):
        return self.best_match_21[unit2]

    def get_matching_unit_list1(self, unit1):
        return self.possible_match_12[unit1]

    def get_matching_unit_list2(self, unit2):
        return self.possible_match_21[unit2]

    def get_agreement_fraction(self, unit1=None, unit2=None):
        if unit1 is None or unit1 == -1 or unit2 is None or unit2 == -1:
            return 0
        else:
            return self.agreement_scores.at[unit1, unit2]


#~ class MappedSortingExtractor(se.SortingExtractor):
    #~ def __init__(self, sorting, unit_map):
        #~ se.SortingExtractor.__init__(self)
        #~ self._sorting = sorting
        #~ self._unit_map = unit_map
        #~ self._unit_ids = list(self._unit_map.keys())

    #~ def get_unit_ids(self, unit_ids=None):
        #~ if unit_ids is None:
            #~ return self._unit_ids
        #~ else:
            #~ return self._unit_ids[unit_ids]

    #~ def get_mapped_unit_ids(self, unit_ids=None):
        #~ if unit_ids is None:
            #~ return list(self._unit_map.values())
        #~ elif isinstance(unit_ids, (int, np.integer)):
            #~ return self._unit_map[unit_ids]
        #~ else:
            #~ return list([self._unit_map[u] for u in self._unit_ids if u in unit_ids])

    #~ def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        #~ unit2 = self._unit_map[unit_id]
        #~ if unit2 != -1:
            #~ return self._sorting.get_unit_spike_train(unit2, start_frame=start_frame, end_frame=end_frame)
        #~ else:
            #~ print(unit_id, " is not matched!")
            #~ return None


def compare_two_sorters(sorting1, sorting2, sorting1_name=None, sorting2_name=None, delta_time=0.4,
                        match_score=0.5, chance_score=0.1, n_jobs=-1, verbose=False):
                            # sampling_frequency=None, 
    '''
    Compares two spike sorter outputs.

    - Spike trains are matched based on their agreement scores
    - Individual spikes are labelled as true positives (TP), false negatives (FN), false positives 1 (FP from spike
      train 1), false positives 2 (FP from spike train 2), misclassifications (CL)

    It also allows to get confusion matrix and agreement fraction, false positive fraction and
    false negative fraction.

    Parameters
    ----------
    sorting1: SortingExtractor
        The first sorting for the comparison
    sorting2: SortingExtractor
        The second sorting for the comparison
    sorting1_name: str
        The name of sorter 1
    sorting2_name: : str
        The name of sorter 2
    delta_time: float
        Number of ms to consider coincident spikes (default 0.4 ms)
    match_score: float
        Minimum agreement score to match units (default 0.5)
    chance_score: float
        Minimum agreement score to for a possible match (default 0.1)
    n_jobs: int
        Number of cores to use in parallel. Uses all available if -1
    verbose: bool
        If True, output is verbose
    Returns
    -------
    sorting_comparison: SortingComparison
        The SortingComparison object

    '''
    return SymmetricSortingComparison(sorting1=sorting1, sorting2=sorting2, sorting1_name=sorting1_name,
                                      sorting2_name=sorting2_name, delta_time=delta_time,
                                      #sampling_frequency=sampling_frequency,
                                      match_score=match_score, chance_score=chance_score,
                                      n_jobs=n_jobs, verbose=verbose)
