import numpy as np

from .basecomparison import BaseTwoSorterComparison
from .comparisontools import make_possible_match, make_best_match, make_hungarian_match


class SymmetricSortingComparison(BaseTwoSorterComparison):
    """
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
    
    """

    def __init__(self, sorting1, sorting2, sorting1_name=None, sorting2_name=None,
                 delta_time=0.4, sampling_frequency=None, match_score=0.5, chance_score=0.1,
                 n_jobs=-1, verbose=False):
        BaseTwoSorterComparison.__init__(self, sorting1, sorting2, sorting1_name=sorting1_name,
                                         sorting2_name=sorting2_name,
                                         delta_time=delta_time,
                                         match_score=match_score, chance_score=chance_score,
                                         n_jobs=n_jobs, verbose=verbose)

    def _do_matching(self):
        if self._verbose:
            print("Matching...")

        self.possible_match_12, self.possible_match_21 = make_possible_match(self.agreement_scores, self.chance_score)
        self.best_match_12, self.best_match_21 = make_best_match(self.agreement_scores, self.chance_score)
        self.hungarian_match_12, self.hungarian_match_21 = make_hungarian_match(self.agreement_scores,
                                                                                self.match_score)

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


def compare_two_sorters(*args, **kwargs):
    return SymmetricSortingComparison(*args, **kwargs)


compare_two_sorters.__doc__ = SymmetricSortingComparison.__doc__
