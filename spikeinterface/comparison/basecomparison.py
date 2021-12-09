import numpy as np
from .comparisontools import (make_possible_match, make_best_match, make_hungarian_match,
                              do_count_event, make_match_count_matrix, make_agreement_scores_from_count)


class BaseComparison:
    """
    Base class for all comparisons (SpikeTrain and Template)
    """

    def __init__(self, name_list,
                 match_score=0.5, chance_score=0.1,
                 verbose=False):
        self.name_list = name_list
        self._verbose = verbose
        self.match_score = match_score
        self.chance_score = chance_score
        self.possible_match_12, self.possible_match_21 = None, None
        self.best_match_12, self.best_match_21 = None, None
        self.hungarian_match_12, self.hungarian_match_21 = None, None

    def _do_agreement(self):
        # populate self.agreement_scores
        NotImplementedError

    def _do_matching(self):
        if self._verbose:
            print("Matching...")

        self.possible_match_12, self.possible_match_21 = make_possible_match(self.agreement_scores, self.chance_score)
        self.best_match_12, self.best_match_21 = make_best_match(self.agreement_scores, self.chance_score)
        self.hungarian_match_12, self.hungarian_match_21 = make_hungarian_match(self.agreement_scores, self.match_score)


class BaseMultiComparison(BaseComparison):
    pass


class BasePairComparison(BaseComparison):
    pass


class MixinSpikeTrainComparison:
    """
    Base class for all comparison classes:
       * GroundTruthComparison
       * MultiSortingComparison
       * SymmetricSortingComparison

    Mainly deals with:
      * sampling_frequency
      * sorting names
      * delta_time to delta_frames
    """
    def __init__(self, delta_time=0.4, n_jobs=-1):
        # self.sorting_list = sorting_list
        # if name_list is None:
        #     name_list = ['sorting{}'.format(i + 1)
        #                  for i in range(len(sorting_list))]
        # BaseComparison.__init__(self, name_list=name_list,
        #                         match_score=match_score, chance_score=chance_score,
        #                         verbose=verbose)
        self.delta_time = delta_time
        self.n_jobs = n_jobs

        # self.name_list = name_list
        # if np.any(['_' in name for name in name_list]):
        #     raise ValueError("Sorter names in 'name_list' cannot contain '_'")

        # if not np.all(s.get_num_segments() == 1 for s in sorting_list):
        #     raise Exception('Comparison module work with sorting having num_segments=1')

        # # take sampling frequency from sorting list and test that they are equivalent.
        # sampling_freqs = np.array([s.get_sampling_frequency() for s in self.sorting_list], dtype='float64')

        # # Some sorter round the sampling freq lets emit a warning
        # sf0 = sampling_freqs[0]
        # if not np.all(sf0 == sampling_freqs):
        #     delta_freq_ratio = np.abs(sampling_freqs - sf0) / sf0
        #     # tolerance of 0.1%
        #     assert np.all(delta_freq_ratio < 0.001), "Inconsistent sampling frequency among sorting list"

        # self.sampling_frequency = sf0
        # self.delta_time = delta_time
        # self.delta_frames = int(self.delta_time / 1000 * self.sampling_frequency)
        # self._n_jobs = n_jobs


class MixinTemplateComparison:
    pass


