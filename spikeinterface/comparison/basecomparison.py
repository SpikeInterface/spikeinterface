import numpy as np
import pandas as pd
from .comparisontools import (do_count_event, make_match_count_matrix, make_agreement_scores_from_count)


class BaseComparison:
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

    def __init__(self, sorting_list, name_list=None, delta_time=0.4,  # sampling_frequency=None,
                 match_score=0.5, chance_score=0.1, n_jobs=-1, verbose=False):

        self.sorting_list = sorting_list
        if name_list is None:
            name_list = ['sorting{}'.format(i + 1) for i in range(len(sorting_list))]
        self.name_list = name_list
        if np.any(['_' in name for name in name_list]):
            raise ValueError("Sorter names in 'name_list' cannot contain '_'")

        if not np.all(s.get_num_segments() == 1 for s in sorting_list):
            raise Exception('Comparison module work with sorting having num_segments=1')

        # take sampling frequency from sorting list and test that they are equivalent.
        sampling_freqs = np.array([s.get_sampling_frequency() for s in self.sorting_list], dtype='float64')

        # Some sorter round the sampling freq lets emit a warning
        sf0 = sampling_freqs[0]
        if not np.all(sf0 == sampling_freqs):
            delta_freq_ratio = np.abs(sampling_freqs - sf0) / sf0
            # tolerence of 0.1%
            assert np.all(delta_freq_ratio < 0.001), "Inconsintent sampling frequency among sorting list"

        self.sampling_frequency = sf0
        self.delta_time = delta_time
        self.delta_frames = int(self.delta_time / 1000 * self.sampling_frequency)
        self.match_score = match_score
        self.chance_score = chance_score
        self._n_jobs = n_jobs
        self._verbose = verbose


class BaseTwoSorterComparison(BaseComparison):
    """
    Base class shared by SortingComparison and GroundTruthComparison
    """

    def __init__(self, sorting1, sorting2, sorting1_name=None, sorting2_name=None,
                 delta_time=0.4, match_score=0.5,
                 chance_score=0.1, n_jobs=1, verbose=False):
        # sampling_frequency=None

        sorting_list = [sorting1, sorting2]
        if sorting1_name is None:
            sorting1_name = 'sorting1'
        if sorting2_name is None:
            sorting2_name = 'sorting2'
        name_list = [sorting1_name, sorting2_name]

        BaseComparison.__init__(self, sorting_list, name_list=name_list, delta_time=delta_time,
                                match_score=match_score,
                                chance_score=chance_score, verbose=verbose, n_jobs=n_jobs)
        # sampling_frequency=sampling_frequency,

        self.unit1_ids = self.sorting1.get_unit_ids()
        self.unit2_ids = self.sorting2.get_unit_ids()

        self._do_agreement()
        self._do_matching()

    @property
    def sorting1(self):
        return self.sorting_list[0]

    @property
    def sorting2(self):
        return self.sorting_list[1]

    @property
    def sorting1_name(self):
        return self.name_list[0]

    @property
    def sorting2_name(self):
        return self.name_list[1]

    def _do_agreement(self):
        if self._verbose:
            print('Agreement scores...')

        # common to GroundTruthComparison and SymmetricSortingComparison
        # spike count for each spike train
        self.event_counts1 = do_count_event(self.sorting1)
        self.event_counts2 = do_count_event(self.sorting2)

        # matrix of  event match count for each pair
        self.match_event_count = make_match_count_matrix(self.sorting1, self.sorting2, self.delta_frames,
                                                         n_jobs=self._n_jobs)

        # agreement matrix score for each pair
        self.agreement_scores = make_agreement_scores_from_count(self.match_event_count, self.event_counts1,
                                                                 self.event_counts2)

    def _do_matching(self):
        # must be implemented in subclass
        raise NotImplementedError

    def get_ordered_agreement_scores(self):
        # order rows
        order0 = self.agreement_scores.max(axis=1).argsort()
        scores = self.agreement_scores.iloc[order0.values[::-1], :]

        # order columns
        indexes = np.arange(scores.shape[1])
        order1 = []
        for r in range(scores.shape[0]):
            possible = indexes[~np.in1d(indexes, order1)]
            if possible.size > 0:
                ind = np.argmax(scores.iloc[r, possible].values)
                order1.append(possible[ind])
        remain = indexes[~np.in1d(indexes, order1)]
        order1.extend(remain)
        scores = scores.iloc[:, order1]

        return scores
