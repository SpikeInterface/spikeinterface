import numpy as np
import pandas as pd

from spikeinterface.core.core_tools import define_function_from_class
from .basecomparison import BasePairComparison, MixinSpikeTrainComparison, MixinTemplateComparison
from .comparisontools import (do_count_event, make_match_count_matrix, 
                              make_agreement_scores_from_count, do_score_labels, do_confusion_matrix, 
                              do_count_score, compute_performance)
from ..postprocessing import compute_template_similarity


class BasePairSorterComparison(BasePairComparison, MixinSpikeTrainComparison):
    """
    Base class shared by SymmetricSortingComparison and GroundTruthComparison
    """

    def __init__(self, sorting1, sorting2, sorting1_name=None, sorting2_name=None,
                 delta_time=0.4, match_score=0.5, chance_score=0.1, n_jobs=1, 
                 verbose=False):
        if sorting1_name is None:
            sorting1_name = 'sorting1'
        if sorting2_name is None:
            sorting2_name = 'sorting2'
        assert sorting1.get_num_segments() == sorting2.get_num_segments(), ("The two sortings must have the same "
                                                                            "number of segments! ")

        BasePairComparison.__init__(self, object1=sorting1, object2=sorting2, 
                                    name1=sorting1_name, name2=sorting2_name,
                                    match_score=match_score, chance_score=chance_score, 
                                    verbose=verbose)
        MixinSpikeTrainComparison.__init__(self, delta_time=delta_time, n_jobs=n_jobs)
        self.set_frames_and_frequency(self.object_list)

        self.unit1_ids = self.sorting1.get_unit_ids()
        self.unit2_ids = self.sorting2.get_unit_ids()
        

        self._do_agreement()
        self._do_matching()

    @property
    def sorting1(self):
        return self.object_list[0]

    @property
    def sorting2(self):
        return self.object_list[1]

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
                                                         n_jobs=self.n_jobs)

        # agreement matrix score for each pair
        self.agreement_scores = make_agreement_scores_from_count(self.match_event_count, self.event_counts1,
                                                                 self.event_counts2)


class SymmetricSortingComparison(BasePairSorterComparison):
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
        BasePairSorterComparison.__init__(self, sorting1, sorting2, sorting1_name=sorting1_name,
                                          sorting2_name=sorting2_name,
                                          delta_time=delta_time,
                                          match_score=match_score, chance_score=chance_score,
                                          n_jobs=n_jobs, verbose=verbose)

    def get_matching(self):
        return self.hungarian_match_12, self.hungarian_match_21

    def get_matching_event_count(self, unit1, unit2):
        if (unit1 is not None) and (unit2 is not None):
            return self.match_event_count.at[unit1, unit2]
        else:
            raise Exception(
                'get_matching_event_count: unit1 and unit2 must not be None.')

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


compare_two_sorters = define_function_from_class(source_class=SymmetricSortingComparison, name="compare_two_sorters")


class GroundTruthComparison(BasePairSorterComparison):
    """
    Compares a sorter to a ground truth.

    This class can:
      * compute a "match between gt_sorting and tested_sorting
      * compute optionally the score label (TP, FN, CL, FP) for each spike
      * count by unit of GT the total of each (TP, FN, CL, FP) into a Dataframe
        GroundTruthComparison.count
      * compute the confusion matrix .get_confusion_matrix()
      * compute some performance metric with several strategy based on
        the count score by unit
      * count well detected units
      * count false positive detected units
      * count redundant units
      * count overmerged units
      * summary all this


    Parameters
    ----------
    gt_sorting: SortingExtractor
        The first sorting for the comparison
    tested_sorting: SortingExtractor
        The second sorting for the comparison
    gt_name: str
        The name of sorter 1
    tested_name: : str
        The name of sorter 2
    delta_time: float
        Number of ms to consider coincident spikes (default 0.4 ms)    match_score: float
        Minimum agreement score to match units (default 0.5)
    chance_score: float
        Minimum agreement score to for a possible match (default 0.1)
    redundant_score: float
        Agreement score above which units are redundant (default 0.2)
    overmerged_score: float
        Agreement score above which units can be overmerged (default 0.2)
    well_detected_score: float
        Agreement score above which units are well detected (default 0.8)
    exhaustive_gt: bool (default True)
        Tell if the ground true is "exhaustive" or not. In other world if the
        GT have all possible units. It allows more performance measurement.
        For instance, MEArec simulated dataset have exhaustive_gt=True
    match_mode: 'hungarian', or 'best'
        What is match used for counting : 'hungarian' or 'best match'.
    n_jobs: int
        Number of cores to use in parallel. Uses all available if -1
    compute_labels: bool
        If True, labels are computed at instantiation (default False)
    compute_misclassifications: bool
        If True, misclassifications are computed at instantiation (default False)
    verbose: bool
        If True, output is verbose

    Returns
    -------
    sorting_comparison: SortingComparison
        The SortingComparison object
    """

    def __init__(self, gt_sorting, tested_sorting, gt_name=None, tested_name=None,
                 delta_time=0.4, sampling_frequency=None, match_score=0.5, well_detected_score=0.8,
                 redundant_score=0.2, overmerged_score=0.2, chance_score=0.1, exhaustive_gt=False, n_jobs=-1,
                 match_mode='hungarian', compute_labels=False, compute_misclassifications=False, verbose=False):

        if gt_name is None:
            gt_name = 'ground truth'
        if tested_name is None:
            tested_name = 'tested'
        BasePairSorterComparison.__init__(self, gt_sorting, tested_sorting, sorting1_name=gt_name,
                                          sorting2_name=tested_name, delta_time=delta_time,
                                          match_score=match_score, chance_score=chance_score, 
                                          n_jobs=n_jobs, verbose=verbose)
        self.exhaustive_gt = exhaustive_gt

        self._compute_misclassifications = compute_misclassifications
        self.redundant_score = redundant_score
        self.overmerged_score = overmerged_score
        self.well_detected_score = well_detected_score

        assert match_mode in ['hungarian', 'best']
        self.match_mode = match_mode
        self._compute_labels = compute_labels

        self._do_count()

        self._labels_st1 = None
        self._labels_st2 = None
        if self._compute_labels:
            self._do_score_labels()

        # confusion matrix is compute on demand
        self._confusion_matrix = None

    def get_labels1(self, unit_id):
        if self._labels_st1 is None:
            self._do_score_labels()

        if unit_id in self.sorting1.get_unit_ids():
            return self._labels_st1[unit_id]
        else:
            raise Exception("Unit_id is not a valid unit")

    def get_labels2(self, unit_id):
        if self._labels_st1 is None:
            self._do_score_labels()

        if unit_id in self.sorting2.get_unit_ids():
            return self._labels_st2[unit_id]
        else:
            raise Exception("Unit_id is not a valid unit")

    def _do_count(self):
        """
        Do raw count into a dataframe.

        Internally use hungarian match or best match.
        """
        if self.match_mode == 'hungarian':
            match_12 = self.hungarian_match_12
        elif self.match_mode == 'best':
            match_12 = self.best_match_12

        self.count_score = do_count_score(self.event_counts1, self.event_counts2,
                                          match_12, self.match_event_count)

    def _do_confusion_matrix(self):
        if self._verbose:
            print("Computing confusion matrix...")

        if self.match_mode == 'hungarian':
            match_12 = self.hungarian_match_12
        elif self.match_mode == 'best':
            match_12 = self.best_match_12

        self._confusion_matrix = do_confusion_matrix(self.event_counts1, self.event_counts2, match_12,
                                                     self.match_event_count)

    def get_confusion_matrix(self):
        """
        Computes the confusion matrix.

        Returns
        -------
        confusion_matrix: pandas.DataFrame
            The confusion matrix
        """
        if self._confusion_matrix is None:
            self._do_confusion_matrix()
        return self._confusion_matrix

    def _do_score_labels(self):
        assert self.match_mode == 'hungarian', \
            'Labels (TP, FP, FN) can be computed only with hungarian match'

        if self._verbose:
            print("Adding labels...")

        self._labels_st1, self._labels_st2 = do_score_labels(self.sorting1, self.sorting2,
                                                             self.delta_frames, self.hungarian_match_12,
                                                             self._compute_misclassifications)

    def get_performance(self, method='by_unit', output='pandas'):
        """
        Get performance rate with several method:
          * 'raw_count' : just render the raw count table
          * 'by_unit' : render perf as rate unit by unit of the GT
          * 'pooled_with_average' : compute rate unit by unit and average

        Parameters
        ----------
        method: str
            'by_unit',  or 'pooled_with_average'
        output: str
            'pandas' or 'dict'

        Returns
        -------
        perf: pandas dataframe/series (or dict)
            dataframe/series (based on 'output') with performance entries
        """
        possibles = ('raw_count', 'by_unit', 'pooled_with_average')
        if method not in possibles:
            raise Exception("'method' can be " + ' or '.join(possibles))

        if method == 'raw_count':
            perf = self.count_score

        elif method == 'by_unit':
            perf = compute_performance(self.count_score)

        elif method == 'pooled_with_average':
            perf = self.get_performance(method='by_unit').mean(axis=0)

        if output == 'dict' and isinstance(perf, pd.Series):
            perf = perf.to_dict()

        return perf

    def print_performance(self, method='pooled_with_average'):
        """
        Print performance with the selected method
        """

        template_txt_performance = _template_txt_performance

        if method == 'by_unit':
            perf = self.get_performance(method=method, output='pandas')
            perf = perf * 100
            d = {k: perf[k].tolist() for k in perf.columns}
            txt = template_txt_performance.format(method=method, **d)
            print(txt)

        elif method == 'pooled_with_average':
            perf = self.get_performance(method=method, output='pandas')
            perf = perf * 100
            txt = template_txt_performance.format(
                method=method, **perf.to_dict())
            print(txt)

    def print_summary(self, well_detected_score=None, redundant_score=None, overmerged_score=None):
        """
        Print a global performance summary that depend on the context:
          * exhaustive= True/False
          * how many gt units (one or several)

        This summary mix several performance metrics.
        """
        txt = _template_summary_part1

        d = dict(
            num_gt=len(self.unit1_ids),
            num_tested=len(self.unit2_ids),
            num_well_detected=self.count_well_detected_units(
                well_detected_score),
            num_redundant=self.count_redundant_units(redundant_score),
            num_overmerged=self.count_overmerged_units(overmerged_score),
        )

        if self.exhaustive_gt:
            txt = txt + _template_summary_part2
            d['num_false_positive_units'] = self.count_false_positive_units()
            d['num_bad'] = self.count_bad_units()

        txt = txt.format(**d)

        print(txt)

    def get_well_detected_units(self, well_detected_score=None):
        """
        Return units list of "well detected units" from tested_sorting.

        "well detected units" are defined as units in tested that
        are well matched to GT units.

        Parameters
        ----------
        well_detected_score: float (default 0.8)
            The agreement score above which tested units
            are counted as "well detected".
        """
        if well_detected_score is not None:
            self.well_detected_score = well_detected_score

        matched_units2 = self.hungarian_match_12
        well_detected_ids = []
        for u2 in self.unit2_ids:
            if u2 in list(matched_units2.values):
                u1 = self.hungarian_match_21[u2]
                score = self.agreement_scores.at[u1, u2]
                if score >= self.well_detected_score:
                    well_detected_ids.append(u2)

        return well_detected_ids

    def count_well_detected_units(self, well_detected_score):
        """
        Count how many well detected units.
        kwargs are the same as get_well_detected_units.
        """
        return len(self.get_well_detected_units(well_detected_score=well_detected_score))

    def get_false_positive_units(self, redundant_score=None):
        """
        Return units list of "false positive units" from tested_sorting.

        "false positive units" are defined as units in tested that
        are not matched at all in GT units.

        Need exhaustive_gt=True

        Parameters
        ----------
        redundant_score: float (default 0.2)
            The agreement score below which tested units
            are counted as "false positive"" (and not "redundant").
        """
        assert self.exhaustive_gt, 'false_positive_units list is valid only if exhaustive_gt=True'

        if redundant_score is not None:
            self.redundant_score = redundant_score

        matched_units2 = list(self.hungarian_match_12.values)
        false_positive_ids = []
        for u2 in self.unit2_ids:
            if u2 not in matched_units2:
                if self.best_match_21[u2] == -1:
                    false_positive_ids.append(u2)
                else:
                    u1 = self.best_match_21[u2]
                    score = self.agreement_scores.at[u1, u2]
                    if score < self.redundant_score:
                        false_positive_ids.append(u2)

        return false_positive_ids

    def count_false_positive_units(self, redundant_score=None):
        """
        See get_false_positive_units().
        """
        return len(self.get_false_positive_units(redundant_score))

    def get_redundant_units(self, redundant_score=None):
        """
        Return "redundant units"

        "redundant units" are defined as units in tested
        that match a GT units with a big agreement score
        but it is not the best match.
        In other world units in GT that detected twice or more.

        Parameters
        ----------
        redundant_score=None: float (default 0.2)
            The agreement score above which tested units
            are counted as "redundant" (and not "false positive" ).
        """
        assert self.exhaustive_gt, 'redundant_units list is valid only if exhaustive_gt=True'

        if redundant_score is not None:
            self.redundant_score = redundant_score
        matched_units2 = list(self.hungarian_match_12.values)
        redundant_ids = []
        for u2 in self.unit2_ids:
            if u2 not in matched_units2 and self.best_match_21[u2] != -1:
                u1 = self.best_match_21[u2]
                if u2 != self.best_match_12[u1]:
                    score = self.agreement_scores.at[u1, u2]
                    if score >= self.redundant_score:
                        redundant_ids.append(u2)

        return redundant_ids

    def count_redundant_units(self, redundant_score=None):
        """
        See get_redundant_units().
        """
        return len(self.get_redundant_units(redundant_score=redundant_score))

    def get_overmerged_units(self, overmerged_score=None):
        """
        Return "overmerged units"

        "overmerged units" are defined as units in tested
        that match more than one GT unit with an agreement score larger than overmerged_score.

        Parameters
        ----------
        overmerged_score: float (default 0.4)
            Tested units with 2 or more agreement scores above 'overmerged_score'
            are counted as "overmerged".
        """
        assert self.exhaustive_gt, 'overmerged_units list is valid only if exhaustive_gt=True'

        if overmerged_score is not None:
            self.overmerged_score = overmerged_score

        overmerged_ids = []
        for u2 in self.unit2_ids:
            scores = self.agreement_scores.loc[:, u2]
            if len(np.where(scores > self.overmerged_score)[0]) > 1:
                overmerged_ids.append(u2)

        return overmerged_ids

    def count_overmerged_units(self, overmerged_score=None):
        """
        See get_overmerged_units().
        """
        return len(self.get_overmerged_units(overmerged_score=overmerged_score))

    def get_bad_units(self):
        """
        Return units list of "bad units".

        "bad units" are defined as units in tested that are not
        in the best match list of GT units.

        So it is the union of "false positive units" + "redundant units".

        Need exhaustive_gt=True
        """
        assert self.exhaustive_gt, 'bad_units list is valid only if exhaustive_gt=True'
        matched_units2 = list(self.hungarian_match_12.values)
        bad_ids = []
        for u2 in self.unit2_ids:
            if u2 not in matched_units2:
                bad_ids.append(u2)
        return bad_ids

    def count_bad_units(self):
        """
        See get_bad_units
        """
        return len(self.get_bad_units())


# usefull also for gathercomparison


_template_txt_performance = """PERFORMANCE ({method})
-----------
ACCURACY: {accuracy}
RECALL: {recall}
PRECISION: {precision}
FALSE DISCOVERY RATE: {false_discovery_rate}
MISS RATE: {miss_rate}
"""

_template_summary_part1 = """SUMMARY
-------
GT num_units: {num_gt}
TESTED num_units: {num_tested}
num_well_detected: {num_well_detected}
num_redundant: {num_redundant}
num_overmerged: {num_overmerged}
"""

_template_summary_part2 = """num_false_positive_units {num_false_positive_units}
num_bad: {num_bad}
"""


compare_sorter_to_ground_truth = define_function_from_class(source_class=GroundTruthComparison, 
                                                            name="compare_sorter_to_ground_truth")


class TemplateComparison(BasePairComparison, MixinTemplateComparison):
    """
    Compares units from different sessions based on template similarity

    Parameters
    ----------
    we1 : WaveformExtractor
        The first waveform extractor to get templates to compare
    we2 : WaveformExtractor
        The second waveform extractor to get templates to compare
    unit_ids1 : list, optional
        List of units from we1 to compare, by default None
    unit_ids2 : list, optional
        List of units from we2 to compare, by default None
    similarity_method : str, optional
        Method for the similaroty matrix, by default "cosine_similarity"
    sparsity_dict : dict, optional
        Dictionary for sparsity, by default None
    verbose : bool, optional
        If True, output is verbose, by default False

    Returns
    -------
    comparison : TemplateComparison
        The output TemplateComparison object
    """
    def __init__(self, we1, we2, we1_name=None, we2_name=None,
                 unit_ids1=None, unit_ids2=None,
                 match_score=0.7, chance_score=0.3,
                 similarity_method="cosine_similarity", sparsity_dict=None,
                 verbose=False):
        if we1_name is None:
            we1_name = "sess1"
        if we2_name is None:
            we2_name = "sess2"
        BasePairComparison.__init__(self, object1=we1, object2=we2,
                                    name1=we1_name, name2=we2_name,
                                    match_score=match_score, chance_score=chance_score,
                                    verbose=verbose)
        MixinTemplateComparison.__init__(self, similarity_method=similarity_method, sparsity_dict=sparsity_dict)

        self.we1 = we1
        self.we2 = we2
        channel_ids1 = we1.recording.get_channel_ids()
        channel_ids2 = we2.recording.get_channel_ids()

        # two options: all channels are shared or partial channels are shared
        if we1.recording.get_num_channels() != we2.recording.get_num_channels():
            raise NotImplementedError
        if np.any([ch1 != ch2 for (ch1, ch2) in zip(channel_ids1, channel_ids2)]):
            # TODO: here we can check location and run it on the union. Might be useful for reconfigurable probes
            raise NotImplementedError

        self.matches = dict()

        if unit_ids1 is None:
            unit_ids1 = we1.sorting.get_unit_ids()

        if unit_ids2 is None:
            unit_ids2 = we2.sorting.get_unit_ids()
        self.unit_ids = [unit_ids1, unit_ids2]

        if sparsity_dict is not None:
            raise NotImplementedError
        else:
            self.sparsity = None

        self._do_agreement()
        self._do_matching()

    def _do_agreement(self):
        if self._verbose:
            print('Agreement scores...')

        agreement_scores = compute_template_similarity(self.we1, self.we2,
                                                       method=self.similarity_method)
        self.agreement_scores = pd.DataFrame(agreement_scores,
                                             index=self.unit_ids[0],
                                             columns=self.unit_ids[1])


compare_templates = define_function_from_class(source_class=TemplateComparison, name="compare_templates")
