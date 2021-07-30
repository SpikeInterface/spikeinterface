import pandas as pd
import numpy as np
from .basecomparison import BaseTwoSorterComparison
from .comparisontools import (do_score_labels, make_possible_match,
                              make_best_match, make_hungarian_match, do_confusion_matrix, do_count_score,
                              compute_performance)


class GroundTruthComparison(BaseTwoSorterComparison):
    """
    Compares a sorter to a ground truth.

    This class can:
      * compute a "macth between gt_sorting and tested_sorting
      * compute optionally the score label (TP, FN, CL, FP) for each spike
      * count by unit of GT the total of each (TP, FN, CL, FP) into a Dataframe 
        GroundTruthComparison.count
      * compute the confusion matrix .get_confusion_matrix()
      * compute some performance metric with several strategy based on 
        the count score by unit
      * count well detected units
      * count false positve detected units
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
        What is match used for counting : 'hugarian' or 'best match'.
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
        BaseTwoSorterComparison.__init__(self, gt_sorting, tested_sorting, sorting1_name=gt_name,
                                         sorting2_name=tested_name, delta_time=delta_time,
                                         match_score=match_score,  # sampling_frequency=sampling_frequency,
                                         chance_score=chance_score, n_jobs=n_jobs,
                                         verbose=verbose)
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

    def _do_matching(self):
        if self._verbose:
            print("Matching...")

        self.possible_match_12, self.possible_match_21 = make_possible_match(self.agreement_scores, self.chance_score)
        self.best_match_12, self.best_match_21 = make_best_match(self.agreement_scores, self.chance_score)
        self.hungarian_match_12, self.hungarian_match_21 = make_hungarian_match(self.agreement_scores,
                                                                                self.match_score)

    def _do_count(self):
        """
        Do raw count into a dataframe.
        
        Internally use hugarian match or best match.
        
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
            # ~ print(perf)
            d = {k: perf[k].tolist() for k in perf.columns}
            txt = template_txt_performance.format(method=method, **d)
            print(txt)

        elif method == 'pooled_with_average':
            perf = self.get_performance(method=method, output='pandas')
            perf = perf * 100
            txt = template_txt_performance.format(method=method, **perf.to_dict())
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
            num_well_detected=self.count_well_detected_units(well_detected_score),
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

        "well detected units" ara defined as units in tested that
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
        
        "false positive units" ara defined as units in tested that
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
            Tested units with 2 or more agrement scores above 'overmerged_score'
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


def compare_sorter_to_ground_truth(*args, **kwargs):
    return GroundTruthComparison(*args, **kwargs)


compare_sorter_to_ground_truth.__doc__ = GroundTruthComparison.__doc__
