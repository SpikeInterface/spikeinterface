from __future__ import annotations

import numpy as np

from spikeinterface.core import BaseSorting
from spikeinterface.core.core_tools import define_function_from_class
from .basecomparison import BasePairComparison, MixinSpikeTrainComparison, MixinTemplateComparison
from .comparisontools import (
    do_count_event,
    make_match_count_matrix,
    make_agreement_scores_from_count,
    calculate_agreement_scores_with_distance,
    do_score_labels,
    do_confusion_matrix,
    do_count_score,
    compute_performance,
)
from spikeinterface.postprocessing import compute_template_similarity_by_pair


class BasePairSorterComparison(BasePairComparison, MixinSpikeTrainComparison):
    """
    Base class shared by SymmetricSortingComparison and GroundTruthComparison
    """

    def __init__(
        self,
        sorting1: BaseSorting,
        sorting2: BaseSorting,
        sorting1_name: str | None = None,
        sorting2_name: str | None = None,
        delta_time: float = 0.4,
        match_score: float = 0.5,
        chance_score: float = 0.1,
        ensure_symmetry: bool = False,
        agreement_method: str = "count",
        verbose: bool = False,
    ):
        if sorting1_name is None:
            sorting1_name = "sorting1"
        if sorting2_name is None:
            sorting2_name = "sorting2"
        assert sorting1.get_num_segments() == sorting2.get_num_segments(), (
            "The two sortings must have the same " "number of segments! "
        )

        BasePairComparison.__init__(
            self,
            object1=sorting1,
            object2=sorting2,
            name1=sorting1_name,
            name2=sorting2_name,
            match_score=match_score,
            chance_score=chance_score,
            verbose=verbose,
        )
        MixinSpikeTrainComparison.__init__(self, delta_time=delta_time)
        self.set_frames_and_frequency(self.object_list)

        self.unit1_ids = self.sorting1.get_unit_ids()
        self.unit2_ids = self.sorting2.get_unit_ids()

        self.ensure_symmetry = ensure_symmetry
        self.agreement_method = agreement_method

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
            print("Agreement scores...")

        # common to GroundTruthComparison and SymmetricSortingComparison
        # spike count for each spike train
        if self.agreement_method == "count":
            self.event_counts1 = do_count_event(self.sorting1)
            self.event_counts2 = do_count_event(self.sorting2)

            # matrix of  event match count for each pair
            self.match_event_count = make_match_count_matrix(
                self.sorting1, self.sorting2, self.delta_frames, ensure_symmetry=self.ensure_symmetry
            )

            # agreement matrix score for each pair
            self.agreement_scores = make_agreement_scores_from_count(
                self.match_event_count, self.event_counts1, self.event_counts2
            )
        elif self.agreement_method == "distance":

            self.agreement_scores = calculate_agreement_scores_with_distance(
                self.sorting1,
                self.sorting2,
                self.delta_frames,
            )

        else:
            raise ValueError("agreement_method must be 'from_count' or 'distance_matrix'")


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
    sorting1 : BaseSorting
        The first sorting for the comparison
    sorting2 : BaseSorting
        The second sorting for the comparison
    sorting1_name : str, default: None
        The name of sorter 1
    sorting2_name : : str, default: None
        The name of sorter 2
    delta_time : float, default: 0.4
        Number of ms to consider coincident spikes
    match_score : float, default: 0.5
        Minimum agreement score to match units
    chance_score : float, default: 0.1
        Minimum agreement score to for a possible match
    agreement_method : "count" | "distance", default: "count"
        The method to compute agreement scores. The "count" method computes agreement scores from spike counts.
        The "distance" method computes agreement scores from spike time distance functions.
    verbose : bool, default: False
        If True, output is verbose

    Returns
    -------
    sorting_comparison : SortingComparison
        The SortingComparison object
    """

    def __init__(
        self,
        sorting1: BaseSorting,
        sorting2: BaseSorting,
        sorting1_name: str | None = None,
        sorting2_name: str | None = None,
        delta_time: float = 0.4,
        match_score: float = 0.5,
        chance_score: float = 0.1,
        agreement_method: str = "count",
        verbose: bool = False,
    ):
        BasePairSorterComparison.__init__(
            self,
            sorting1,
            sorting2,
            sorting1_name=sorting1_name,
            sorting2_name=sorting2_name,
            delta_time=delta_time,
            match_score=match_score,
            chance_score=chance_score,
            ensure_symmetry=True,
            agreement_method=agreement_method,
            verbose=verbose,
        )

    def get_matching(self):
        return self.hungarian_match_12, self.hungarian_match_21

    def get_matching_event_count(self, unit1, unit2):
        if self.agreement_method == "count":
            if (unit1 is not None) and (unit2 is not None):
                return self.match_event_count.at[unit1, unit2]
            else:
                raise Exception("get_matching_event_count: unit1 and unit2 must not be None.")
        else:
            raise Exception("get_matching_event_count is valid only if agreement_method='from_count'")

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
    gt_sorting : BaseSorting
        The first sorting for the comparison
    tested_sorting : BaseSorting
        The second sorting for the comparison
    gt_name : str, default: None
        The name of sorter 1
    tested_name : : str, default: None
        The name of sorter 2
    delta_time : float, default: 0.4
        Number of ms to consider coincident spikes.
        This means that two spikes are considered simultaneous if they are within `delta_time` of each other or
        mathematically abs(spike1_time - spike2_time) <= delta_time.
    match_score : float, default: 0.5
        Minimum agreement score to match units
    chance_score : float, default: 0.1
        Minimum agreement score to for a possible match
    redundant_score : float, default: 0.2
        Agreement score above which units are redundant
    overmerged_score : float, default: 0.2
        Agreement score above which units can be overmerged
    well_detected_score : float, default: 0.8
        Agreement score above which units are well detected
    exhaustive_gt : bool, default: False
        Tell if the ground true is "exhaustive" or not. In other world if the
        GT have all possible units. It allows more performance measurement.
        For instance, MEArec simulated dataset have exhaustive_gt=True
    match_mode : "hungarian" | "best", default: "hungarian"
        The method to match units
    agreement_method : "count" | "distance", default: "count"
        The method to compute agreement scores. The "count" method computes agreement scores from spike counts.
        The "distance" method computes agreement scores from spike time distance functions.
    compute_labels : bool, default: False
        If True, labels are computed at instantiation
    compute_misclassifications : bool, default: False
        If True, misclassifications are computed at instantiation
    verbose : bool, default: False
        If True, output is verbose

    Returns
    -------
    sorting_comparison : SortingComparison
        The SortingComparison object
    """

    def __init__(
        self,
        gt_sorting: BaseSorting,
        tested_sorting: BaseSorting,
        gt_name: str | None = None,
        tested_name: str | None = None,
        delta_time: float = 0.4,
        match_score: float = 0.5,
        well_detected_score: float = 0.8,
        redundant_score: float = 0.2,
        overmerged_score: float = 0.2,
        chance_score: float = 0.1,
        exhaustive_gt: bool = False,
        agreement_method: str = "count",
        match_mode: str = "hungarian",
        compute_labels: bool = False,
        compute_misclassifications: bool = False,
        verbose: bool = False,
    ):
        import pandas as pd

        if gt_name is None:
            gt_name = "ground truth"
        if tested_name is None:
            tested_name = "tested"
        BasePairSorterComparison.__init__(
            self,
            gt_sorting,
            tested_sorting,
            sorting1_name=gt_name,
            sorting2_name=tested_name,
            delta_time=delta_time,
            match_score=match_score,
            chance_score=chance_score,
            ensure_symmetry=False,
            agreement_method=agreement_method,
            verbose=verbose,
        )
        self.exhaustive_gt = exhaustive_gt

        self._compute_misclassifications = compute_misclassifications
        self.redundant_score = redundant_score
        self.overmerged_score = overmerged_score
        self.well_detected_score = well_detected_score

        assert match_mode in ["hungarian", "best"]
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
        if self.match_mode == "hungarian":
            match_12 = self.hungarian_match_12
        elif self.match_mode == "best":
            match_12 = self.best_match_12

        self.count_score = do_count_score(self.event_counts1, self.event_counts2, match_12, self.match_event_count)

    def _do_confusion_matrix(self):
        if self._verbose:
            print("Computing confusion matrix...")

        if self.match_mode == "hungarian":
            match_12 = self.hungarian_match_12
        elif self.match_mode == "best":
            match_12 = self.best_match_12

        self._confusion_matrix = do_confusion_matrix(
            self.event_counts1, self.event_counts2, match_12, self.match_event_count
        )

    def get_confusion_matrix(self):
        """
        Computes the confusion matrix.

        Returns
        -------
        confusion_matrix : pandas.DataFrame
            The confusion matrix
        """
        if self._confusion_matrix is None:
            self._do_confusion_matrix()
        return self._confusion_matrix

    def _do_score_labels(self):
        assert self.match_mode == "hungarian", "Labels (TP, FP, FN) can be computed only with hungarian match"

        if self._verbose:
            print("Adding labels...")

        self._labels_st1, self._labels_st2 = do_score_labels(
            self.sorting1, self.sorting2, self.delta_frames, self.hungarian_match_12, self._compute_misclassifications
        )

    def get_performance(self, method="by_unit", output="pandas"):
        """
        Get performance rate with several method:
          * "raw_count" : just render the raw count table
          * "by_unit" : render perf as rate unit by unit of the GT
          * "pooled_with_average" : compute rate unit by unit and average

        Parameters
        ----------
        method : "by_unit" | "pooled_with_average", default: "by_unit"
            The method to compute performance
        output : "pandas" | "dict", default: "pandas"
            The output format

        Returns
        -------
        perf : pandas dataframe/series (or dict)
            dataframe/series (based on "output") with performance entries
        """
        import pandas as pd

        possibles = ("raw_count", "by_unit", "pooled_with_average")
        if method not in possibles:
            raise Exception("'method' can be " + " or ".join(possibles))

        if method == "raw_count":
            perf = self.count_score

        elif method == "by_unit":
            perf = compute_performance(self.count_score)

        elif method == "pooled_with_average":
            perf = self.get_performance(method="by_unit").mean(axis=0)

        if output == "dict" and isinstance(perf, (pd.DataFrame, pd.Series)):
            perf = perf.to_dict()

        return perf

    def print_performance(self, method="pooled_with_average"):
        """
        Print performance with the selected method

        Parameters
        ----------
        method : "by_unit" | "pooled_with_average", default: "pooled_with_average"
            The method to compute performance
        """

        template_txt_performance = _template_txt_performance

        if method == "by_unit":
            perf = self.get_performance(method=method, output="pandas")
            perf = perf * 100
            d = {k: perf[k].tolist() for k in perf.columns}
            txt = template_txt_performance.format(method=method, **d)
            print(txt)

        elif method == "pooled_with_average":
            perf = self.get_performance(method=method, output="pandas")
            perf = perf * 100
            txt = template_txt_performance.format(method=method, **perf.to_dict())
            print(txt)

    def print_summary(self, well_detected_score=None, redundant_score=None, overmerged_score=None):
        """
        Print a global performance summary that depend on the context:
          * exhaustive= True/False
          * how many gt units (one or several)

        This summary mix several performance metrics.

        Parameters
        ----------
        well_detected_score : float, default: None
            The agreement score above which tested units
            are counted as "well detected".
        redundant_score : float, default: None
            The agreement score below which tested units
            are counted as "false positive"" (and not "redundant").
        overmerged_score : float, default: None
            Tested units with 2 or more agreement scores above "overmerged_score"
            are counted as "overmerged".

        """
        txt = _template_summary_part1

        d = dict(
            num_gt=len(self.unit1_ids),
            num_tested=len(self.unit2_ids),
            num_well_detected=self.count_well_detected_units(well_detected_score),
        )

        if self.exhaustive_gt:
            txt = txt + _template_summary_part2
            d["num_redundant"] = self.count_redundant_units(redundant_score)
            d["num_overmerged"] = self.count_overmerged_units(overmerged_score)
            d["num_false_positive_units"] = self.count_false_positive_units()
            d["num_bad"] = self.count_bad_units()

        txt = txt.format(**d)

        print(txt)

    def get_well_detected_units(self, well_detected_score=None):
        """
        Return units list of "well detected units" from tested_sorting.

        "well detected units" are defined as units in tested that
        are well matched to GT units.

        Parameters
        ----------
        well_detected_score : float, default: None
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

        Parameters
        ----------
        well_detected_score : float, default: None
            The agreement score above which tested units
            are counted as "well detected".
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
        redundant_score : float, default: None
            The agreement score below which tested units
            are counted as "false positive"" (and not "redundant").
        """
        assert self.exhaustive_gt, "false_positive_units list is valid only if exhaustive_gt=True"

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

        Parameters
        ----------
        redundant_score : float | None, default: None
            The agreement score below which tested units
            are counted as "false positive"" (and not "redundant").
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
        redundant_score : float, default: None
            The agreement score above which tested units
            are counted as "redundant" (and not "false positive" ).
        """
        assert self.exhaustive_gt, "redundant_units list is valid only if exhaustive_gt=True"

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

         Parameters
         ----------
         redundant_score : float, default: None
             The agreement score below which tested units
             are counted as "false positive"" (and not "redundant").
        """
        return len(self.get_redundant_units(redundant_score=redundant_score))

    def get_overmerged_units(self, overmerged_score=None):
        """
        Return "overmerged units"

        "overmerged units" are defined as units in tested
        that match more than one GT unit with an agreement score larger than overmerged_score.

        Parameters
        ----------
        overmerged_score : float, default: None
            Tested units with 2 or more agreement scores above "overmerged_score"
            are counted as "overmerged".
        """
        assert self.exhaustive_gt, "overmerged_units list is valid only if exhaustive_gt=True"

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

        Parameters
        ----------
        overmerged_score : float, default: None
            Tested units with 2 or more agreement scores above "overmerged_score"
            are counted as "overmerged".
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
        assert self.exhaustive_gt, "bad_units list is valid only if exhaustive_gt=True"
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

    def count_units_categories(
        self,
        well_detected_score=None,
        overmerged_score=None,
        redundant_score=None,
    ):
        import pandas as pd

        count = pd.Series(dtype="int64")

        count["num_gt"] = len(self.sorting1.get_unit_ids())
        count["num_sorter"] = len(self.sorting2.get_unit_ids())
        count["num_well_detected"] = self.count_well_detected_units(well_detected_score)
        if self.exhaustive_gt:
            count["num_overmerged"] = self.count_overmerged_units(overmerged_score)
            count["num_redundant"] = self.count_redundant_units(redundant_score)
            count["num_false_positive"] = self.count_false_positive_units(redundant_score)
            count["num_bad"] = self.count_bad_units()

        return count


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
"""

_template_summary_part2 = """num_redundant: {num_redundant}
num_overmerged: {num_overmerged}
num_false_positive_units {num_false_positive_units}
num_bad: {num_bad}
"""


compare_sorter_to_ground_truth = define_function_from_class(
    source_class=GroundTruthComparison, name="compare_sorter_to_ground_truth"
)


class TemplateComparison(BasePairComparison, MixinTemplateComparison):
    """
    Compares units from different sessions based on template similarity

    Parameters
    ----------
    sorting_analyzer_1 : SortingAnalyzer
        The first SortingAnalyzer to get templates to compare.
    sorting_analyzer_2 : SortingAnalyzer
        The second SortingAnalyzer to get templates to compare.
    unit_ids1 : list, default: None
        List of units from sorting_analyzer_1 to compare.
    unit_ids2 : list, default: None
        List of units from sorting_analyzer_2 to compare.
    name1 : str, default: "sess1"
        Name of first session.
    name2 : str, default: "sess2"
        Name of second session.
    similarity_method : "cosine" | "l1" | "l2", default: "cosine"
        Method for the similarity matrix.
    support : "dense" | "union" | "intersection", default: "union"
        The support to compute the similarity matrix.
    num_shifts : int, default: 0
        Number of shifts to use to shift templates to maximize similarity.
    verbose : bool, default: False
        If True, output is verbose.
    chance_score : float, default: 0.3
         Minimum agreement score to for a possible match
    match_score : float, default: 0.7
        Minimum agreement score to match units


    Returns
    -------
    comparison : TemplateComparison
        The output TemplateComparison object.
    """

    def __init__(
        self,
        sorting_analyzer_1,
        sorting_analyzer_2,
        name1=None,
        name2=None,
        unit_ids1=None,
        unit_ids2=None,
        match_score=0.7,
        chance_score=0.3,
        similarity_method="cosine",
        support="union",
        num_shifts=0,
        verbose=False,
    ):
        if name1 is None:
            name1 = "sess1"
        if name2 is None:
            name2 = "sess2"
        BasePairComparison.__init__(
            self,
            object1=sorting_analyzer_1,
            object2=sorting_analyzer_2,
            name1=name1,
            name2=name2,
            match_score=match_score,
            chance_score=chance_score,
            verbose=verbose,
        )
        MixinTemplateComparison.__init__(
            self, similarity_method=similarity_method, support=support, num_shifts=num_shifts
        )

        self.sorting_analyzer_1 = sorting_analyzer_1
        self.sorting_analyzer_2 = sorting_analyzer_2
        channel_ids1 = sorting_analyzer_1.recording.get_channel_ids()
        channel_ids2 = sorting_analyzer_2.recording.get_channel_ids()

        # two options: all channels are shared or partial channels are shared
        if sorting_analyzer_1.recording.get_num_channels() != sorting_analyzer_2.recording.get_num_channels():
            raise ValueError("The two recordings must have the same number of channels")
        if np.any([ch1 != ch2 for (ch1, ch2) in zip(channel_ids1, channel_ids2)]):
            raise ValueError("The two recordings must have the same channel ids")

        self.matches = dict()

        if unit_ids1 is None:
            unit_ids1 = sorting_analyzer_1.sorting.get_unit_ids()

        if unit_ids2 is None:
            unit_ids2 = sorting_analyzer_2.sorting.get_unit_ids()
        self.unit_ids = [unit_ids1, unit_ids2]

        self._do_agreement()
        self._do_matching()

    def _do_agreement(self):
        if self._verbose:
            print("Agreement scores...")

        agreement_scores = compute_template_similarity_by_pair(
            self.sorting_analyzer_1,
            self.sorting_analyzer_2,
            method=self.similarity_method,
            support=self.support,
            num_shifts=self.num_shifts,
        )
        import pandas as pd

        self.agreement_scores = pd.DataFrame(agreement_scores, index=self.unit_ids[0], columns=self.unit_ids[1])


compare_templates = define_function_from_class(source_class=TemplateComparison, name="compare_templates")
