
from .groundtruthstudy import GroundTruthStudy
from .studytools import iter_computed_sorting
from .connectivitycomparison import ConnectivityGTComparison

class ConnectivityGtStudy(GroundTruthStudy):

    def run_comparisons(self, exhaustive_gt=True, window_ms=100.0, bin_ms=1.0, well_detected_score=0.8, **kwargs):
        self.comparisons = {}
        for rec_name, sorter_name, sorting in iter_computed_sorting(self.study_folder):
            gt_sorting = self.get_ground_truth(rec_name)
            comp = ConnectivityGTComparison(gt_sorting, sorting, exhaustive_gt=exhaustive_gt, window_ms=window_ms, bin_ms=bin_ms, well_detected_score=well_detected_score)
            self.comparisons[(rec_name, sorter_name)] = comp

        self.exhaustive_gt = exhaustive_gt
