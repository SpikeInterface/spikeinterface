
from .groundtruthstudy import GroundTruthStudy
from .studytools import iter_computed_sorting
from .collisioncomparison import CollisionGTComparison

class CollisionGtStudy(GroundTruthStudy):

    def run_comparisons(self, exhaustive_gt=True, collision_lag=2.0, **kwargs):
        self.comparisons = {}
        for rec_name, sorter_name, sorting in iter_computed_sorting(self.study_folder):
            gt_sorting = self.get_ground_truth(rec_name)
            comp = CollisionGTComparison(gt_sorting, sorting, exhaustive_gt=exhaustive_gt, collision_lag=collision_lag)
            self.comparisons[(rec_name, sorter_name)] = comp
        self.exhaustive_gt = exhaustive_gt
        self.collision_lag =collision_lag
