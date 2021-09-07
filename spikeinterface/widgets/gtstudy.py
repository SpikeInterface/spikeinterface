"""
Various widgets on top of GroundTruthStudy to summary results:
  * run times
  * performances
  * count units
"""
import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget



class StudyComparisonRunTimesWidget(BaseWidget):
    """
    Plot run times for a study.

    Parameters
    ----------
    gt_comparison: GroundTruthComparison
        The ground truth sorting comparison object
    figure: matplotlib figure
        The figure to be used. If not given a figure is created

    Returns
    -------
    W: ConfusionMatrixWidget
        The output widget
    """
    def __init__(self, study, exhaustive_gt=False, ax=None):
        
        self.study = study
        
        BaseWidget.__init__(self, ax=ax)
        
    def plot(self):
        pass





def plot_gt_study_run_times(*args, **kwargs):
    W = StudyComparisonRunTimesWidget(*args, **kwargs)
    W.plot()
    return W
plot_gt_study_run_times.__doc__ = StudyComparisonRunTimesWidget.__doc__

