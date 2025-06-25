"""
Module to benchmark:
  * sorters with or without ground-truth
  * some sorting components (clustering, motion, template matching)
"""

from .residual_analysis import analyse_residual, make_residual_recording
from .benchmark_sorter import SorterStudy
from .benchmark_sorter_without_gt import SorterStudyWithoutGroundTruth
