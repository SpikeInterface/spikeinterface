"""
Module to benchmark:
  * sorters
  * some sorting components (clustering, motion, template matching)
"""

from .benchmark_sorter import SorterStudy
from .benchmark_sorter_without_gt import SorterStudyWithoutGroundTruth