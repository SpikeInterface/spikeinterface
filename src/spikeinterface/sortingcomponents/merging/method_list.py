from __future__ import annotations
from .circus import CircusMerging
from .lussac import LussacMerging
from .knn import KNNMerging

merging_methods = {"circus": CircusMerging, "lussac": LussacMerging, "knn": KNNMerging}
