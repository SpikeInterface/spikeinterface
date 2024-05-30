from __future__ import annotations

from .circus import CircusMerging
from .lussac import LussacMerging
from .drift import DriftMerging

merging_methods = {"circus": CircusMerging, "lussac": LussacMerging, "drift": DriftMerging}
