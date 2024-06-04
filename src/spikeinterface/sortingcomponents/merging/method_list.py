from __future__ import annotations
from .circus import CircusMerging
from .lussac import LussacMerging

merging_methods = {"circus": CircusMerging, "lussac": LussacMerging}
