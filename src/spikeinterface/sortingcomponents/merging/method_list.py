from __future__ import annotations
from .circus import CircusMerging
from .drift import DriftMerging


merging_methods = {"circus": CircusMerging, "drift": DriftMerging}


try:
    import lussac.utils as utils
    HAVE_LUSSAC = True
except Exception:
    HAVE_LUSSAC = False

if HAVE_LUSSAC:
    from .lussac import LussacMerging
    merging_methods = {"lussac": LussacMerging}


