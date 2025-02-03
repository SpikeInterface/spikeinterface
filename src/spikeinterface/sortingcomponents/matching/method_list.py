from __future__ import annotations

from .naive import NaiveMatching
from .tdc import TridesclousPeeler
from .circus import CircusPeeler, CircusOMPSVDPeeler
from .wobble import WobbleMatch

matching_methods = {
    "naive": NaiveMatching,
    "tdc-peeler": TridesclousPeeler,
    "circus": CircusPeeler,
    "circus-omp-svd": CircusOMPSVDPeeler,
    "wobble": WobbleMatch,
}

try:
    # kilosort licence (GPL 3) is forcing use to make an external package
    from spikeinterface_kilosort_components import KiloSortMatching

    matching_methods["kilosort-matching"] = KiloSortMatching
except ImportError:
    pass
