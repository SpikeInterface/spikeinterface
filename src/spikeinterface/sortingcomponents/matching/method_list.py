from __future__ import annotations

from .naive import NaiveMatching
from .tdc import TridesclousPeeler
from .circus import CircusPeeler, CircusOMPSVDPeeler
from .wobble import WobbleMatch
from .tdc_drift import TridesclousDriftPeeler

matching_methods = {
    "naive": NaiveMatching,
    "tdc-peeler": TridesclousPeeler,
    "circus": CircusPeeler,
    "circus-omp-svd": CircusOMPSVDPeeler,
    "wobble": WobbleMatch,
    "tdc-drift-peeler": TridesclousDriftPeeler,
}
