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
