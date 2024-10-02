from __future__ import annotations

from .naive import NaiveMatching
from .tdc import TridesclousPeeler, TridesclousPeeler2
from .circus import CircusPeeler, CircusOMPSVDPeeler
from .wobble import WobbleMatch

matching_methods = {
    "naive": NaiveMatching,
    "tdc-peeler": TridesclousPeeler,
    "tdc-peeler2": TridesclousPeeler2,
    "circus": CircusPeeler,
    "circus-omp-svd": CircusOMPSVDPeeler,
    "wobble": WobbleMatch,
}
