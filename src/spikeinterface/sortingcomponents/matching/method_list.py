from __future__ import annotations

from .naive import NaiveMatching
from .tdc import TridesclousPeeler
from .circus import CircusPeeler, CircusOMPPeeler, CircusOMPSVDPeeler
from .wobble import WobbleMatch

matching_methods = {
    "naive": NaiveMatching,
    "tridesclous": TridesclousPeeler,
    "circus": CircusPeeler,
    "circus-omp": CircusOMPPeeler,
    "circus-omp-svd": CircusOMPSVDPeeler,
    "wobble": WobbleMatch,
}
