from .naive import NaiveMatching
from .tdc import TridesclousPeeler
from .circus import CircusPeeler, CircusOMPPeeler
from .wobble import WobbleMatch

matching_methods = {
    "naive": NaiveMatching,
    "tridesclous": TridesclousPeeler,
    "circus": CircusPeeler,
    "circus-omp": CircusOMPPeeler,
    "wobble": WobbleMatch,
}
