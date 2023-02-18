from .circus import CircusOMPPeeler, CircusPeeler
from .naive import NaiveMatching
from .tdc import TridesclousPeeler

matching_methods = {
    "naive": NaiveMatching,
    "tridesclous": TridesclousPeeler,
    "circus": CircusPeeler,
    "circus-omp": CircusOMPPeeler,
}
