from .naive import NaiveMatching
from .tdc import TridesclousPeeler
from .circus import CircusPeeler, CircusOMPPeeler


matching_methods = {
    'naive' : NaiveMatching,
    'tridesclous' : TridesclousPeeler,
    'circus' : CircusPeeler,
    'circus-omp' : CircusOMPPeeler
}
