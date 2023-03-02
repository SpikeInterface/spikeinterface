from .naive import NaiveMatching
from .tdc import TridesclousPeeler
from .circus import CircusPeeler, CircusOMPPeeler
from .spike_psvae import SpikePSVAE

matching_methods = {
    'naive' : NaiveMatching,
    'tridesclous' : TridesclousPeeler,
    'circus' : CircusPeeler,
    'circus-omp' : CircusOMPPeeler,
    'spike-psvae' : SpikePSVAE,
}
