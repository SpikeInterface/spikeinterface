from __future__ import annotations

from .nearest import NearestTemplatesPeeler, NearestTemplatesSVDPeeler
from .tdc_peeler import TridesclousPeeler
from .circus import CircusOMPPeeler
from .wobble import WobbleMatch

matching_methods = {
    "nearest": NearestTemplatesPeeler,
    "nearest-svd": NearestTemplatesSVDPeeler,
    "tdc-peeler": TridesclousPeeler,
    "circus-omp": CircusOMPPeeler,
    "wobble": WobbleMatch,
}

try:
    # Kilosort licence (GPL 3) is forcing us to make and use an external package
    from spikeinterface_kilosort_components.kilosort_matching import KiloSortMatching

    matching_methods["kilosort-matching"] = KiloSortMatching
except ImportError:
    pass
