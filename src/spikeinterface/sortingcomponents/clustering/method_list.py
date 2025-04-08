from __future__ import annotations

from .dummy import DummyClustering
from .position import PositionClustering
from .sliding_hdbscan import SlidingHdbscanClustering

# from .sliding_nn import SlidingNNClustering
from .position_and_pca import PositionAndPCAClustering
from .position_ptp_scaled import PositionPTPScaledClustering
from .position_and_features import PositionAndFeaturesClustering
from .random_projections import RandomProjectionClustering
from .circus import CircusClustering
from .tdc import TdcClustering
from .graph_clustering import GraphClustering

clustering_methods = {
    "dummy": DummyClustering,
    "position": PositionClustering,
    "position_ptp_scaled": PositionPTPScaledClustering,
    "position_and_pca": PositionAndPCAClustering,
    "sliding_hdbscan": SlidingHdbscanClustering,
    # "sliding_nn": SlidingNNClustering,
    "position_and_features": PositionAndFeaturesClustering,
    "random_projections": RandomProjectionClustering,
    "circus": CircusClustering,
    "tdc_clustering": TdcClustering,
    "graph_clustering": GraphClustering,
}

try:
    # Kilosort licence (GPL 3) is forcing us to make and use an external package
    from spikeinterface_kilosort_components import KiloSortClustering

    clustering_methods["kilosort_clustering"] = KiloSortClustering
except ImportError:
    pass
