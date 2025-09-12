from __future__ import annotations

from .dummy import DummyClustering
from .position import PositionClustering

from .random_projections import RandomProjectionClustering
from .iterative_hdbscan import CircusClustering
from .iterative_isosplit import TdcClustering
from .graph_clustering import GraphClustering

clustering_methods = {
    "dummy": DummyClustering,
    "position": PositionClustering,
    "random_projections": RandomProjectionClustering,
    "circus-clustering": CircusClustering,
    "tdc-clustering": TdcClustering,
    "graph-clustering": GraphClustering,
}


try:
    # Kilosort licence (GPL 3) is forcing us to make and use an external package
    from spikeinterface_kilosort_components.kilosort_clustering import KiloSortClustering

    clustering_methods["kilosort-clustering"] = KiloSortClustering
except ImportError:
    pass
