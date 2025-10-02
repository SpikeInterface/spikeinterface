from __future__ import annotations

from .dummy import DummyClustering
from .positions import PositionsClustering

from .random_projections import RandomProjectionClustering
from .iterative_hdbscan import IterativeHDBSCANClustering
from .iterative_isosplit import IterativeISOSPLITClustering
from .graph_clustering import GraphClustering

clustering_methods = {
    "dummy": DummyClustering,
    "hdbscan-positions": PositionsClustering,
    "random-projections": RandomProjectionClustering,
    "iterative-hdbscan": IterativeHDBSCANClustering,
    "iterative-isosplit": IterativeISOSPLITClustering,
    "graph-clustering": GraphClustering,
}


try:
    # Kilosort licence (GPL 3) is forcing us to make and use an external package
    from spikeinterface_kilosort_components.kilosort_clustering import KiloSortClustering

    clustering_methods["kilosort-clustering"] = KiloSortClustering
except ImportError:
    pass
