from __future__ import annotations

from .dummy import DummyClustering
from .position import PositionClustering
from .sliding_hdbscan import SlidingHdbscanClustering
from .sliding_nn import SlidingNNClustering
from .position_and_pca import PositionAndPCAClustering
from .position_ptp_scaled import PositionPTPScaledClustering
from .position_and_features import PositionAndFeaturesClustering
from .random_projections import RandomProjectionClustering
from .circus import CircusClustering

clustering_methods = {
    "dummy": DummyClustering,
    "position": PositionClustering,
    "position_ptp_scaled": PositionPTPScaledClustering,
    "position_and_pca": PositionAndPCAClustering,
    "sliding_hdbscan": SlidingHdbscanClustering,
    "sliding_nn": SlidingNNClustering,
    "position_and_features": PositionAndFeaturesClustering,
    "random_projections": RandomProjectionClustering,
    "circus": CircusClustering,
}
