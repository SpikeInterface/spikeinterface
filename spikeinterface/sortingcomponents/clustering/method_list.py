from .dummy import DummyClustering
from .position import PositionClustering
from .sliding_hdbscan import SlidingHdbscanClustering
from .sliding_nn import SlidingNNClustering
#from .position_and_pca import PositionAndPCAClustering


clustering_methods = {
    'dummy' : DummyClustering,
    'position' : PositionClustering,
#    'position_and_pca' : PositionAndPCAClustering,
    'sliding_hdbscan' : SlidingHdbscanClustering,
    'sliding_nn' : SlidingNN
}