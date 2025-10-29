# the method list is imported on the fly by find_clusters_from_peaks otherwise it is not
# possible to inject external methods (like ks) dynamically
# from .method_list import clustering_methods

from .main import find_clusters_from_peaks
