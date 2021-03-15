from .timeseries import plot_timeseries, TimeseriesWidget
from .rasters import plot_rasters, RasterWidget
from .probemap import plot_probe_map, ProbeMapWidget

from .confusionmatrix import plot_confusion_matrix, ConfusionMatrixWidget
from .agreementmatrix import plot_agreement_matrix, AgreementMatrixWidget
from .multicompgraph import (
    plot_multicomp_graph, MultiCompGraphWidget,
    plot_multicomp_agreement, MultiCompGlobalAgreementWidget,
    plot_multicomp_agreement_by_sorter, MultiCompAgreementBySorterWidget)
