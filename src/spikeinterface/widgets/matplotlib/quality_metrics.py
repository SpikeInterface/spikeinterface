from ..quality_metrics import QualityMetricsWidget
from .metrics import MetricsPlotter


class QualityMetricsPlotter(MetricsPlotter):
    pass        


QualityMetricsPlotter.register(QualityMetricsWidget)
