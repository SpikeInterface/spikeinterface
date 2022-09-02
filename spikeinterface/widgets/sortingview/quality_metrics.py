from .metrics import MetricsPlotter
from ..quality_metrics import QualityMetricsWidget


class QualityMetricsPlotter(MetricsPlotter):
    pass


QualityMetricsPlotter.register(QualityMetricsWidget)
