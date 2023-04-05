from .metrics import MetricsPlotter
from ..quality_metrics import QualityMetricsWidget


class QualityMetricsPlotter(MetricsPlotter):
    default_label = "SpikeInterface - Quality Metrics"

    pass


QualityMetricsPlotter.register(QualityMetricsWidget)
