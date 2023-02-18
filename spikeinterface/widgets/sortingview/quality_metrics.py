from ..quality_metrics import QualityMetricsWidget
from .metrics import MetricsPlotter


class QualityMetricsPlotter(MetricsPlotter):
    default_label = "SpikeInterface - Quality Metrics"

    pass


QualityMetricsPlotter.register(QualityMetricsWidget)
