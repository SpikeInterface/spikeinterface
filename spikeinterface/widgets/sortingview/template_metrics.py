from ..template_metrics import TemplateMetricsWidget
from .metrics import MetricsPlotter


class TemplateMetricsPlotter(MetricsPlotter):
    default_label = "SpikeInterface - Template Metrics"

    pass


TemplateMetricsPlotter.register(TemplateMetricsWidget)
