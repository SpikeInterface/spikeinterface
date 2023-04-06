from .metrics import MetricsPlotter
from ..template_metrics import TemplateMetricsWidget


class TemplateMetricsPlotter(MetricsPlotter):
    default_label = "SpikeInterface - Template Metrics"

    pass


TemplateMetricsPlotter.register(TemplateMetricsWidget)
