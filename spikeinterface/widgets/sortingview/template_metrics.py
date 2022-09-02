from .metrics import MetricsPlotter
from ..template_metrics import TemplateMetricsWidget


class TemplateMetricsPlotter(MetricsPlotter):
    pass


TemplateMetricsPlotter.register(TemplateMetricsWidget)
