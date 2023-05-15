from ..template_metrics import TemplateMetricsWidget
from .metrics import MetricsPlotter


class TemplateMetricsPlotter(MetricsPlotter):
    pass        


TemplateMetricsPlotter.register(TemplateMetricsWidget)
