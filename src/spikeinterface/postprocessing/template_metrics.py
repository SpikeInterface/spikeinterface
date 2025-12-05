import warnings


from spikeinterface.metrics.template import ComputeTemplateMetrics as ComputeTemplateMetricsNew
from spikeinterface.metrics.template import compute_template_metrics as compute_template_metrics_new


class ComputeTemplateMetrics(ComputeTemplateMetricsNew):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The module 'spikeinterface.postprocessing.template_metrics' is deprecated and will be removed in 0.105.0."
            "Please use 'spikeinterface.metrics.template' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def compute_template_metrics(*args, **kwargs):
    warnings.warn(
        "The module 'spikeinterface.postprocessing.template_metrics' is deprecated and will be removed in 0.105.0."
        "Please use 'spikeinterface.metrics.template' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_template_metrics_new(*args, **kwargs)
