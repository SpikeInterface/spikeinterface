import warnings

warnings.warn(
    "The module 'spikeinterface.postprocessing.template_metrics' is deprecated and will be removed in 0.105.0."
    "Please use 'spikeinterface.metrics.template' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from spikeinterface.metrics.template import *  # noqa: F403
