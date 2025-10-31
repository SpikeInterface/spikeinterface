import warnings

warnings.warn(
    "The module 'spikeinterface.qualitymetrics' is deprecated and will be removed in 0.105.0."
    "Please use 'spikeinterface.metrics.quality' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from spikeinterface.metrics.quality import *  # noqa: F403
