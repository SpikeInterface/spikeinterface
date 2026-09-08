import warnings

warnings.warn(
    "The 'testing_tools' submodule is deprecated. " "Use spikeinterface.core.generate instead",
    FutureWarning,
    stacklevel=2,
)

from .generate import *
