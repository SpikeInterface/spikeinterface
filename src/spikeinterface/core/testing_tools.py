from __future__ import annotations

import warnings

warnings.warn(
    "The 'testing_tools' submodule is deprecated. " "Use spikeinterface.core.generate instead",
    DeprecationWarning,
    stacklevel=2,
)

from .generate import *
