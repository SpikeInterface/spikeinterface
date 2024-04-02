from __future__ import annotations

import warnings

warnings.warn(
    "The 'testing_tools' submodule is deprecated. " "Use spikeinterface.generation instead",
    DeprecationWarning,
    stacklevel=2,
)

from ..generation.generate import *
