import pytest

import numpy as np

from spikeinterface.core import generate_recording
from spikeinterface.preprocessing import silence_artifacts


def test_silence_artifacts():
    # one segment only
    rec = generate_recording(durations=[10.0, 10])
    new_rec = silence_artifacts(rec, detect_threshold=5, freq_max=5.0, min_duration_ms=50)


if __name__ == "__main__":
    test_silence_artifacts()
