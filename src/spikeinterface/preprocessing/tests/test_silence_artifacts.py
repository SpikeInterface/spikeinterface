import pytest

import numpy as np

from spikeinterface.core import generate_recording
from spikeinterface.preprocessing import silence_artifacts


def test_silence_artifacts():
    # one segment only
    rec = generate_recording(durations=[10.0])

    rec_rmart_mean = silence_artifacts(rec)


if __name__ == "__main__":
    test_remove_artifacts()
