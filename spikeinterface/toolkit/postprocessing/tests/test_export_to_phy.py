import unittest
import shutil
from pathlib import Path

import pytest



from spikeinterface import WaveformExtractor
from spikeinterface.extractors import toy_example
from spikeinterface.toolkit.postprocessing import export_to_phy


def test_export_to_phy():
    recording, sorting = toy_example(num_segments=2, num_units=10)


if __name__ == '__main__':
    test_export_to_phy()
