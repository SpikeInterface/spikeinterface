import pytest

from spikeinterface.postprocessing.tests.conftest import _small_sorting_analyzer


@pytest.fixture(scope="module")
def small_sorting_analyzer():
    return _small_sorting_analyzer()
