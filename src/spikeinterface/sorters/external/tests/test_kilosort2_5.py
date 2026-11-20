from spikeinterface.sorters import Kilosort2_5Sorter

import getpass

if getpass.getuser() == "samuel":
    kilosort2_5_path = "/home/samuel/Documents/SpikeInterface/Kilosort2.5"
    Kilosort2_5Sorter.set_kilosort2_5_path(kilosort2_5_path)

import unittest
import pytest

from spikeinterface.sorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
@pytest.mark.skipif(not Kilosort2_5Sorter.is_installed(), reason="kilosort2.5 not installed")
class Kilosort2_5CommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = Kilosort2_5Sorter


def test_default_batch_size(tmp_path, monkeypatch):
    monkeypatch.setattr(Kilosort2_5Sorter, "check_compiled", classmethod(lambda cls: False))
    monkeypatch.setattr(Kilosort2_5Sorter, "kilosort2_5_path", str(tmp_path))
    track_and_sort = tmp_path / "mainLoop" / "trackAndSort.m"
    track_and_sort.parent.mkdir()

    # batch loading from before the spike holes fix
    track_and_sort.write_text("batchstart = 0:NT:NT*nBatches;")
    with pytest.warns(UserWarning, match="spike holes"):
        assert Kilosort2_5Sorter._get_default_batch_size(ntbuff=64) == 64 * 1024 + 64

    # and after it (v2.5.2)
    track_and_sort.write_text("ntpad = rez.ops.ntbuff;\nbatchstart = ntpad:NT:NT*nBatches;")
    assert Kilosort2_5Sorter._get_default_batch_size(ntbuff=64) == 64 * 1024 - 64


if __name__ == "__main__":
    test = Kilosort2_5CommonTestSuite()
    test.setUp()
    test.test_with_run()
