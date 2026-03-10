import tempfile
import unittest

import pytest

from spikeinterface.extractors.tests.common_tests import SortingCommonTestSuite, local_folder
from spikeinterface.extractors.xclustextractors import XClustSortingExtractor


class XClustSortingTest(SortingCommonTestSuite, unittest.TestCase):
    ExtractorClass = XClustSortingExtractor
    downloads = ["xclust"]
    entities = [
        dict(folder_path=local_folder / "xclust" / "TT2", sampling_frequency=30_000.0),
        dict(folder_path=local_folder / "xclust" / "TT6", sampling_frequency=30_000.0),
        dict(
            file_path_list=sorted((local_folder / "xclust" / "TT6").glob("*.CEL")),
            sampling_frequency=30_000.0,
        ),
    ]


class TestXClustErrors(unittest.TestCase):
    def test_both_args_raises(self):
        with pytest.raises(ValueError, match="Provide either 'folder_path' or 'file_path_list', not both."):
            XClustSortingExtractor(
                folder_path="/some/path",
                file_path_list=["/some/file.CEL"],
                sampling_frequency=30_000.0,
            )

    def test_neither_arg_raises(self):
        with pytest.raises(ValueError, match="Provide one of 'folder_path' or 'file_path_list'."):
            XClustSortingExtractor(sampling_frequency=30_000.0)

    def test_empty_folder_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError, match=f"No .CEL files found in {tmp_dir}"):
                XClustSortingExtractor(folder_path=tmp_dir, sampling_frequency=30_000.0)
