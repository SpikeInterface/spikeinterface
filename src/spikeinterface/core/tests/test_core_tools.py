import platform
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import importlib
import pytest
import numpy as np

from spikeinterface.core.core_tools import (
    recursive_path_modifier,
    make_paths_relative,
    make_paths_absolute,
    check_paths_relative,
)
from spikeinterface.core.binaryrecordingextractor import BinaryRecordingExtractor
from spikeinterface.core.generate import NoiseGeneratorRecording
from spikeinterface.core.numpyextractors import NumpySorting


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_path_utils_functions():
    if platform.system() != "Windows":
        # posix path
        d = {
            "kwargs": {
                "path": "/yep/sub/path1",
                "recording": {
                    "module": "mock_module",
                    "class": "mock_class",
                    "version": "1.2",
                    "annotations": {},
                    "kwargs": {"path": "/yep/sub/path2"},
                },
            }
        }

        d2 = recursive_path_modifier(d, lambda p: p.replace("/yep", "/yop"))
        assert d2["kwargs"]["path"].startswith("/yop")
        assert d2["kwargs"]["recording"]["kwargs"]["path"].startswith("/yop")

        d3 = make_paths_relative(d, Path("/yep"))
        assert d3["kwargs"]["path"] == "sub/path1"
        assert d3["kwargs"]["recording"]["kwargs"]["path"] == "sub/path2"

        d4 = make_paths_absolute(d3, "/yop")
        assert d4["kwargs"]["path"].startswith("/yop")
        assert d4["kwargs"]["recording"]["kwargs"]["path"].startswith("/yop")

    if platform.system() == "Windows":
        # test for windows Path
        d = {
            "kwargs": {
                "path": r"c:\yep\sub\path1",
                "recording": {
                    "module": "mock_module",
                    "class": "mock_class",
                    "version": "1.2",
                    "annotations": {},
                    "kwargs": {"path": r"c:\yep\sub\path2"},
                },
            }
        }

        d2 = make_paths_relative(d, "c:\\yep")
        # the str be must unix like path even on windows for more portability
        assert d2["kwargs"]["path"] == "sub/path1"
        assert d2["kwargs"]["recording"]["kwargs"]["path"] == "sub/path2"

        # same drive
        assert check_paths_relative(d, r"c:\yep")
        # not the same drive
        assert not check_paths_relative(d, r"d:\yep")

        d = {
            "kwargs": {
                "path": r"\\host\share\yep\sub\path1",
            }
        }
        # UNC cannot be relative to d: drive
        assert not check_paths_relative(d, r"d:\yep")

        # UNC can be relative to the same UNC
        assert check_paths_relative(d, r"\\host\share")


if __name__ == "__main__":
    test_path_utils_functions()
