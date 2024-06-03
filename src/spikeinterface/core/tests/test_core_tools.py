import platform
import math
from pathlib import Path
import pytest
import numpy as np

from spikeinterface.core.core_tools import (
    recursive_path_modifier,
    make_paths_relative,
    make_paths_absolute,
    check_paths_relative,
    normal_pdf,
    convert_string_to_bytes,
    add_suffix,
)


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def test_add_suffix():
    # first case - no dot provided before extension
    file_path = "testpath"
    possible_suffix = ["raw", "bin", "path"]
    file_path_with_suffix = add_suffix(file_path, possible_suffix)
    expected_path = "testpath.raw"
    assert str(file_path_with_suffix) == expected_path

    # second case - dot provided before extension
    file_path = "testpath"
    possible_suffix = [".raw", ".bin", ".path"]
    file_path_with_suffix = add_suffix(file_path, possible_suffix)
    expected_path = "testpath.raw"
    assert str(file_path_with_suffix) == expected_path


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

    def test_convert_string_to_bytes():
        # Test SI prefixes
        assert convert_string_to_bytes("1k") == 1000
        assert convert_string_to_bytes("1M") == 1000000
        assert convert_string_to_bytes("1G") == 1000000000
        assert convert_string_to_bytes("1T") == 1000000000000
        assert convert_string_to_bytes("1P") == 1000000000000000
        # Test IEC prefixes
        assert convert_string_to_bytes("1Ki") == 1024
        assert convert_string_to_bytes("1Mi") == 1048576
        assert convert_string_to_bytes("1Gi") == 1073741824
        assert convert_string_to_bytes("1Ti") == 1099511627776
        assert convert_string_to_bytes("1Pi") == 1125899906842624
        # Test mixed values
        assert convert_string_to_bytes("1.5k") == 1500
        assert convert_string_to_bytes("2.5M") == 2500000
        assert convert_string_to_bytes("0.5G") == 500000000
        assert convert_string_to_bytes("1.2T") == 1200000000000
        assert convert_string_to_bytes("1.5Pi") == 1688849860263936
        # Test zero values
        assert convert_string_to_bytes("0k") == 0
        assert convert_string_to_bytes("0Ki") == 0
        # Test invalid inputs (should raise assertion error)
        with pytest.raises(AssertionError) as e:
            convert_string_to_bytes("1Z")
            assert str(e.value) == "Unknown suffix: Z"

        with pytest.raises(AssertionError) as e:
            convert_string_to_bytes("1Xi")
            assert str(e.value) == "Unknown suffix: Xi"


def test_normal_pdf() -> None:
    mu = 4.160771
    sigma = 2.9334
    dx = 0.001

    xaxis = np.arange(-15, 25, dx)
    gauss = normal_pdf(xaxis, mu=mu, sigma=sigma)

    assert math.isclose(1.0, dx * np.sum(gauss))  # Checking that sum of pdf is 1
    assert math.isclose(mu, dx * np.sum(xaxis * gauss))  # Checking that mean is mu
    assert math.isclose(sigma**2, dx * np.sum(xaxis**2 * gauss) - mu**2)  # Checking that variance is sigma^2

    print(normal_pdf(-0.9355, mu=mu, sigma=sigma))
    assert math.isclose(normal_pdf(-0.9355, mu=mu, sigma=sigma), 0.03006929091)


if __name__ == "__main__":
    test_path_utils_functions()
