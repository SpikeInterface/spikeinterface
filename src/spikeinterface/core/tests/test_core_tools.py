import os
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


def test_path_utils_functions(create_cache_folder):
    cache_folder = create_cache_folder
    if platform.system() != "Windows":
        # make the paths on the system
        test_path1 = cache_folder / "yep/sub/path1"
        test_path2 = cache_folder / "yep/sub/path2"
        test_path_non_existing = "/yup/sub/path3"
        test_path1.parent.mkdir(parents=True, exist_ok=True)
        test_path2.parent.mkdir(parents=True, exist_ok=True)
        test_path1.touch()
        test_path2.touch()
        # posix path
        d = {
            "kwargs": {
                "path": test_path1,
                "recording": {
                    "module": "mock_module",
                    "class": "mock_class",
                    "version": "1.2",
                    "annotations": {},
                    "kwargs": {"path": test_path2, "non_existing_path": test_path_non_existing},
                },
            }
        }
        d2 = recursive_path_modifier(d, lambda p: str(p).replace("/yep", "/yop"))
        assert "/yop" in d2["kwargs"]["path"]
        assert "/yop" in d2["kwargs"]["recording"]["kwargs"]["path"]
        assert "/yop" not in d2["kwargs"]["recording"]["kwargs"]["non_existing_path"]

        d3 = make_paths_relative(d, cache_folder / "yep")
        assert d3["kwargs"]["path"] == "sub/path1"
        assert d3["kwargs"]["recording"]["kwargs"]["path"] == "sub/path2"

        abs_path1 = cache_folder / "yop/sub/path1"
        abs_path2 = cache_folder / "yop/sub/path2"
        abs_path1.parent.mkdir(parents=True, exist_ok=True)
        abs_path2.parent.mkdir(parents=True, exist_ok=True)
        abs_path1.touch()
        abs_path2.touch()
        d4 = make_paths_absolute(d3, cache_folder / "yop")
        assert "/yop" in d4["kwargs"]["path"]
        assert "yop" in d4["kwargs"]["recording"]["kwargs"]["path"]
        assert "yop" not in d4["kwargs"]["recording"]["kwargs"]["non_existing_path"]

    if platform.system() == "Windows":
        # test for windows Path
        test_path1 = cache_folder / "yep" / "sub" / "path1"
        test_path2 = cache_folder / "yep" / "sub" / "path2"
        test_path_non_existing = "yop/sub/path3"
        test_path1.parent.mkdir(parents=True, exist_ok=True)
        test_path2.parent.mkdir(parents=True, exist_ok=True)
        test_path1.touch()
        test_path2.touch()

        d = {
            "kwargs": {
                "path": rf"{test_path1}",
                "recording": {
                    "module": "mock_module",
                    "class": "mock_class",
                    "version": "1.2",
                    "annotations": {},
                    "kwargs": {"path": rf"{test_path2}", "non_existing_path": test_path_non_existing},
                },
            }
        }

        d2 = make_paths_relative(d, cache_folder / "yep")
        # the str be must unix like path even on windows for more portability
        assert d2["kwargs"]["path"] == "sub/path1"
        assert d2["kwargs"]["recording"]["kwargs"]["path"] == "sub/path2"
        # the non existing path is not modified
        assert d2["kwargs"]["recording"]["kwargs"]["non_existing_path"] == "yop/sub/path3"

        # same drive
        assert check_paths_relative(d, cache_folder / "yep", check_if_exists=True)
        # not the same drive
        assert not check_paths_relative(d, r"d:\yep", check_if_exists=True)

        d = {
            "kwargs": {
                "path": r"\\host\share\yep\sub\path1",
            }
        }
        # UNC cannot be relative to d: drive
        assert not check_paths_relative(d, r"d:\yep", check_if_exists=False)

        # UNC can be relative to the same UNC
        assert check_paths_relative(d, r"\\host\share", check_if_exists=False)


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
