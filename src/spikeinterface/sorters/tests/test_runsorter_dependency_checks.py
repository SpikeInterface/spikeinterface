import os
import pytest
from pathlib import Path
import shutil
import platform
from spikeinterface import generate_ground_truth_recording
from spikeinterface.sorters.utils import has_spython, has_docker_python, has_docker, has_singularity
from spikeinterface.sorters import run_sorter
import subprocess
import sys
import copy


def _monkeypatch_return_false():
    """
    A function to monkeypatch the `has_<dependency>` functions,
    ensuring the always return `False` at runtime.
    """
    return False


def _monkeypatch_return_true():
    """
    Monkeypatch for some `has_<dependency>` functions to
    return `True` so functions that are later in the
    `runsorter` code can be checked.
    """
    return True


class TestRunersorterDependencyChecks:
    """
    This class tests whether expected dependency checks prior to sorting are run.
    The run_sorter function should raise an error if:
        - singularity is not installed
        - spython is not installed (python package)
        - docker is not installed
        - docker is not installed (python package)
    when running singularity / docker respectively.

    Two separate checks should be run. First, that the
    relevant `has_<dependency>` function (indicating if the dependency
    is installed) is working. Unfortunately it is not possible to
    easily test this core singularity and docker installs, so this is not done.
    `uninstall_python_dependency()` allows a test to check if the
    `has_spython()` and `has_docker_dependency()` return `False` as expected
    when these python modules are not installed.

    Second, the `run_sorters()` function should return the appropriate error
    when these functions return that the dependency is not available. This is
    easier to test as these `has_<dependency>` reporting functions can be
    monkeypatched to return False at runtime. This is done for these 4
    dependency checks, and tests check the expected error is raised.

    Notes
    ----
    `has_nvidia()` and `has_docker_nvidia_installed()` are not tested
    as these are complex GPU-related dependencies which are difficult to mock.
    """

    @pytest.fixture(scope="function")
    def uninstall_python_dependency(self, request):
        """
        This python fixture mocks python modules not being importable
        by setting the relevant `sys.modules` dict entry to `None`.
        It uses `yield` so that the function can tear-down the test
        (even if it failed) and replace the patched `sys.module` entry.

        This function uses an `indirect` parameterization, meaning the
        `request.param` is passed to the fixture at the start of the
        test function. This is used to reuse code for nearly identical
        `spython` and `docker` python dependency tests.
        """
        dep_name = request.param
        assert dep_name in ["spython", "docker"]

        try:
            if dep_name == "spython":
                import spython
            else:
                import docker
            dependency_installed = True
        except:
            dependency_installed = False

        if dependency_installed:
            copy_import = sys.modules[dep_name]
            sys.modules[dep_name] = None
        yield
        if dependency_installed:
            sys.modules[dep_name] = copy_import

    @pytest.fixture(scope="session")
    def recording(self):
        """
        Make a small recording to have something to pass to the sorter.
        """
        recording, _ = generate_ground_truth_recording(durations=[10])
        return recording

    @pytest.mark.skipif(platform.system() != "Linux", reason="spython install only for Linux.")
    @pytest.mark.parametrize("uninstall_python_dependency", ["spython"], indirect=True)
    def test_has_spython(self, recording, uninstall_python_dependency):
        """
        Test the `has_spython()` function, see class docstring and
        `uninstall_python_dependency()` for details.
        """
        assert has_spython() is False

    @pytest.mark.parametrize("uninstall_python_dependency", ["docker"], indirect=True)
    def test_has_docker_python(self, recording, uninstall_python_dependency):
        """
        Test the `has_docker_python()` function, see class docstring and
        `uninstall_python_dependency()` for details.
        """
        assert has_docker_python() is False

    def test_no_singularity_error_raised(self, recording, monkeypatch):
        """
        When running a sorting, if singularity dependencies (singularity
        itself or the `spython` package`) are not installed, an error is raised.
        Beacause it is hard to actually uninstall these dependencies, the
        `has_<dependency>` functions that let `run_sorter` know if the dependency
        are installed are monkeypatched. This is done so at runtime these always
        return False. Then, test the expected error is raised when the dependency
        is not found.
        """
        monkeypatch.setattr(f"spikeinterface.sorters.runsorter.has_singularity", _monkeypatch_return_false)

        with pytest.raises(RuntimeError) as e:
            run_sorter("kilosort2_5", recording, singularity_image=True)

        assert "Singularity is not installed." in str(e)

    def test_no_spython_error_raised(self, recording, monkeypatch):
        """
        See `test_no_singularity_error_raised()`.
        """
        # make sure singularity test returns true as that comes first
        monkeypatch.setattr(f"spikeinterface.sorters.runsorter.has_singularity", _monkeypatch_return_true)
        monkeypatch.setattr(f"spikeinterface.sorters.runsorter.has_spython", _monkeypatch_return_false)

        with pytest.raises(RuntimeError) as e:
            run_sorter("kilosort2_5", recording, singularity_image=True)

        assert "The python `spython` package must be installed" in str(e)

    def test_no_docker_error_raised(self, recording, monkeypatch):
        """
        See `test_no_singularity_error_raised()`.
        """
        monkeypatch.setattr(f"spikeinterface.sorters.runsorter.has_docker", _monkeypatch_return_false)

        with pytest.raises(RuntimeError) as e:
            run_sorter("kilosort2_5", recording, docker_image=True)

        assert "Docker is not installed." in str(e)

    def test_as_no_docker_python_error_raised(self, recording, monkeypatch):
        """
        See `test_no_singularity_error_raised()`.
        """
        # make sure docker test returns true as that comes first
        monkeypatch.setattr(f"spikeinterface.sorters.runsorter.has_docker", _monkeypatch_return_true)
        monkeypatch.setattr(f"spikeinterface.sorters.runsorter.has_docker_python", _monkeypatch_return_false)

        with pytest.raises(RuntimeError) as e:
            run_sorter("kilosort2_5", recording, docker_image=True)

        assert "The python `docker` package must be installed" in str(e)
