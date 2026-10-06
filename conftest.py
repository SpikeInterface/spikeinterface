import platform
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def create_cache_folder(tmp_path_factory):
    cache_folder = tmp_path_factory.mktemp("cache_folder")
    return cache_folder


@pytest.fixture(scope="module")
def debug_plots(request):
    """Return True if debug plots should be shown."""
    return request.config.getoption("--debug-plots")


def pytest_addoption(parser):
    parser.addoption(
        "--debug-plots",
        action="store_true",
        default=False,
        help="Enable debug plots during tests",
    )
    # Users on Linux get fork by default but the tests run with forkserver (the default since Python 3.14)
    parser.addoption(
        "--mp-context",
        default="forkserver" if platform.system() == "Linux" else None,
        help="Multiprocessing context used by the tests instead of the default one (forkserver on Linux)",
    )


def pytest_configure(config):
    mp_context = config.getoption("--mp-context")
    if mp_context is not None:
        import multiprocessing
        import spikeinterface.core.globals as si_globals

        # The default is patched as well so it survives reset_global_job_kwargs()
        multiprocessing.set_start_method(mp_context, force=True)
        si_globals._default_job_kwargs["mp_context"] = mp_context
        si_globals.global_job_kwargs["mp_context"] = mp_context


def pytest_collection_modifyitems(config, items):
    """
    This function marks (in the pytest sense) the tests according to their name and file_path location
    Marking them in turn allows the tests to be run by using the pytest -m marker_name option.
    """

    rootdir = Path(config.rootdir)
    modules_location = rootdir / "src" / "spikeinterface"
    for item in items:
        if config.getoption("--mp-context") is not None and item.name == "test_global_job_kwargs":
            item.add_marker(pytest.mark.skip(reason="--mp-context changes the default job kwargs"))

        try:
            rel_path = Path(item.fspath).relative_to(modules_location)
        except:
            continue

        module = rel_path.parts[0]
        if module == "sorters":
            if "internal" in rel_path.parts:
                item.add_marker("sorters_internal")
            elif "external" in rel_path.parts:
                item.add_marker("sorters_external")
            else:
                item.add_marker("sorters")
        else:
            item.add_marker(module)
