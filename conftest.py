import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def create_cache_folder(tmp_path_factory):
    cache_folder = tmp_path_factory.mktemp("cache_folder")
    return cache_folder

def pytest_collection_modifyitems(config, items):
    """
    This function marks (in the pytest sense) the tests according to their name and file_path location
    Marking them in turn allows the tests to be run by using the pytest -m marker_name option.
    """

    rootdir = Path(config.rootdir)
    modules_location = rootdir / "src" / "spikeinterface"
    for item in items:
        rel_path = Path(item.fspath).relative_to(modules_location)
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
