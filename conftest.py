import pytest
import shutil
import os
from pathlib import Path


ON_GITHUB = bool(os.getenv('GITHUB_ACTIONS'))


# define marks
mark_names = ["core", "extractors", "preprocessing", "postprocessing",
              "sorters_external", "sorters_internal", "sorters",
              "qualitymetrics", "comparison", "curation",
              "widgets", "exporters", "sortingcomponents", "generation"]


# define global test folder
def pytest_sessionstart(session):
    # setup_stuff
    pytest.global_test_folder = Path(__file__).parent / "test_folder"
    if pytest.global_test_folder.is_dir():
        shutil.rmtree(pytest.global_test_folder)
    pytest.global_test_folder.mkdir()

    for mark_name in mark_names:
        (pytest.global_test_folder / mark_name).mkdir()

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



def pytest_sessionfinish(session, exitstatus):
    # teardown_stuff only if tests passed
    # We don't delete the test folder in the CI because it was causing problems with the code coverage.
    if exitstatus == 0:
        if pytest.global_test_folder.is_dir() and not ON_GITHUB:
            shutil.rmtree(pytest.global_test_folder)
