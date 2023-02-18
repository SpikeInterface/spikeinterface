import os
import shutil
from pathlib import Path

import pytest

ON_GITHUB = bool(os.getenv("GITHUB_ACTIONS"))


# define marks
mark_names = [
    "core",
    "extractors",
    "preprocessing",
    "postprocessing",
    "sorters",
    "si_sorters",
    "qualitymetrics",
    "comparison",
    "curation",
    "widgets",
    "exporters",
    "sortingcomponents",
]


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

    # python 3.4/3.5 compat: rootdir = pathlib.Path(str(config.rootdir))
    rootdir = Path(config.rootdir)

    for item in items:
        rel_path = Path(item.fspath).relative_to(rootdir)

        if "si_based_sorters" not in rel_path.name:
            for mark_name in mark_names:
                if f"/{mark_name}/" in str(rel_path):
                    mark = getattr(pytest.mark, mark_name)
                    item.add_marker(mark)
        else:
            item.add_marker("si_sorters")


def pytest_sessionfinish(session, exitstatus):
    # teardown_stuff only if tests passed
    # We don't delete the test folder in the CI because it was causing problems with the code coverage.
    if exitstatus == 0:
        if pytest.global_test_folder.is_dir() and not ON_GITHUB:
            shutil.rmtree(pytest.global_test_folder)
