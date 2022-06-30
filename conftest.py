import pytest
import shutil
from pathlib import Path


# define marks
mark_names = ["core", "extractors", "preprocessing", "postprocessing",
              "sorters", "qualitymetrics", "comparison", "widgets", 
              "exporters", "sortingcomponents"]


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
    # python 3.4/3.5 compat: rootdir = pathlib.Path(str(config.rootdir))
    rootdir = Path(config.rootdir)

    for item in items:
        rel_path = Path(item.fspath).relative_to(rootdir)

        for mark_name in mark_names:
            if f"/{mark_name}/" in str(rel_path):
                mark = getattr(pytest.mark, mark_name)
                item.add_marker(mark)


def pytest_sessionfinish(session, exitstatus):
    # teardown_stuff only if tests passed
    if exitstatus == 0:
        if pytest.global_test_folder.is_dir():
            shutil.rmtree(pytest.global_test_folder)
