from pathlib import Path
import shutil


def _clean():
    # clean
    folders = [
        "./toy_example_recording",
        "./sorter_with_tridesclous2",
        "./analyzer_with_tridesclous2",
        "./sorter_with_kilosort4",
        "./analyzer_with_kilosort4",
    ]
    for folder in folders:
        if Path(folder).exists():
            shutil.rmtree(folder)


if __name__ == "__main__":

    _clean()
