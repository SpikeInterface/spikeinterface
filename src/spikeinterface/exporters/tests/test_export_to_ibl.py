import shutil

import numpy as np

from spikeinterface.exporters import export_to_ibl

from spikeinterface.exporters.tests.common import (
    make_sorting_analyzer,
    sorting_analyzer_sparse_for_export,
)


def test_export_to_ibl(sorting_analyzer_sparse_for_export, create_cache_folder):
    cache_folder = create_cache_folder
    output_folder = cache_folder / "ibl_output"
    for f in (output_folder,):
        if f.is_dir():
            shutil.rmtree(f)

    export_to_ibl(
        sorting_analyzer_sparse_for_export,
        output_folder,
        lfp_recording=sorting_analyzer_sparse_for_export.recording,
    )


if __name__ == "__main__":
    sorting_analyzer = make_sorting_analyzer(sparse=True)
    test_export_to_ibl(sorting_analyzer)
