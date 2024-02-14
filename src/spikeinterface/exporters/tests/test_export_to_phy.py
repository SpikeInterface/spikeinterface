import pytest
import shutil
from pathlib import Path

import numpy as np

from spikeinterface.postprocessing import compute_principal_components


from spikeinterface.core import compute_sparsity
from spikeinterface.exporters import export_to_phy

from spikeinterface.exporters.tests.common import (
    cache_folder,
    make_sorting_result,
    sorting_result_sparse_for_export,
    sorting_result_with_group_for_export,
    sorting_result_dense_for_export,
)


def test_export_to_phy_dense(sorting_result_dense_for_export):
    output_folder1 = cache_folder / "phy_output_dense"
    for f in (output_folder1,):
        if f.is_dir():
            shutil.rmtree(f)

    sorting_result = sorting_result_dense_for_export

    export_to_phy(
        sorting_result,
        output_folder1,
        compute_pc_features=True,
        compute_amplitudes=True,
        n_jobs=1,
        chunk_size=10000,
        progress_bar=True,
    )


def test_export_to_phy_sparse(sorting_result_sparse_for_export):
    output_folder1 = cache_folder / "phy_output_1"
    output_folder2 = cache_folder / "phy_output_2"
    for f in (output_folder1, output_folder2):
        if f.is_dir():
            shutil.rmtree(f)

    sorting_result = sorting_result_sparse_for_export

    export_to_phy(
        sorting_result,
        output_folder1,
        compute_pc_features=True,
        compute_amplitudes=True,
        n_jobs=1,
        chunk_size=10000,
        progress_bar=True,
    )

    # Test for previous crash when copy_binary=False.
    export_to_phy(
        sorting_result,
        output_folder2,
        compute_pc_features=False,
        compute_amplitudes=False,
        n_jobs=2,
        chunk_size=10000,
        progress_bar=False,
        copy_binary=False,
    )


def test_export_to_phy_by_property(sorting_result_with_group_for_export):
    output_folder = cache_folder / "phy_output_property"

    for f in (output_folder,):
        if f.is_dir():
            shutil.rmtree(f)

    sorting_result = sorting_result_with_group_for_export
    print(sorting_result.sparsity)

    export_to_phy(
        sorting_result,
        output_folder,
        compute_pc_features=True,
        compute_amplitudes=True,
        n_jobs=1,
        chunk_size=10000,
        progress_bar=True,
    )

    template_inds = np.load(output_folder / "template_ind.npy")
    assert template_inds.shape == (sorting_result.unit_ids.size, 4)


if __name__ == "__main__":
    sorting_result_sparse = make_sorting_result(sparse=True)
    sorting_result_group = make_sorting_result(sparse=False, with_group=True)
    sorting_result_dense = make_sorting_result(sparse=False)

    test_export_to_phy_dense(sorting_result_dense)
    test_export_to_phy_sparse(sorting_result_sparse)
    test_export_to_phy_by_property(sorting_result_group)
