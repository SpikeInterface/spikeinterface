import shutil

import numpy as np


from spikeinterface.exporters import export_to_phy

from spikeinterface.exporters.tests.common import (
    make_sorting_analyzer,
    sorting_analyzer_dense_for_export,
    sorting_analyzer_sparse_for_export,
    sorting_analyzer_with_group_for_export,
)


def test_export_to_phy_dense(sorting_analyzer_dense_for_export, create_cache_folder):
    cache_folder = create_cache_folder
    output_folder1 = cache_folder / "phy_output_dense"
    for f in (output_folder1,):
        if f.is_dir():
            shutil.rmtree(f)

    sorting_analyzer = sorting_analyzer_dense_for_export

    export_to_phy(
        sorting_analyzer,
        output_folder1,
        compute_pc_features=True,
        compute_amplitudes=True,
        n_jobs=1,
        chunk_size=10000,
        progress_bar=True,
    )


def test_export_to_phy_sparse(sorting_analyzer_sparse_for_export, create_cache_folder):
    cache_folder = create_cache_folder
    output_folder1 = cache_folder / "phy_output_1"
    output_folder2 = cache_folder / "phy_output_2"
    for f in (output_folder1, output_folder2):
        if f.is_dir():
            shutil.rmtree(f)

    sorting_analyzer = sorting_analyzer_sparse_for_export

    export_to_phy(
        sorting_analyzer,
        output_folder1,
        compute_pc_features=True,
        compute_amplitudes=True,
        n_jobs=1,
        chunk_size=10000,
        progress_bar=True,
    )

    # Test for previous crash when copy_binary=False.
    export_to_phy(
        sorting_analyzer,
        output_folder2,
        compute_pc_features=False,
        compute_amplitudes=False,
        n_jobs=2,
        chunk_size=10000,
        progress_bar=False,
        copy_binary=False,
    )


def test_export_to_phy_by_property(sorting_analyzer_with_group_for_export, create_cache_folder):
    cache_folder = create_cache_folder
    output_folder = cache_folder / "phy_output_property"

    for f in (output_folder,):
        if f.is_dir():
            shutil.rmtree(f)

    sorting_analyzer = sorting_analyzer_with_group_for_export
    print(sorting_analyzer.sparsity)

    export_to_phy(
        sorting_analyzer,
        output_folder,
        compute_pc_features=True,
        compute_amplitudes=True,
        n_jobs=1,
        chunk_size=10000,
        progress_bar=True,
    )

    template_inds = np.load(output_folder / "template_ind.npy")
    assert template_inds.shape == (sorting_analyzer.unit_ids.size, 4)


def test_export_to_phy_metrics(sorting_analyzer_sparse_for_export, create_cache_folder):
    cache_folder = create_cache_folder

    sorting_analyzer = sorting_analyzer_sparse_for_export

    # quality metrics are computed already
    qm = sorting_analyzer.get_extension("quality_metrics").get_data()
    output_folder = cache_folder / "phy_output_qm"
    export_to_phy(
        sorting_analyzer,
        output_folder,
        compute_pc_features=False,
        compute_amplitudes=False,
        n_jobs=1,
        chunk_size=10000,
        progress_bar=True,
        add_quality_metrics=True,
    )
    for col_name in qm.columns:
        assert (output_folder / f"cluster_{col_name}.tsv").is_file()

    # quality metrics are computed already
    tm_ext = sorting_analyzer.compute("template_metrics")
    tm = tm_ext.get_data()
    output_folder = cache_folder / "phy_output_tm_not_qm"
    export_to_phy(
        sorting_analyzer,
        output_folder,
        compute_pc_features=False,
        compute_amplitudes=False,
        n_jobs=1,
        chunk_size=10000,
        progress_bar=True,
        add_quality_metrics=False,
        add_template_metrics=True,
    )
    for col_name in tm.columns:
        assert (output_folder / f"cluster_{col_name}.tsv").is_file()
    for col_name in qm.columns:
        assert not (output_folder / f"cluster_{col_name}.tsv").is_file()

    # custom metrics
    sorting_analyzer.sorting.set_property("custom_metric", np.random.rand(sorting_analyzer.unit_ids.size))
    output_folder = cache_folder / "phy_output_custom"
    export_to_phy(
        sorting_analyzer,
        output_folder,
        compute_pc_features=False,
        compute_amplitudes=False,
        n_jobs=1,
        chunk_size=10000,
        progress_bar=True,
        add_quality_metrics=False,
        add_template_metrics=False,
        additional_properties=["custom_metric"],
    )
    assert (output_folder / "cluster_custom_metric.tsv").is_file()
    for col_name in tm.columns:
        assert not (output_folder / f"cluster_{col_name}.tsv").is_file()
    for col_name in qm.columns:
        assert not (output_folder / f"cluster_{col_name}.tsv").is_file()


if __name__ == "__main__":
    sorting_analyzer_sparse = make_sorting_analyzer(sparse=True)
    sorting_analyzer_group = make_sorting_analyzer(sparse=False, with_group=True)
    sorting_analyzer_dense = make_sorting_analyzer(sparse=False)

    test_export_to_phy_dense(sorting_analyzer_dense)
    test_export_to_phy_sparse(sorting_analyzer_sparse)
    test_export_to_phy_by_property(sorting_analyzer_group)
