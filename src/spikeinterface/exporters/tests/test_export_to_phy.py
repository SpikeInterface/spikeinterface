import shutil

import numpy as np
import pytest

from spikeinterface.core import generate_ground_truth_recording, create_sorting_analyzer
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


def _check_spikes_subset_files(output_folder, sorting_analyzer):
    spikes_file = output_folder / "_phy_spikes_subset.spikes.npy"
    channels_file = output_folder / "_phy_spikes_subset.channels.npy"
    waveforms_file = output_folder / "_phy_spikes_subset.waveforms.npy"
    for f in (spikes_file, channels_file, waveforms_file):
        assert f.is_file()

    subset_spikes = np.load(spikes_file)
    subset_channels = np.load(channels_file)
    subset_waveforms = np.load(waveforms_file)

    waveforms_ext = sorting_analyzer.get_extension("waveforms")
    n_subset = sorting_analyzer.get_extension("random_spikes").get_data().size

    assert subset_spikes.shape == (n_subset,)
    assert subset_channels.shape[0] == n_subset
    assert subset_waveforms.shape[0] == n_subset
    assert subset_waveforms.shape[1] == waveforms_ext.nbefore + waveforms_ext.nafter
    assert subset_spikes.dtype == np.int64
    assert subset_channels.dtype == np.int32

    # spike indices are sorted and point to valid rows of spike_times.npy
    spike_times = np.load(output_folder / "spike_times.npy")
    assert np.all(np.diff(subset_spikes) >= 0)
    assert subset_spikes.min() >= 0
    assert subset_spikes.max() < spike_times.shape[0]

    # channel indices are either -1 (padding) or a valid channel index
    num_chans = sorting_analyzer.get_num_channels()
    assert np.all((subset_channels == -1) | ((subset_channels >= 0) & (subset_channels < num_chans)))

    # if sparsity is saved, each spike's channels must match its cluster's sparse channel set
    # (dense exports don't write "template_ind.npy" since every unit uses all channels)
    template_ind_file = output_folder / "template_ind.npy"
    if template_ind_file.is_file():
        spike_clusters = np.load(output_folder / "spike_clusters.npy")[:, 0]
        template_ind = np.load(template_ind_file)
        expected_channels = template_ind[spike_clusters[subset_spikes]]
        np.testing.assert_array_equal(subset_channels, expected_channels)

    # waveforms on padded (-1) channels must be zero
    waveforms_by_channel = np.moveaxis(subset_waveforms, 1, 2)  # (n_subset, n_channels, n_samples)
    assert np.all(waveforms_by_channel[subset_channels == -1] == 0)


def test_export_to_phy_add_waveforms_sparse(sorting_analyzer_sparse_for_export, create_cache_folder):
    cache_folder = create_cache_folder
    output_folder = cache_folder / "phy_output_add_waveforms_sparse"
    if output_folder.is_dir():
        shutil.rmtree(output_folder)

    sorting_analyzer = sorting_analyzer_sparse_for_export
    assert sorting_analyzer.has_extension("waveforms")

    export_to_phy(
        sorting_analyzer,
        output_folder,
        compute_pc_features=False,
        compute_amplitudes=False,
        add_waveforms=True,
        n_jobs=1,
        chunk_size=10000,
        progress_bar=False,
    )

    _check_spikes_subset_files(output_folder, sorting_analyzer)


def test_export_to_phy_add_waveforms_dense(sorting_analyzer_dense_for_export, create_cache_folder):
    cache_folder = create_cache_folder
    output_folder = cache_folder / "phy_output_add_waveforms_dense"
    if output_folder.is_dir():
        shutil.rmtree(output_folder)

    sorting_analyzer = sorting_analyzer_dense_for_export
    assert sorting_analyzer.has_extension("waveforms")

    export_to_phy(
        sorting_analyzer,
        output_folder,
        compute_pc_features=False,
        compute_amplitudes=False,
        add_waveforms=True,
        n_jobs=1,
        chunk_size=10000,
        progress_bar=False,
    )

    _check_spikes_subset_files(output_folder, sorting_analyzer)


def test_export_to_phy_add_waveforms_missing_extension(create_cache_folder):
    cache_folder = create_cache_folder
    output_folder = cache_folder / "phy_output_add_waveforms_missing"
    if output_folder.is_dir():
        shutil.rmtree(output_folder)

    recording, sorting = generate_ground_truth_recording(
        durations=[10.0],
        sampling_frequency=28000.0,
        num_channels=8,
        num_units=4,
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_levels=5.0, strategy="on_the_fly"),
        seed=2205,
    )
    sorting_analyzer = create_sorting_analyzer(sorting=sorting, recording=recording, format="memory", sparse=True)
    # "random_spikes" is needed to compute "templates" without the "waveforms" extension
    sorting_analyzer.compute("random_spikes")
    sorting_analyzer.compute("templates")
    sorting_analyzer.compute("template_similarity")
    assert not sorting_analyzer.has_extension("waveforms")

    # add_waveforms requires "waveforms"/"random_spikes" to already be computed: since
    # "waveforms" is missing, it should warn and skip saving the spikes-subset files
    with pytest.warns(UserWarning, match="Cannot save the phy spikes-waveforms subset"):
        export_to_phy(
            sorting_analyzer,
            output_folder,
            compute_pc_features=False,
            compute_amplitudes=False,
            add_waveforms=True,
            n_jobs=1,
            chunk_size=10000,
            progress_bar=False,
        )

    assert not sorting_analyzer.has_extension("waveforms")
    assert not (output_folder / "_phy_spikes_subset.spikes.npy").is_file()
    assert not (output_folder / "_phy_spikes_subset.channels.npy").is_file()
    assert not (output_folder / "_phy_spikes_subset.waveforms.npy").is_file()


if __name__ == "__main__":
    sorting_analyzer_sparse = make_sorting_analyzer(sparse=True)
    sorting_analyzer_group = make_sorting_analyzer(sparse=False, with_group=True)
    sorting_analyzer_dense = make_sorting_analyzer(sparse=False)

    test_export_to_phy_dense(sorting_analyzer_dense)
    test_export_to_phy_sparse(sorting_analyzer_sparse)
    test_export_to_phy_by_property(sorting_analyzer_group)
