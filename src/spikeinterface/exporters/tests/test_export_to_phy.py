import pytest
import shutil
from pathlib import Path

import numpy as np

from spikeinterface.postprocessing import compute_principal_components


from spikeinterface.core import compute_sparsity
from spikeinterface.exporters import export_to_phy

from spikeinterface.exporters.tests.common import (
    cache_folder,
    make_waveforms_extractor,
    waveforms_extractor_sparse_for_export,
    waveforms_extractor_dense_for_export,
    waveforms_extractor_with_group_for_export,
)


def test_export_to_phy(waveforms_extractor_sparse_for_export):
    output_folder1 = cache_folder / "phy_output_1"
    output_folder2 = cache_folder / "phy_output_2"
    for f in (output_folder1, output_folder2):
        if f.is_dir():
            shutil.rmtree(f)

    waveform_extractor = waveforms_extractor_sparse_for_export

    export_to_phy(
        waveform_extractor,
        output_folder1,
        compute_pc_features=True,
        compute_amplitudes=True,
        n_jobs=1,
        chunk_size=10000,
        progress_bar=True,
    )

    # Test for previous crash when copy_binary=False.
    export_to_phy(
        waveform_extractor,
        output_folder2,
        compute_pc_features=False,
        compute_amplitudes=False,
        n_jobs=2,
        chunk_size=10000,
        progress_bar=False,
        copy_binary=False,
    )


def test_export_to_phy_by_property(waveforms_extractor_with_group_for_export):
    output_folder = cache_folder / "phy_output"
    output_folder_rm = cache_folder / "phy_output_rm"

    for f in (output_folder, output_folder_rm):
        if f.is_dir():
            shutil.rmtree(f)

    waveform_extractor = waveforms_extractor_with_group_for_export

    sparsity_group = compute_sparsity(waveform_extractor, method="by_property", by_property="group")
    export_to_phy(
        waveform_extractor,
        output_folder,
        compute_pc_features=True,
        compute_amplitudes=True,
        sparsity=sparsity_group,
        n_jobs=1,
        chunk_size=10000,
        progress_bar=True,
    )

    template_inds = np.load(output_folder / "template_ind.npy")
    assert template_inds.shape == (waveform_extractor.unit_ids.size, 4)

    # Remove one channel
    # recording_rm = recording.channel_slice([0, 2, 3, 4, 5, 6, 7])
    # waveform_extractor_rm = extract_waveforms(recording_rm, sorting, waveform_folder_rm, sparse=False)
    # sparsity_group = compute_sparsity(waveform_extractor_rm, method="by_property", by_property="group")

    # export_to_phy(
    #     waveform_extractor_rm,
    #     output_folder_rm,
    #     compute_pc_features=True,
    #     compute_amplitudes=True,
    #     sparsity=sparsity_group,
    #     n_jobs=1,
    #     chunk_size=10000,
    #     progress_bar=True,
    # )

    # template_inds = np.load(output_folder_rm / "template_ind.npy")
    # assert template_inds.shape == (num_units, 4)
    # assert len(np.where(template_inds == -1)[0]) > 0


def test_export_to_phy_by_sparsity(waveforms_extractor_dense_for_export):
    output_folder_radius = cache_folder / "phy_output_radius"
    output_folder_multi_sparse = cache_folder / "phy_output_multi_sparse"
    for f in (output_folder_radius, output_folder_multi_sparse):
        if f.is_dir():
            shutil.rmtree(f)

    waveform_extractor = waveforms_extractor_dense_for_export

    sparsity_radius = compute_sparsity(waveform_extractor, method="radius", radius_um=50.0)
    export_to_phy(
        waveform_extractor,
        output_folder_radius,
        compute_pc_features=True,
        compute_amplitudes=True,
        sparsity=sparsity_radius,
        n_jobs=1,
        chunk_size=10000,
        progress_bar=True,
    )

    template_ind = np.load(output_folder_radius / "template_ind.npy")
    pc_ind = np.load(output_folder_radius / "pc_feature_ind.npy")
    # templates have different shapes!
    assert -1 in template_ind
    assert -1 in pc_ind

    # pre-compute PC with another sparsity
    sparsity_radius_small = compute_sparsity(waveform_extractor, method="radius", radius_um=30.0)
    pc = compute_principal_components(waveform_extractor, sparsity=sparsity_radius_small)
    export_to_phy(
        waveform_extractor,
        output_folder_multi_sparse,
        compute_pc_features=True,
        compute_amplitudes=True,
        sparsity=sparsity_radius,
        n_jobs=1,
        chunk_size=10000,
        progress_bar=True,
    )

    template_ind = np.load(output_folder_multi_sparse / "template_ind.npy")
    pc_ind = np.load(output_folder_multi_sparse / "pc_feature_ind.npy")
    # templates have different shapes!
    assert -1 in template_ind
    assert -1 in pc_ind
    # PC sparsity is more stringent than teplate sparsity
    assert pc_ind.shape[1] < template_ind.shape[1]


if __name__ == "__main__":
    we_sparse = make_waveforms_extractor(sparse=True)
    we_group = make_waveforms_extractor(sparse=False, with_group=True)
    we_dense = make_waveforms_extractor(sparse=False)

    test_export_to_phy(we_sparse)
    test_export_to_phy_by_property(we_group)
    test_export_to_phy_by_sparsity(we_dense)
