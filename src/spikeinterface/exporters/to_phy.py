from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import shutil
import warnings

from spikeinterface.core import (
    write_binary_recording,
    BinaryRecordingExtractor,
    BinaryFolderRecording,
    ChannelSparsity,
    SortingAnalyzer,
)
from spikeinterface.core.job_tools import _shared_job_kwargs_doc, fix_job_kwargs


def export_to_phy(
    sorting_analyzer: SortingAnalyzer,
    output_folder: str | Path,
    compute_pc_features: bool = True,
    compute_amplitudes: bool = True,
    sparsity: Optional[ChannelSparsity] = None,
    copy_binary: bool = True,
    remove_if_exists: bool = False,
    template_mode: str = "average",
    add_quality_metrics: bool = True,
    add_template_metrics: bool = True,
    additional_properties: list | None = None,
    dtype: Optional[npt.DTypeLike] = None,
    verbose: bool = True,
    use_relative_path: bool = False,
    **job_kwargs,
):
    """
    Exports a sorting analyzer to the phy template-gui format.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object
    output_folder : str | Path
        The output folder where the phy template-gui files are saved
    compute_pc_features : bool, default: True
        If True, pc features are computed
    compute_amplitudes : bool, default: True
        If True, waveforms amplitudes are computed
    sparsity : ChannelSparsity or None, default: None
        The sparsity object
    copy_binary : bool, default: True
        If True, the recording is copied and saved in the phy "output_folder"
    remove_if_exists : bool, default: False
        If True and "output_folder" exists, it is removed and overwritten
    template_mode : str, default: "average"
        Parameter "mode" to be given to SortingAnalyzer.get_template()
    add_quality_metrics : bool, default: True
        If True, quality metrics (if computed) are saved as Phy tsv and will appear in the ClusterView.
    add_template_metrics : bool, default: True
        If True, template metrics (if computed) are saved as Phy tsv and will appear in the ClusterView.
    additional_properties : list | None, default: None
        List of additional properties to be saved as Phy tsv and will appear in the ClusterView.
    dtype : dtype or None, default: None
        Dtype to save binary data
    verbose : bool, default: True
        If True, output is verbose
    use_relative_path : bool, default: False
        If True and `copy_binary=True` saves the binary file `dat_path` in the `params.py` relative to `output_folder` (ie `dat_path=r"recording.dat"`). If `copy_binary=False`, then uses a path relative to the `output_folder`
        If False, uses an absolute path in the `params.py` (ie `dat_path=r"path/to/the/recording.dat"`)
    {}

    """
    import pandas as pd

    assert isinstance(sorting_analyzer, SortingAnalyzer), "sorting_analyzer must be a SortingAnalyzer object"
    sorting = sorting_analyzer.sorting

    assert (
        sorting_analyzer.get_num_segments() == 1
    ), f"Export to phy only works with one segment, your extractor has {sorting_analyzer.get_num_segments()} segments"
    num_chans = sorting_analyzer.get_num_channels()
    fs = sorting_analyzer.sampling_frequency

    job_kwargs = fix_job_kwargs(job_kwargs)

    # check sparsity
    if (num_chans > 64) and (sparsity is None and not sorting_analyzer.is_sparse()):
        warnings.warn(
            "Exporting to Phy with many channels and without sparsity might result in a heavy and less "
            "informative visualization. You can use use a sparse SortingAnalyzer or you can use the 'sparsity' "
            "argument to enforce sparsity (see compute_sparsity())"
        )

    save_sparse = True
    if sorting_analyzer.is_sparse():
        used_sparsity = sorting_analyzer.sparsity
        if sparsity is not None:
            warnings.warn("If the sorting_analyzer is sparse the 'sparsity' argument is ignored")
    elif sparsity is not None:
        used_sparsity = sparsity
    else:
        used_sparsity = ChannelSparsity.create_dense(sorting_analyzer)
        save_sparse = False
    # convenient sparsity dict for the 3 cases to retrieve channl_inds
    sparse_dict = used_sparsity.unit_id_to_channel_indices

    empty_flag = False
    non_empty_units = []
    for unit in sorting.unit_ids:
        if len(sorting.get_unit_spike_train(unit)) > 0:
            non_empty_units.append(unit)
        else:
            empty_flag = True
    unit_ids = non_empty_units
    if empty_flag:
        warnings.warn("Empty units have been removed while exporting to Phy")

    if len(unit_ids) == 0:
        raise Exception("No non-empty units in the sorting result, can't save to Phy.")

    output_folder = Path(output_folder).resolve()
    if output_folder.is_dir():
        if remove_if_exists:
            shutil.rmtree(output_folder)
        else:
            raise FileExistsError(f"{output_folder} already exists")

    output_folder.mkdir(parents=True)

    # save dat file
    if dtype is None:
        dtype = sorting_analyzer.get_dtype()

    if sorting_analyzer.has_recording():
        if copy_binary:
            rec_path = output_folder / "recording.dat"
            write_binary_recording(sorting_analyzer.recording, file_paths=rec_path, dtype=dtype, **job_kwargs)
        elif isinstance(sorting_analyzer.recording, BinaryRecordingExtractor):
            if isinstance(sorting_analyzer.recording, BinaryFolderRecording):
                bin_kwargs = sorting_analyzer.recording._bin_kwargs
            else:
                bin_kwargs = sorting_analyzer.recording._kwargs
            rec_path = bin_kwargs["file_paths"][0]
            dtype = sorting_analyzer.recording.get_dtype()
        else:
            rec_path = "None"
    else:  # don't save recording.dat
        if copy_binary:
            warnings.warn("Recording will not be copied since sorting_analyzer is recordingless.")
        rec_path = "None"

    dtype_str = np.dtype(dtype).name

    # write params.py
    with (output_folder / "params.py").open("w") as f:
        if use_relative_path:
            if copy_binary:
                f.write(f"dat_path = r'recording.dat'\n")
            elif rec_path == "None":
                f.write(f"dat_path = {rec_path}\n")
            else:
                f.write(f"dat_path = r'{str(Path(rec_path).relative_to(output_folder))}'\n")
        else:
            f.write(f"dat_path = r'{str(rec_path)}'\n")
        f.write(f"n_channels_dat = {num_chans}\n")
        f.write(f"dtype = '{dtype_str}'\n")
        f.write(f"offset = 0\n")
        f.write(f"sample_rate = {fs}\n")
        f.write(f"hp_filtered = {sorting_analyzer.is_filtered()}")

    # export spike_times/spike_templates/spike_clusters
    # here spike_labels is a remapping to unit_index
    all_spikes_seg0 = sorting.to_spike_vector(concatenated=False)[0]
    spike_times = all_spikes_seg0["sample_index"]
    spike_labels = all_spikes_seg0["unit_index"]
    np.save(str(output_folder / "spike_times.npy"), spike_times[:, np.newaxis])
    np.save(str(output_folder / "spike_templates.npy"), spike_labels[:, np.newaxis])
    np.save(str(output_folder / "spike_clusters.npy"), spike_labels[:, np.newaxis])

    # export templates/templates_ind/similar_templates
    # shape (num_units, num_samples, max_num_channels)
    templates_ext = sorting_analyzer.get_extension("templates")
    assert templates_ext is not None, "export_to_phy requires a SortingAnalyzer with the extension 'templates'"
    max_num_channels = max(len(chan_inds) for chan_inds in sparse_dict.values())
    dense_templates = templates_ext.get_templates(unit_ids=unit_ids, operator=template_mode)
    num_samples = dense_templates.shape[1]
    templates = np.zeros((len(unit_ids), num_samples, max_num_channels), dtype="float64")
    # here we pad template inds with -1 if len of sparse channels is unequal
    templates_ind = -np.ones((len(unit_ids), max_num_channels), dtype="int64")
    for unit_ind, unit_id in enumerate(unit_ids):
        chan_inds = sparse_dict[unit_id]
        template = dense_templates[unit_ind][:, chan_inds]
        templates[unit_ind, :, :][:, : len(chan_inds)] = template
        templates_ind[unit_ind, : len(chan_inds)] = chan_inds

    if not sorting_analyzer.has_extension("template_similarity"):
        sorting_analyzer.compute("template_similarity")
    template_similarity = sorting_analyzer.get_extension("template_similarity").get_data()

    np.save(str(output_folder / "templates.npy"), templates)
    if save_sparse:
        np.save(str(output_folder / "template_ind.npy"), templates_ind)
    np.save(str(output_folder / "similar_templates.npy"), template_similarity)

    channel_maps = np.arange(num_chans, dtype="int32")
    channel_map_si = sorting_analyzer.channel_ids
    channel_positions = sorting_analyzer.get_channel_locations().astype("float32")
    channel_groups = sorting_analyzer.get_recording_property("group")
    if channel_groups is None:
        channel_groups = np.zeros(num_chans, dtype="int32")
    np.save(str(output_folder / "channel_map.npy"), channel_maps)
    np.save(str(output_folder / "channel_map_si.npy"), channel_map_si)
    np.save(str(output_folder / "channel_positions.npy"), channel_positions)
    np.save(str(output_folder / "channel_groups.npy"), channel_groups)

    if compute_amplitudes:
        if not sorting_analyzer.has_extension("spike_amplitudes"):
            sorting_analyzer.compute("spike_amplitudes", **job_kwargs)
        amplitudes = sorting_analyzer.get_extension("spike_amplitudes").get_data()
        amplitudes = amplitudes[:, np.newaxis]
        np.save(str(output_folder / "amplitudes.npy"), amplitudes)

    if compute_pc_features:
        if not sorting_analyzer.has_extension("principal_components"):
            sorting_analyzer.compute("principal_components", n_components=5, mode="by_channel_local", **job_kwargs)

        pca_extension = sorting_analyzer.get_extension("principal_components")

        pca_extension.run_for_all_spikes(output_folder / "pc_features.npy", **job_kwargs)

        max_num_channels_pc = max(len(chan_inds) for chan_inds in used_sparsity.unit_id_to_channel_indices.values())
        pc_feature_ind = -np.ones((len(unit_ids), max_num_channels_pc), dtype="int64")
        for unit_ind, unit_id in enumerate(unit_ids):
            chan_inds = used_sparsity.unit_id_to_channel_indices[unit_id]
            pc_feature_ind[unit_ind, : len(chan_inds)] = chan_inds
        np.save(str(output_folder / "pc_feature_ind.npy"), pc_feature_ind)

    # Save .tsv metadata
    cluster_group = pd.DataFrame(
        {"cluster_id": [i for i in range(len(unit_ids))], "group": ["unsorted"] * len(unit_ids)}
    )
    cluster_group.to_csv(output_folder / "cluster_group.tsv", sep="\t", index=False)
    si_unit_ids = pd.DataFrame({"cluster_id": [i for i in range(len(unit_ids))], "si_unit_id": unit_ids})
    si_unit_ids.to_csv(output_folder / "cluster_si_unit_ids.tsv", sep="\t", index=False)

    unit_groups = sorting.get_property("group")
    if unit_groups is None:
        unit_groups = np.zeros(len(unit_ids), dtype="int32")
    channel_group = pd.DataFrame({"cluster_id": [i for i in range(len(unit_ids))], "channel_group": unit_groups})
    channel_group.to_csv(output_folder / "cluster_channel_group.tsv", sep="\t", index=False)

    if sorting_analyzer.has_extension("quality_metrics") and add_quality_metrics:
        qm_data = sorting_analyzer.get_extension("quality_metrics").get_data()
        for column_name in qm_data.columns:
            # already computed by phy
            if column_name not in ["num_spikes", "firing_rate"]:
                metric = pd.DataFrame(
                    {"cluster_id": [i for i in range(len(unit_ids))], column_name: qm_data[column_name].values}
                )
                metric.to_csv(output_folder / f"cluster_{column_name}.tsv", sep="\t", index=False)
    if sorting_analyzer.has_extension("template_metrics") and add_template_metrics:
        tm_data = sorting_analyzer.get_extension("template_metrics").get_data()
        for column_name in tm_data.columns:
            metric = pd.DataFrame(
                {"cluster_id": [i for i in range(len(unit_ids))], column_name: tm_data[column_name].values}
            )
            metric.to_csv(output_folder / f"cluster_{column_name}.tsv", sep="\t", index=False)
    if additional_properties is not None:
        for prop_name in additional_properties:
            prop_data = sorting.get_property(prop_name)
            if prop_data is not None:
                prop = pd.DataFrame({"cluster_id": [i for i in range(len(unit_ids))], prop_name: prop_data})
                prop.to_csv(output_folder / f"cluster_{prop_name}.tsv", sep="\t", index=False)

    if verbose:
        print("Run:\nphy template-gui ", str(output_folder / "params.py"))


export_to_phy.__doc__ = export_to_phy.__doc__.format(_shared_job_kwargs_doc)
