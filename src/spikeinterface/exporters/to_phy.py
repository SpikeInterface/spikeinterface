from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import shutil
import warnings

import spikeinterface
from spikeinterface.core import (
    write_binary_recording,
    BinaryRecordingExtractor,
    WaveformExtractor,
    BinaryFolderRecording,
    ChannelSparsity,
)
from spikeinterface.core.job_tools import _shared_job_kwargs_doc, fix_job_kwargs
from spikeinterface.postprocessing import (
    compute_spike_amplitudes,
    compute_template_similarity,
    compute_principal_components,
)


def export_to_phy(
    waveform_extractor: WaveformExtractor,
    output_folder: str | Path,
    compute_pc_features: bool = True,
    compute_amplitudes: bool = True,
    sparsity: Optional[ChannelSparsity] = None,
    copy_binary: bool = True,
    remove_if_exists: bool = False,
    peak_sign: Literal["both", "neg", "pos"] = "neg",
    template_mode: str = "median",
    dtype: Optional[npt.DTypeLike] = None,
    verbose: bool = True,
    use_relative_path: bool = False,
    **job_kwargs,
):
    """
    Exports a waveform extractor to the phy template-gui format.

    Parameters
    ----------
    waveform_extractor: a WaveformExtractor or None
        If WaveformExtractor is provide then the compute is faster otherwise
    output_folder: str | Path
        The output folder where the phy template-gui files are saved
    compute_pc_features: bool, default: True
        If True, pc features are computed
    compute_amplitudes: bool, default: True
        If True, waveforms amplitudes are computed
    sparsity: ChannelSparsity or None, default: None
        The sparsity object
    copy_binary: bool, default: True
        If True, the recording is copied and saved in the phy "output_folder"
    remove_if_exists: bool, default: False
        If True and "output_folder" exists, it is removed and overwritten
    peak_sign: "neg" | "pos" | "both", default: "neg"
        Used by compute_spike_amplitudes
    template_mode: str, default: "median"
        Parameter "mode" to be given to WaveformExtractor.get_template()
    dtype: dtype or None, default: None
        Dtype to save binary data
    verbose: bool, default: True
        If True, output is verbose
    use_relative_path : bool, default: False
        If True and `copy_binary=True` saves the binary file `dat_path` in the `params.py` relative to `output_folder` (ie `dat_path=r"recording.dat"`). If `copy_binary=False`, then uses a path relative to the `output_folder`
        If False, uses an absolute path in the `params.py` (ie `dat_path=r"path/to/the/recording.dat"`)
    {}

    """
    import pandas as pd

    assert isinstance(
        waveform_extractor, spikeinterface.core.waveform_extractor.WaveformExtractor
    ), "waveform_extractor must be a WaveformExtractor object"
    sorting = waveform_extractor.sorting

    assert (
        waveform_extractor.get_num_segments() == 1
    ), f"Export to phy only works with one segment, your extractor has {waveform_extractor.get_num_segments()} segments"
    num_chans = waveform_extractor.get_num_channels()
    fs = waveform_extractor.sampling_frequency

    job_kwargs = fix_job_kwargs(job_kwargs)

    # check sparsity
    if (num_chans > 64) and (sparsity is None and not waveform_extractor.is_sparse()):
        warnings.warn(
            "Exporting to Phy with many channels and without sparsity might result in a heavy and less "
            "informative visualization. You can use use a sparse WaveformExtractor or you can use the 'sparsity' "
            "argument to enforce sparsity (see compute_sparsity())"
        )

    save_sparse = True
    if waveform_extractor.is_sparse():
        used_sparsity = waveform_extractor.sparsity
        if sparsity is not None:
            warnings.warn("If the waveform_extractor is sparse the 'sparsity' argument is ignored")
    elif sparsity is not None:
        used_sparsity = sparsity
    else:
        used_sparsity = ChannelSparsity.create_dense(waveform_extractor)
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

    output_folder = Path(output_folder).absolute()
    if output_folder.is_dir():
        if remove_if_exists:
            shutil.rmtree(output_folder)
        else:
            raise FileExistsError(f"{output_folder} already exists")

    output_folder.mkdir(parents=True)

    # save dat file
    if dtype is None:
        if waveform_extractor.has_recording():
            dtype = waveform_extractor.recording.get_dtype()
        else:
            dtype = waveform_extractor.dtype

    if waveform_extractor.has_recording():
        if copy_binary:
            rec_path = output_folder / "recording.dat"
            write_binary_recording(waveform_extractor.recording, file_paths=rec_path, dtype=dtype, **job_kwargs)
        elif isinstance(waveform_extractor.recording, BinaryRecordingExtractor):
            if isinstance(waveform_extractor.recording, BinaryFolderRecording):
                bin_kwargs = waveform_extractor.recording._bin_kwargs
            else:
                bin_kwargs = waveform_extractor.recording._kwargs
            rec_path = bin_kwargs["file_paths"][0]
            dtype = waveform_extractor.recording.get_dtype()
        else:
            rec_path = "None"
    else:  # don't save recording.dat
        if copy_binary:
            warnings.warn("Recording will not be copied since waveform extractor is recordingless.")
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
        f.write(f"hp_filtered = {waveform_extractor.is_filtered()}")

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
    max_num_channels = max(len(chan_inds) for chan_inds in sparse_dict.values())
    num_samples = waveform_extractor.nbefore + waveform_extractor.nafter
    templates = np.zeros((len(unit_ids), num_samples, max_num_channels), dtype="float64")
    # here we pad template inds with -1 if len of sparse channels is unequal
    templates_ind = -np.ones((len(unit_ids), max_num_channels), dtype="int64")
    for unit_ind, unit_id in enumerate(unit_ids):
        chan_inds = sparse_dict[unit_id]
        template = waveform_extractor.get_template(unit_id, mode=template_mode, sparsity=sparsity)
        templates[unit_ind, :, :][:, : len(chan_inds)] = template
        templates_ind[unit_ind, : len(chan_inds)] = chan_inds

    if waveform_extractor.has_extension("similarity"):
        tmc = waveform_extractor.load_extension("similarity")
        template_similarity = tmc.get_data()
    else:
        template_similarity = compute_template_similarity(waveform_extractor, method="cosine_similarity")

    np.save(str(output_folder / "templates.npy"), templates)
    if save_sparse:
        np.save(str(output_folder / "template_ind.npy"), templates_ind)
    np.save(str(output_folder / "similar_templates.npy"), template_similarity)

    channel_maps = np.arange(num_chans, dtype="int32")
    channel_map_si = waveform_extractor.channel_ids
    channel_positions = waveform_extractor.get_channel_locations().astype("float32")
    channel_groups = waveform_extractor.get_recording_property("group")
    if channel_groups is None:
        channel_groups = np.zeros(num_chans, dtype="int32")
    np.save(str(output_folder / "channel_map.npy"), channel_maps)
    np.save(str(output_folder / "channel_map_si.npy"), channel_map_si)
    np.save(str(output_folder / "channel_positions.npy"), channel_positions)
    np.save(str(output_folder / "channel_groups.npy"), channel_groups)

    if compute_amplitudes:
        if waveform_extractor.has_extension("spike_amplitudes"):
            sac = waveform_extractor.load_extension("spike_amplitudes")
            amplitudes = sac.get_data(outputs="concatenated")
        else:
            amplitudes = compute_spike_amplitudes(
                waveform_extractor, peak_sign=peak_sign, outputs="concatenated", **job_kwargs
            )
        # one segment only
        amplitudes = amplitudes[0][:, np.newaxis]
        np.save(str(output_folder / "amplitudes.npy"), amplitudes)

    if compute_pc_features:
        if waveform_extractor.has_extension("principal_components"):
            pc = waveform_extractor.load_extension("principal_components")
        else:
            pc = compute_principal_components(
                waveform_extractor, n_components=5, mode="by_channel_local", sparsity=sparsity
            )
        pc_sparsity = pc.get_sparsity()
        if pc_sparsity is None:
            pc_sparsity = used_sparsity
        max_num_channels_pc = max(len(chan_inds) for chan_inds in pc_sparsity.unit_id_to_channel_indices.values())

        pc.run_for_all_spikes(output_folder / "pc_features.npy", **job_kwargs)

        pc_feature_ind = -np.ones((len(unit_ids), max_num_channels_pc), dtype="int64")
        for unit_ind, unit_id in enumerate(unit_ids):
            chan_inds = pc_sparsity.unit_id_to_channel_indices[unit_id]
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

    if waveform_extractor.has_extension("quality_metrics"):
        qm = waveform_extractor.load_extension("quality_metrics")
        qm_data = qm.get_data()
        for column_name in qm_data.columns:
            # already computed by phy
            if column_name not in ["num_spikes", "firing_rate"]:
                metric = pd.DataFrame(
                    {"cluster_id": [i for i in range(len(unit_ids))], column_name: qm_data[column_name].values}
                )
                metric.to_csv(output_folder / f"cluster_{column_name}.tsv", sep="\t", index=False)

    if verbose:
        print("Run:\nphy template-gui ", str(output_folder / "params.py"))


export_to_phy.__doc__ = export_to_phy.__doc__.format(_shared_job_kwargs_doc)
