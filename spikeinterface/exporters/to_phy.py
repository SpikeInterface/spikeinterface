from pathlib import Path
import csv

import numpy as np
import shutil
import pandas as pd

import spikeinterface
from spikeinterface.core import write_binary_recording, BinaryRecordingExtractor
from spikeinterface.core.job_tools import _shared_job_kwargs_doc
from spikeinterface.postprocessing import (get_template_channel_sparsity,
                                           compute_spike_amplitudes, compute_template_similarity,
                                           compute_principal_components)


def export_to_phy(waveform_extractor, output_folder, compute_pc_features=True,
                  compute_amplitudes=True, sparsity=None, copy_binary=True,
                  remove_if_exists=False, peak_sign='neg', template_mode='median',
                  dtype=None, verbose=True, **job_kwargs):
    """
    Exports a waveform extractor to the phy template-gui format.

    Parameters
    ----------
    waveform_extractor: a WaveformExtractor or None
        If WaveformExtractor is provide then the compute is faster otherwise
    output_folder: str
        The output folder where the phy template-gui files are saved
    compute_pc_features: bool
        If True (default), pc features are computed
    compute_amplitudes: bool
        If True (default), waveforms amplitudes are computed
    sparsity: dict or None
        The sparsity dictionary (key: unit_id, value: list of channel_ids)
    copy_binary: bool
        If True, the recording is copied and saved in the phy 'output_folder'
    remove_if_exists: bool
        If True and 'output_folder' exists, it is removed and overwritten
    peak_sign: 'neg', 'pos', 'both'
        Used by compute_spike_amplitudes
    template_mode: str
        Parameter 'mode' to be given to WaveformExtractor.get_template()
    dtype: dtype or None
        Dtype to save binary data
    verbose: bool
        If True, output is verbose
    {}
    
    """
    assert isinstance(waveform_extractor, spikeinterface.core.waveform_extractor.WaveformExtractor), \
        'waveform_extractor must be a WaveformExtractor object'
    sorting = waveform_extractor.sorting
    assert waveform_extractor.has_recording(), "Export to phy is not supported for 'recordingless' waveform extractors"
    recording = waveform_extractor.recording

    assert waveform_extractor.get_num_segments() == 1, "Export to phy only works with one segment"
    channel_ids = recording.channel_ids
    num_chans = recording.get_num_channels()
    fs = recording.sampling_frequency

    # handle sparsity
    if sparsity is None:
        if num_chans > 64:
            print(
                "WARNING: export to Phy with many channels and without sparsity might result in a heavy and less "
                "informative visualization. You can use the 'sparsity' argument to enforce sparsity "
                "(see spikeinterface.postprocessing.get_template_channel_sparsity() function for sparsity details.)"
            )
        sparsity = {unit_id: channel_ids for unit_id in sorting.unit_ids}

    empty_flag = False
    non_empty_units = []
    for unit in sorting.unit_ids:
        if len(sorting.get_unit_spike_train(unit)) > 0:
            non_empty_units.append(unit)
        else:
            empty_flag = True
    unit_ids = non_empty_units
    if empty_flag:
        print('Warning: empty units have been removed when being exported to Phy')

    if not recording.is_filtered():
        print("Warning: recording is not filtered! It's recommended to filter the recording before exporting to phy.\n"
              "You can run spikeinterface.preprocessing.bandpass_filter(recording)")

    if len(unit_ids) == 0:
        raise Exception("No non-empty units in the sorting result, can't save to Phy.")

    output_folder = Path(output_folder).absolute()
    if output_folder.is_dir():
        if remove_if_exists:
            shutil.rmtree(output_folder)
        else:
            raise FileExistsError(f'{output_folder} already exists')

    output_folder.mkdir(parents=True)

    # save dat file
    if dtype is None:
        dtype = recording.get_dtype()

    if copy_binary:
        rec_path = output_folder / 'recording.dat'
        write_binary_recording(recording, file_paths=rec_path, verbose=verbose, dtype=dtype, **job_kwargs)
    elif isinstance(recording, BinaryRecordingExtractor):
        rec_path = recording._kwargs['file_paths'][0]
        dtype = recording.get_dtype()
    else:  # don't save recording.dat
        rec_path = 'None'

    dtype_str = np.dtype(dtype).name

    # write params.py
    with (output_folder / 'params.py').open('w') as f:
        f.write(f"dat_path = r'{str(rec_path)}'\n")
        f.write(f"n_channels_dat = {num_chans}\n")
        f.write(f"dtype = '{dtype_str}'\n")
        f.write(f"offset = 0\n")
        f.write(f"sample_rate = {fs}\n")
        f.write(f"hp_filtered = {recording.is_filtered()}")

    # export spike_times/spike_templates/spike_clusters
    # here spike_labels is a remapping to unit_index
    all_spikes = sorting.get_all_spike_trains(outputs='unit_index')
    spike_times, spike_labels = all_spikes[0]
    np.save(str(output_folder / 'spike_times.npy'), spike_times[:, np.newaxis])
    np.save(str(output_folder / 'spike_templates.npy'), spike_labels[:, np.newaxis])
    np.save(str(output_folder / 'spike_clusters.npy'), spike_labels[:, np.newaxis])

    # export templates/templates_ind/similar_templates
    # shape (num_units, num_samples, max_num_channels)
    templates = []
    templates_ind = []

    # here we pad template inds with -1 if len of sparse channels is unequal
    max_num_channels_templates = np.max([len(channels) for channels in sparsity.values()])
    for unit_id in unit_ids:
        template = waveform_extractor.get_template(unit_id, mode=template_mode, sparsity=sparsity)
        inds = waveform_extractor.recording.ids_to_indices(sparsity[unit_id])
        if template.shape[-1] < max_num_channels_templates:
            # fill missing channels
            template_full = np.zeros((template.shape[0], max_num_channels_templates))
            template_full[:, :template.shape[-1]] = template
            inds_full = np.concatenate((inds, np.array([-1] * (max_num_channels_templates - template.shape[-1]))))
        else:
            template_full = template
            inds_full = inds
        templates.append(template_full)
        templates_ind.append(inds_full)

    template_similarity = compute_template_similarity(waveform_extractor, method='cosine_similarity')

    np.save(str(output_folder / 'templates.npy'), templates)
    np.save(str(output_folder / 'template_ind.npy'), templates_ind)
    np.save(str(output_folder / 'similar_templates.npy'), template_similarity)

    channel_maps = np.arange(num_chans, dtype='int32')
    channel_map_si = waveform_extractor.recording.get_channel_ids()
    channel_positions = recording.get_channel_locations().astype('float32')
    channel_groups = recording.get_channel_groups()
    if channel_groups is None:
        channel_groups = np.zeros(num_chans, dtype='int32')
    np.save(str(output_folder / 'channel_map.npy'), channel_maps)
    np.save(str(output_folder / 'channel_map_si.npy'), channel_map_si)
    np.save(str(output_folder / 'channel_positions.npy'), channel_positions)
    np.save(str(output_folder / 'channel_groups.npy'), channel_groups)

    if compute_amplitudes:
        if waveform_extractor.is_extension('spike_amplitudes'):
            sac = waveform_extractor.load_extension('spike_amplitudes')
            amplitudes = sac.get_data(outputs='concatenated')
        else:
            amplitudes = compute_spike_amplitudes(waveform_extractor, peak_sign=peak_sign, outputs='concatenated', 
                                                  **job_kwargs)
        # one segment only
        amplitudes = amplitudes[0][:, np.newaxis]
        np.save(str(output_folder / 'amplitudes.npy'), amplitudes)

    if compute_pc_features:
        if waveform_extractor.is_extension('principal_components'):
            pc = waveform_extractor.load_extension('principal_components')
        else:
            pc = compute_principal_components(waveform_extractor, n_components=5, mode='by_channel_local')

        if pc.get_sparsity() is None:
            pc_sparsity = sparsity
            pc_feature_ind = templates_ind
        else:
            pc_sparsity = pc.get_sparsity()
            pc_feature_ind = None
        
        max_num_channels_pc = np.max([len(channels) for channels in pc_sparsity.values()])
        pc.run_for_all_spikes(output_folder / 'pc_features.npy',
                              sparsity=pc_sparsity, peak_sign=peak_sign,
                              **job_kwargs)

        if pc_feature_ind is None:
            pc_feature_ind = -np.ones((len(unit_ids), max_num_channels_pc), dtype='int64')
            for u, unit_id in enumerate(unit_ids):
                sparse_channel_indices = recording.ids_to_indices(pc_sparsity[unit_id])
                pc_feature_ind[u, :len(sparse_channel_indices)] = sparse_channel_indices
        np.save(str(output_folder / 'pc_feature_ind.npy'), pc_feature_ind)

    # Save .tsv metadata
    cluster_group = pd.DataFrame({'cluster_id': [i for i in range(len(unit_ids))],
                                  'group': ['unsorted'] * len(unit_ids)})
    cluster_group.to_csv(output_folder / 'cluster_group.tsv',
                         sep="\t", index=False)
    si_unit_ids = pd.DataFrame({'cluster_id': [i for i in range(len(unit_ids))],
                                'si_unit_id': unit_ids})
    si_unit_ids.to_csv(output_folder / 'cluster_si_unit_ids.tsv',
                       sep="\t", index=False)

    unit_groups = sorting.get_property('group')
    if unit_groups is None:
        unit_groups = np.zeros(len(unit_ids), dtype='int32')
    channel_group = pd.DataFrame({'cluster_id': [i for i in range(len(unit_ids))],
                                  'channel_group': unit_groups})
    channel_group.to_csv(output_folder / 'cluster_channel_group.tsv',
                         sep="\t", index=False)
    
    if waveform_extractor.is_extension('quality_metrics'):
        qm = waveform_extractor.load_extension('quality_metrics')
        qm_data = qm.get_data()
        for column_name in qm_data.columns:
            # already computed by phy
            if column_name not in ["num_spikes", "firing_rate"]:
                metric = pd.DataFrame({'cluster_id': [i for i in range(len(unit_ids))],
                                       column_name: qm_data[column_name].values})
                metric.to_csv(output_folder / f'cluster_{column_name}.tsv',
                              sep="\t", index=False)

    if verbose:
        print('Run:\nphy template-gui ', str(output_folder / 'params.py'))


export_to_phy.__doc__ = export_to_phy.__doc__.format(_shared_job_kwargs_doc)
