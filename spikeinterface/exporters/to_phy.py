from pathlib import Path
import csv

import numpy as np
import shutil

import spikeinterface
from spikeinterface.core import write_binary_recording, BinaryRecordingExtractor
from spikeinterface.core.job_tools import _shared_job_kwargs_doc
from spikeinterface.toolkit import (get_template_channel_sparsity,
                                    get_spike_amplitudes, compute_template_similarity,
                                    WaveformPrincipalComponent)


def export_to_phy(waveform_extractor, output_folder, compute_pc_features=True,
                  compute_amplitudes=True, by_property=None,
                  max_channels_per_template=16,
                  copy_binary=True,
                  remove_if_exists=False,
                  peak_sign='neg', template_mode='median',
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
        If True (default), waveforms amplitudes are compute
    by_property: object or None
        If given and 'by_property' is a property of both the associated recording and sorting objects,
        the templates are exported split by the provided 'by_property' (e.g. "group")
    max_channels_per_template: int or None
        Maximum channels per unit to return. If None, all channels are returned
    copy_binary: bool
        If True, the recording is copied and saved in the phy 'output_folder'.
    remove_if_exists: bool
        If True and 'output_folder' exists, it is removed and overwritten.
    peak_sign: 'neg', 'pos', 'both'
        Used by get_spike_amplitudes
    template_mode: str
        Parameter 'mode' to be given to WaveformExtractor.get_template()
    dtype: dtype or None

    verbose: bool
        If True, output is verbose.
    {}
    """
    assert isinstance(waveform_extractor, spikeinterface.core.waveform_extractor.WaveformExtractor), \
        'waveform_extractor must be a WaveformExtractor object'
    recording = waveform_extractor.recording
    sorting = waveform_extractor.sorting

    assert recording.get_num_segments() == sorting.get_num_segments(), \
        "The recording and sorting objects must have the same number of segments!"

    assert recording.get_num_segments() == 1, "Export to phy work only with one segment"

    unit_ids = sorting.unit_ids
    channel_ids = recording.channel_ids
    num_chans = recording.get_num_channels()
    fs = recording.get_sampling_frequency()

    # phy don't support unit_ids as str we need to remap
    remap_unit_ids = np.arange(unit_ids.size)

    empty_flag = False
    non_empty_units = []
    for unit in sorting.get_unit_ids():
        if len(sorting.get_unit_spike_train(unit)) > 0:
            non_empty_units.append(unit)
        else:
            empty_flag = True
    unit_ids = non_empty_units
    if empty_flag:
        print('Warning: empty units have been removed when being exported to Phy')

    if not recording.is_filtered():
        print("Warning: recording is not filtered! It's recommended to filter the recording before exporting to phy.\n"
              "You can run spikeinterface.toolkit.preprocessing.bandpass_filter(recording)")

    if len(unit_ids) == 0:
        raise Exception("No non-empty units in the sorting result, can't save to Phy.")

    output_folder = Path(output_folder).absolute()
    if output_folder.is_dir():
        if remove_if_exists:
            shutil.rmtree(output_folder)
        else:
            raise FileExistsError(f'{output_folder} already exists')

    output_folder.mkdir()

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
    # shape (num_units, num_samples, num_channels)
    templates = []
    templates_ind = []
    if by_property is not None:
        rec_by = waveform_extractor.recording.split_by(by_property)
        num_channels = np.max([rec.get_num_channels() for rec in rec_by.values()])
        for unit_id in unit_ids:
            template = waveform_extractor.get_template(unit_id, mode=template_mode, by_property=by_property)
            _, inds = waveform_extractor.get_waveforms(unit_id, with_channel_index=True, by_property=by_property)
            if max_channels_per_template is None:
                inds = np.arange(channel_ids, dtype='int64')
            else:
                if max_channels_per_template < num_channels:
                    amps = np.max(np.abs(template), axis=0)
                    inds = np.argsort(amps)[::-1]
                    inds = inds[:max_channels_per_template]
                    template = template[:, inds]
            if template.shape[-1] < num_channels:
                # fix missing channels
                template_full = np.zeros((template.shape[0], num_channels))
                template_full[:, :template.shape[-1]] = template
                inds_full = np.concatenate((inds, np.array([-1] * (num_channels - template.shape[-1]))))
            else:
                template_full = template
                inds_full = inds
            templates.append(template_full)
            templates_ind.append(inds_full)
    else:
        template_sparsity = None
        if max_channels_per_template is not None:
            if max_channels_per_template < recording.get_num_channels():
                template_sparsity = get_template_channel_sparsity(waveform_extractor, method="best_channels",
                                                                  num_channels=max_channels_per_template,
                                                                  outputs="index")
        for unit_id in unit_ids:
            template = waveform_extractor.get_template(unit_id, mode=template_mode)
            if max_channels_per_template is None and template_sparsity is None:
                inds = np.arange(channel_ids, dtype='int64')
            else:
                inds = template_sparsity[unit_id]
                template = template[:, inds]
            templates.append(template.astype('float32'))
            templates_ind.append(inds)
    templates = np.array(templates)
    templates_ind = np.array(templates_ind)

    template_similarity = compute_template_similarity(waveform_extractor, method='cosine_similarity')

    np.save(str(output_folder / 'templates.npy'), templates)
    np.save(str(output_folder / 'template_ind.npy'), templates_ind)
    np.save(str(output_folder / 'similar_templates.npy'), template_similarity)

    channel_maps = np.arange(num_chans, dtype='int32')
    channel_map_si = unit_ids
    channel_positions = recording.get_channel_locations().astype('float32')
    channel_groups = recording.get_channel_groups()
    if channel_groups is None:
        channel_groups = np.zeros(num_chans, dtype='int32')
    np.save(str(output_folder / 'channel_map.npy'), channel_maps)
    np.save(str(output_folder / 'channel_map_si.npy'), channel_map_si)
    np.save(str(output_folder / 'channel_positions.npy'), channel_positions)
    np.save(str(output_folder / 'channel_groups.npy'), channel_groups)

    if compute_amplitudes:
        amplitudes = get_spike_amplitudes(waveform_extractor, peak_sign=peak_sign, outputs='concatenated', **job_kwargs)
        # one segment only
        amplitudes = amplitudes[0][:, np.newaxis]
        np.save(str(output_folder / 'amplitudes.npy'), amplitudes)

    if compute_pc_features:
        pc = WaveformPrincipalComponent(waveform_extractor)
        pc.set_params(n_components=5, mode='by_channel_local')
        pc.run_for_all_spikes(output_folder / 'pc_features.npy',
                              max_channels_per_template=max_channels_per_template, peak_sign=peak_sign,
                              **job_kwargs)

        max_channels_per_template = min(max_channels_per_template, len(channel_ids))
        pc_feature_ind = np.zeros((len(unit_ids), max_channels_per_template), dtype='int64')
        best_channels_index = get_template_channel_sparsity(waveform_extractor, method='best_channels',
                                                            peak_sign=peak_sign, num_channels=max_channels_per_template,
                                                            outputs='index')

        for u, unit_id in enumerate(sorting.unit_ids):
            pc_feature_ind[u, :] = best_channels_index[unit_id]
        np.save(str(output_folder / 'pc_feature_ind.npy'), pc_feature_ind)

    # Save .tsv metadata
    with (output_folder / 'cluster_group.tsv').open('w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        writer.writerow(['cluster_id', 'group'])
        for i, u in enumerate(sorting.get_unit_ids()):
            writer.writerow([i, 'unsorted'])

    unit_groups = sorting.get_property('group')
    if unit_groups is None:
        unit_groups = np.zeros(len(unit_ids), dtype='int32')

    with (output_folder / 'cluster_channel_group.tsv').open('w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        writer.writerow(['cluster_id', 'channel_group'])
        for i, unit_id in enumerate(unit_ids):
            writer.writerow([i, unit_groups[i]])

    if verbose:
        print('Run:\nphy template-gui ', str(output_folder / 'params.py'))


export_to_phy.__doc__ = export_to_phy.__doc__.format(_shared_job_kwargs_doc)
