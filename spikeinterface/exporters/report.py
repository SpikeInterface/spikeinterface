from pathlib import Path


import spikeinterface.widgets as sw
import spikeinterface.toolkit as st

import matplotlib.pyplot as plt

def export_report(waveform_extractor, output_folder, remove_if_exists=False):
    we = waveform_extractor
    sorting = we.sorting
    unit_ids = sorting.unit_ids
    
    
    output_folder = Path(output_folder).absolute()
    if output_folder.is_dir():
        if remove_if_exists:
            shutil.rmtree(output_folder)
        else:
            raise FileExistsError(f'{output_folder} already exists')
    output_folder.mkdir()
    
    print(we)
    print(output_folder)
    
    pca = st.WaveformPrincipalComponent(we)
    pca.set_params(n_components=5, mode='by_channel_local')
    pca.run()    
    metrics = st.compute_quality_metrics(we, waveform_principal_component=pca)
    metrics.to_excel(output_folder / 'quality metrics.xlsx')
    
    fig = plt.figure(figsize=(20, 10))
    w = sw.plot_unit_localization(we, figure=fig)
    fig.savefig(output_folder / 'unit_localization.png')
    
    fig, ax = plt.subplots(figsize=(20, 10))
    sw.plot_units_depth_vs_amplitude(we,ax=ax)
    fig.savefig(output_folder / 'units_depth_vs_amplitude.png')
    
    fig = plt.figure(figsize=(20, 10))
    sw.plot_amplitudes_distribution(we, figure=fig)
    fig.savefig(output_folder / 'amplitudes_distribution.png')
    
    # units
    units_folder = output_folder / 'units'
    units_folder.mkdir()
    
    for unit_id in unit_ids[:2]:
        print(unit_id)
        
        fig, axs = plt.subplots(figsize=(20, 10), nrows=2, ncols=2)
        
        sw.plot_unit_probe_map(we, unit_ids=[unit_id],  axes=[axs[0,0]])
        sw.plot_unit_waveforms(we, unit_ids=[unit_id], radius_um=60, ax=axs[0,1])
        sw.plot_unit_waveform_density_map(we, unit_ids=[unit_id], max_channels=1, ax=axs[1,1], same_axis=True)
        sw.plot_isi_distribution(sorting, unit_ids=[unit_id],  window_ms=500.0, bin_ms=5.0,  ax=axs[1,0])
        
        # TODO
        # plot_amplitudes_timeseries
        
        fig.suptitle(f'unit {unit_id}')
        fig.savefig(units_folder / f'{unit_id}.png')
    