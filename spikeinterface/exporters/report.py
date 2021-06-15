from pathlib import Path

import pandas as pd

import spikeinterface.widgets as sw
import spikeinterface.toolkit as st

import matplotlib.pyplot as plt

def export_report(waveform_extractor, output_folder, remove_if_exists=False, **job_wargs):
    we = waveform_extractor
    sorting = we.sorting
    unit_ids = sorting.unit_ids
    
    # some computation
    amplitudes = st.get_unit_amplitudes(we,  peak_sign='neg', outputs='by_units', **job_wargs)
    
    output_folder = Path(output_folder).absolute()
    if output_folder.is_dir():
        if remove_if_exists:
            shutil.rmtree(output_folder)
        else:
            raise FileExistsError(f'{output_folder} already exists')
    output_folder.mkdir()
    
    # unit list
    units = pd.DataFrame(index=unit_ids)# , columns=['max_on_channel_id', 'amplitude'])
    units.index.name = 'unit_id'
    units['max_on_channel_id'] = pd.Series(st.get_template_extremum_channel(we, peak_sign='neg', outputs='id'))
    units['amplitude'] = pd.Series(st.get_template_extremum_amplitude(we, peak_sign='neg'))
    units.to_csv(output_folder / 'unit list.csv', sep='\t')
    
    # metrics
    pca = st.WaveformPrincipalComponent(we)
    pca.set_params(n_components=5, mode='by_channel_local')
    pca.run()    
    metrics = st.compute_quality_metrics(we, waveform_principal_component=pca)
    metrics.to_csv(output_folder / 'quality metrics.csv')
    
    unit_colors = sw.get_unit_colors(sorting)
    
    # global figures
    fig = plt.figure(figsize=(20, 10))
    w = sw.plot_unit_localization(we, figure=fig, unit_colors=unit_colors)
    fig.savefig(output_folder / 'unit_localization.png')
    
    fig, ax = plt.subplots(figsize=(20, 10))
    sw.plot_units_depth_vs_amplitude(we, ax=ax, unit_colors=unit_colors)
    fig.savefig(output_folder / 'units_depth_vs_amplitude.png')
    
    fig = plt.figure(figsize=(20, 10))
    sw.plot_amplitudes_distribution(we, figure=fig, unit_colors=unit_colors)
    fig.savefig(output_folder / 'amplitudes_distribution.png')

    # units
    units_folder = output_folder / 'units'
    units_folder.mkdir()

    for unit_id in unit_ids:
        fig = plt.figure(constrained_layout=False, figsize=(15, 7),)
        sw.plot_unit_summary(we, unit_id, amplitudes, figure=fig)
        fig.suptitle(f'unit {unit_id}')
        fig.savefig(units_folder / f'{unit_id}.png')
    