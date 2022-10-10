from pathlib import Path
import shutil
import pandas as pd

from spikeinterface.core.job_tools import _shared_job_kwargs_doc
import spikeinterface.widgets as sw
from spikeinterface.postprocessing import (compute_spike_amplitudes,
                                           compute_principal_components,
                                           compute_unit_locations,
                                           get_template_extremum_channel, 
                                           get_template_extremum_amplitude)
from spikeinterface.qualitymetrics import compute_quality_metrics

import matplotlib.pyplot as plt


def export_report(waveform_extractor, output_folder, remove_if_exists=False, format="png",
                  show_figures=False, peak_sign='neg', **job_kwargs):
    """
    Exports a SI spike sorting report. The report includes summary figures of the spike sorting output
    (e.g. amplitude distributions, unit localization and depth VS amplitude) as well as unit-specific reports,
    that include waveforms, templates, template maps, ISI distributions, and more.
    
    
    Parameters
    ----------
    waveform_extractor: a WaveformExtractor or None
        If WaveformExtractor is provide then the compute is faster otherwise
    output_folder: str
        The output folder where the report files are saved
    remove_if_exists: bool
        If True and the output folder exists, it is removed
    format: str
        'png' (default) or 'pdf' or any format handled by matplotlib
    peak_sign: 'neg' or 'pos'
        used to compute amplitudes and metrics
    show_figures: bool
        If True, figures are shown. If False (default), figures are closed after saving.
    {}
    """
    we = waveform_extractor
    sorting = we.sorting
    unit_ids = sorting.unit_ids

    # lets matplotlib do this check svg is also cool
    # assert format in ["png", "pdf"], "'format' can be 'png' or 'pdf'"

    if we.is_extension('spike_amplitudes'):
        sac = we.load_extension('spike_amplitudes')
        amplitudes = sac.get_data(outputs='by_unit')
    else:
        amplitudes = compute_spike_amplitudes(we, peak_sign=peak_sign, outputs='by_unit', **job_kwargs)

    output_folder = Path(output_folder).absolute()
    if output_folder.is_dir():
        if remove_if_exists:
            shutil.rmtree(output_folder)
        else:
            raise FileExistsError(f'{output_folder} already exists')
    output_folder.mkdir(parents=True, exist_ok=True)

    # unit list
    units = pd.DataFrame(index=unit_ids)  #  , columns=['max_on_channel_id', 'amplitude'])
    units.index.name = 'unit_id'
    units['max_on_channel_id'] = pd.Series(get_template_extremum_channel(we, peak_sign='neg', outputs='id'))
    units['amplitude'] = pd.Series(get_template_extremum_amplitude(we, peak_sign='neg'))
    units.to_csv(output_folder / 'unit list.csv', sep='\t')
    
    # metrics
    if we.is_extension('quality_metrics'):
        qmc = we.load_extension('quality_metrics')
        metrics = qmc.get_data()
    else:
        # compute principal_components if not done
        if not we.is_extension('principal_components'):
            pca = compute_principal_components(we, load_if_exists=True,
                                               n_components=5, mode='by_channel_local')
        metrics = compute_quality_metrics(we)
    metrics.to_csv(output_folder / 'quality metrics.csv')

    # pre-compute unit locations
    if not we.is_extension('unit_locations'):
        unit_locations = compute_unit_locations(we)

    unit_colors = sw.get_unit_colors(sorting)

    # global figures
    fig = plt.figure(figsize=(20, 10))
    w = sw.plot_unit_locations(we, figure=fig, unit_colors=unit_colors)
    fig.savefig(output_folder / f'unit_localization.{format}')
    if not show_figures:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(20, 10))
    sw.plot_units_depth_vs_amplitude(we, ax=ax, unit_colors=unit_colors)
    fig.savefig(output_folder / f'units_depth_vs_amplitude.{format}')
    if not show_figures:
        plt.close(fig)

    fig = plt.figure(figsize=(20, 10))
    sw.plot_amplitudes_distribution(we, figure=fig, unit_colors=unit_colors)
    fig.savefig(output_folder / f'amplitudes_distribution.{format}')
    if not show_figures:
        plt.close(fig)

    # units
    units_folder = output_folder / 'units'
    units_folder.mkdir()

    for unit_id in unit_ids:
        fig = plt.figure(constrained_layout=False, figsize=(15, 7), )
        sw.plot_unit_summary(we, unit_id, figure=fig)
        fig.suptitle(f'unit {unit_id}')
        fig.savefig(units_folder / f'{unit_id}.{format}')
        if not show_figures:
            plt.close(fig)


export_report.__doc__ = export_report.__doc__.format(_shared_job_kwargs_doc)
