from pathlib import Path
import shutil
import pandas as pd

from spikeinterface.core.job_tools import _shared_job_kwargs_doc
import spikeinterface.widgets as sw
import spikeinterface.toolkit as st

import matplotlib.pyplot as plt


def export_report(waveform_extractor, output_folder, remove_if_exists=False, format="png",
                  metrics=None, amplitudes=None, **job_wargs):
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
    metrics: pandas.DataFrame or None
        Quality metrics to export to csv. If None, quality metrics are computed.
    amplitudes: dict or None
        Amplitudes 'by_unit' as returned by the st.postprocessing.get_spike_amplitudes(..., output="by_unit") function.
        If None, amplitudes are computed.
    {}
    """
    we = waveform_extractor
    sorting = we.sorting
    unit_ids = sorting.unit_ids

    # lets matplotlib do this check svg is also cool
    # assert format in ["png", "pdf"], "'format' can be 'png' or 'pdf'"

    if amplitudes is None:
        # compute amplituds if not provided
        amplitudes = st.get_spike_amplitudes(we, peak_sign='neg', outputs='by_unit', **job_wargs)

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
    units['max_on_channel_id'] = pd.Series(st.get_template_extremum_channel(we, peak_sign='neg', outputs='id'))
    units['amplitude'] = pd.Series(st.get_template_extremum_amplitude(we, peak_sign='neg'))
    units.to_csv(output_folder / 'unit list.csv', sep='\t')

    # metrics
    if metrics is None:
        pca = st.compute_principal_components(we, load_if_exists=True,
                                              n_components=5, mode='by_channel_local')
        metrics = st.compute_quality_metrics(we, waveform_principal_component=pca)
    metrics.to_csv(output_folder / 'quality metrics.csv')

    unit_colors = sw.get_unit_colors(sorting)

    # global figures
    fig = plt.figure(figsize=(20, 10))
    w = sw.plot_unit_localization(we, figure=fig, unit_colors=unit_colors)
    fig.savefig(output_folder / f'unit_localization.{format}')

    fig, ax = plt.subplots(figsize=(20, 10))
    sw.plot_units_depth_vs_amplitude(we, ax=ax, unit_colors=unit_colors)
    fig.savefig(output_folder / f'units_depth_vs_amplitude.{format}')

    fig = plt.figure(figsize=(20, 10))
    sw.plot_amplitudes_distribution(we, amplitudes=amplitudes, figure=fig, unit_colors=unit_colors)
    fig.savefig(output_folder / f'amplitudes_distribution.{format}')

    # units
    units_folder = output_folder / 'units'
    units_folder.mkdir()

    for unit_id in unit_ids:
        fig = plt.figure(constrained_layout=False, figsize=(15, 7), )
        sw.plot_unit_summary(we, unit_id, amplitudes, figure=fig)
        fig.suptitle(f'unit {unit_id}')
        fig.savefig(units_folder / f'{unit_id}.{format}')


export_report.__doc__ = export_report.__doc__.format(_shared_job_kwargs_doc)
