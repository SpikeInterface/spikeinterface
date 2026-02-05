from __future__ import annotations

from pathlib import Path
import shutil
import warnings

from spikeinterface.core.job_tools import _shared_job_kwargs_doc, fix_job_kwargs
import spikeinterface.widgets as sw
from spikeinterface.core import get_template_extremum_channel, get_template_extremum_amplitude
from spikeinterface.postprocessing import compute_correlograms


def export_report(
    sorting_analyzer,
    output_folder,
    remove_if_exists=False,
    format="png",
    show_figures=False,
    peak_sign="neg",
    force_computation=False,
    **job_kwargs,
):
    """
    Exports a SI spike sorting report. The report includes summary figures of the spike sorting output.
    What is plotted depends on what has been calculated. Unit locations and unit waveforms are always included.
    Unit waveform densities, correlograms and spike amplitudes are plotted if `waveforms`, `correlograms`,
    and `spike_amplitudes` have been computed for the given `sorting_analyzer`.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object
    output_folder : str
        The output folder where the report files are saved
    remove_if_exists : bool, default: False
        If True and the output folder exists, it is removed
    format : str, default: "png"
        The output figure format (any format handled by matplotlib)
    peak_sign : "neg" or "pos", default: "neg"
        used to compute amplitudes and metrics
    show_figures : bool, default: False
        If True, figures are shown. If False, figures are closed after saving
    force_computation :  bool, default: False
        Force or not some heavy computaion before exporting
    {}
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    job_kwargs = fix_job_kwargs(job_kwargs)
    sorting = sorting_analyzer.sorting
    unit_ids = sorting_analyzer.unit_ids

    # load or compute spike_amplitudes
    if sorting_analyzer.has_extension("spike_amplitudes"):
        spike_amplitudes = sorting_analyzer.get_extension("spike_amplitudes").get_data(outputs="by_unit")
    elif force_computation:
        sorting_analyzer.compute("spike_amplitudes", **job_kwargs)
        spike_amplitudes = sorting_analyzer.get_extension("spike_amplitudes").get_data(outputs="by_unit")
    else:
        spike_amplitudes = None
        warnings.warn(
            "export_report(): spike_amplitudes will not be exported. Use sorting_analyzer.compute('spike_amplitudes') if you want to include them."
        )

    # load or compute quality_metrics
    if sorting_analyzer.has_extension("quality_metrics"):
        metrics = sorting_analyzer.get_extension("quality_metrics").get_data()
    elif force_computation:
        sorting_analyzer.compute("quality_metrics")
        metrics = sorting_analyzer.get_extension("quality_metrics").get_data()
    else:
        metrics = None
        warnings.warn(
            "export_report(): quality metrics will not be exported. Use sorting_analyzer.compute('quality_metrics') if you want to include them."
        )

    # load or compute correlograms
    if sorting_analyzer.has_extension("correlograms"):
        correlograms, bins = sorting_analyzer.get_extension("correlograms").get_data()
    elif force_computation:
        correlograms, bins = compute_correlograms(sorting_analyzer, window_ms=100.0, bin_ms=1.0)
    else:
        correlograms = None
        warnings.warn(
            "export_report(): correlograms will not be exported. Use sorting_anlyzer.compute('correlograms') if you want to include them."
        )

    # pre-compute unit locations if not done
    if not sorting_analyzer.has_extension("unit_locations"):
        sorting_analyzer.compute("unit_locations")

    output_folder = Path(output_folder).absolute()
    if output_folder.is_dir():
        if remove_if_exists:
            shutil.rmtree(output_folder)
        else:
            raise FileExistsError(f"{output_folder} already exists")
    output_folder.mkdir(parents=True, exist_ok=True)

    # unit list
    units = pd.DataFrame(index=unit_ids)  #  , columns=['max_on_channel_id', 'amplitude'])
    units.index.name = "unit_id"
    units["max_on_channel_id"] = pd.Series(
        get_template_extremum_channel(sorting_analyzer, peak_sign=peak_sign, outputs="id")
    )
    units["amplitude"] = pd.Series(get_template_extremum_amplitude(sorting_analyzer, peak_sign=peak_sign))
    units.to_csv(output_folder / "unit list.csv", sep="\t")

    unit_colors = sw.get_unit_colors(sorting)

    # global figures
    fig = plt.figure(figsize=(20, 10))
    w = sw.plot_unit_locations(sorting_analyzer, figure=fig, unit_colors=unit_colors)
    fig.savefig(output_folder / f"unit_locations.{format}")
    if not show_figures:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(20, 10))
    sw.plot_unit_depths(sorting_analyzer, ax=ax, unit_colors=unit_colors)
    fig.savefig(output_folder / f"unit_depths.{format}")
    if not show_figures:
        plt.close(fig)

    if spike_amplitudes and len(unit_ids) < 100:
        fig = plt.figure(figsize=(20, 10))
        sw.plot_all_amplitudes_distributions(sorting_analyzer, figure=fig, unit_colors=unit_colors)
        fig.savefig(output_folder / f"amplitudes_distribution.{format}")
        if not show_figures:
            plt.close(fig)

    if metrics is not None:
        metrics.to_csv(output_folder / "quality metrics.csv")

    # units
    units_folder = output_folder / "units"
    units_folder.mkdir()

    for unit_id in unit_ids:
        fig = plt.figure(
            constrained_layout=False,
            figsize=(15, 7),
        )
        sw.plot_unit_summary(sorting_analyzer, unit_id, figure=fig)
        fig.suptitle(f"unit {unit_id}")
        fig.savefig(units_folder / f"{unit_id}.{format}")
        if not show_figures:
            plt.close(fig)


export_report.__doc__ = export_report.__doc__.format(_shared_job_kwargs_doc)
