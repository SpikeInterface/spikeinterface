from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import warnings

from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from spikeinterface.core.job_tools import _shared_job_kwargs_doc, fix_job_kwargs
import spikeinterface.widgets as sw
from spikeinterface.core import get_template_amplitude_on_main_channel
from spikeinterface.postprocessing import compute_correlograms

_export_report_worker_context = None


def _save_unit_summary(sorting_analyzer, unit_id, units_folder, format, show_figures):
    import matplotlib.pyplot as plt

    fig = plt.figure(
        constrained_layout=False,
        figsize=(15, 7),
    )
    sw.plot_unit_summary(sorting_analyzer, unit_id, figure=fig)
    fig.suptitle(f"unit {unit_id}")
    fig.savefig(units_folder / f"{unit_id}.{format}")
    if not show_figures:
        plt.close(fig)


def _init_export_report_worker(
    analyzer_or_folder, analyzer_format, backend_options, units_folder, format, max_threads_per_worker
):
    global _export_report_worker_context

    if isinstance(analyzer_or_folder, (str, Path)):
        from spikeinterface.core import load_sorting_analyzer

        sorting_analyzer = load_sorting_analyzer(
            analyzer_or_folder,
            format=analyzer_format,
            backend_options=backend_options,
            lazy=True,
            load_extensions=False,
        )
    else:
        sorting_analyzer = analyzer_or_folder

    thread_limiter = None
    if max_threads_per_worker is not None:
        thread_limiter = threadpool_limits(limits=max_threads_per_worker)
    _export_report_worker_context = (
        sorting_analyzer,
        units_folder,
        format,
        thread_limiter,
    )


def _export_report_unit_worker(unit_id):
    sorting_analyzer, units_folder, format, _ = _export_report_worker_context
    _save_unit_summary(sorting_analyzer, unit_id, units_folder, format, show_figures=False)


def _extension_names_not_matching_disk_params(sorting_analyzer):
    # An extension name being both loaded and saved does not mean the loaded version matches the
    # saved one: SortingAnalyzer.compute(..., save=False) keeps an already-saved extension name but
    # recomputes its data/params in memory only (e.g. to try a different unit_locations method).
    # Workers reload from disk, so they would silently render with the saved parameters while the
    # main process (global figures) uses the fresh in-memory ones. Compare normalized params, not
    # just names, to also catch that supported recompute-with-save=False case.
    from spikeinterface.core.core_tools import check_json
    from spikeinterface.core.sortinganalyzer import get_extension_class

    saved_extensions = sorting_analyzer.get_saved_extension_names()
    mismatched_names = []
    for name in sorting_analyzer.get_loaded_extension_names():
        if name not in saved_extensions:
            mismatched_names.append(name)
            continue
        on_disk_extension = get_extension_class(name)(sorting_analyzer)
        on_disk_extension.load_params()
        loaded_params = check_json(sorting_analyzer.extensions[name].params)
        saved_params = check_json(on_disk_extension.params)
        if loaded_params != saved_params:
            mismatched_names.append(name)
    return mismatched_names


def _prepare_unit_summary_templates(sorting_analyzer):
    if not sorting_analyzer.has_extension("waveforms"):
        return

    templates_extension = sorting_analyzer.get_extension("templates")
    for percentile in (1, 25, 75, 99):
        templates_extension.get_templates(operator="percentile", percentile=percentile, save=False)
    if sorting_analyzer.format != "memory" and not sorting_analyzer.is_read_only():
        templates_extension.save()


def export_report(
    sorting_analyzer,
    output_folder,
    remove_if_exists=False,
    format="png",
    show_figures=False,
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
    show_figures : bool, default: False
        If True, figures are shown. If False, figures are closed after saving
    force_computation :  bool, default: False
        Force or not some heavy computation before exporting
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
            "export_report(): correlograms will not be exported. Use sorting_analyzer.compute('correlograms') if you want to include them."
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
    # max_on_channel_id is kept (old name)
    units["max_on_channel_id"] = sorting_analyzer.get_main_channels(outputs="id", with_dict=False)
    units["main_channel_id"] = sorting_analyzer.get_main_channels(outputs="id", with_dict=False)

    units["amplitude"] = pd.Series(get_template_amplitude_on_main_channel(sorting_analyzer))
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

    n_jobs = min(job_kwargs["n_jobs"], len(unit_ids))
    if platform.system() == "Windows":
        n_jobs = min(n_jobs, 61)
    pool_engine = job_kwargs["pool_engine"]
    if n_jobs > 1 and show_figures:
        warnings.warn("Set `show_figures=False` to enable parallel report export. Falling back to serial export.")
        n_jobs = 1
    if n_jobs > 1 and pool_engine != "process":
        warnings.warn(
            "Set `pool_engine='process'` to enable parallel report export because Matplotlib is not thread-safe. "
            "Falling back to serial export."
        )
        n_jobs = 1
    if n_jobs > 1 and sorting_analyzer.format == "zarr":
        warnings.warn(
            "Parallel report export from a Zarr SortingAnalyzer would load extension arrays once per worker. "
            "Use `n_jobs=1` or save the analyzer as `binary_folder`. Falling back to serial export."
        )
        n_jobs = 1

    if n_jobs > 1:
        mp_context = job_kwargs["mp_context"]
        if mp_context == "fork" and platform.system() == "Darwin":
            warnings.warn(
                'Use `mp_context="spawn"` on macOS because "fork" is not considered safe. '
                "Falling back to serial export."
            )
            n_jobs = 1
        if mp_context == "fork" and platform.system() == "Windows":
            raise ValueError("'fork' `mp_context` is not supported on Windows. Use `mp_context='spawn'` instead.")

        if n_jobs > 1:
            start_method = mp.get_context(mp_context).get_start_method()
            if sorting_analyzer.format == "memory" and start_method != "fork":
                warnings.warn(
                    "Save the SortingAnalyzer as `binary_folder` to use a spawn process context, or use "
                    "`mp_context='fork'` on Linux. Falling back to serial export."
                )
                n_jobs = 1
            elif sorting_analyzer.format != "memory" and sorting_analyzer.is_read_only():
                warnings.warn(
                    "Use `n_jobs=1` or a writable `binary_folder` SortingAnalyzer. Falling back to serial export."
                )
                n_jobs = 1
            elif sorting_analyzer.format != "memory":
                mismatched_extensions = _extension_names_not_matching_disk_params(sorting_analyzer)
                if mismatched_extensions:
                    warnings.warn(
                        "Parallel report export reloads the SortingAnalyzer from disk in each worker, so extensions "
                        f"computed in memory only or with different parameters than on disk ({', '.join(mismatched_extensions)}) "
                        "would not be visible there, or would be visible with different parameters (this happens for a lazy "
                        "SortingAnalyzer, or after sorting_analyzer.compute(..., save=False), for instance). "
                        "Save the current extension state or use `n_jobs=1`. Falling back to serial export."
                    )
                    n_jobs = 1

    if n_jobs > 1:
        _prepare_unit_summary_templates(sorting_analyzer)
        analyzer_or_folder = sorting_analyzer if sorting_analyzer.format == "memory" else str(sorting_analyzer.folder)
        init_args = (
            analyzer_or_folder,
            sorting_analyzer.format,
            sorting_analyzer._backend_options,
            units_folder,
            format,
            job_kwargs["max_threads_per_worker"],
        )
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=mp.get_context(mp_context),
            initializer=_init_export_report_worker,
            initargs=init_args,
        ) as executor:
            results = executor.map(_export_report_unit_worker, unit_ids)
            if job_kwargs["progress_bar"]:
                results = tqdm(results, total=len(unit_ids), desc=f"export_report (workers: {n_jobs} processes)")
            for _ in results:
                pass
    else:
        for unit_id in unit_ids:
            _save_unit_summary(sorting_analyzer, unit_id, units_folder, format, show_figures)


export_report.__doc__ = export_report.__doc__.format(_shared_job_kwargs_doc)
