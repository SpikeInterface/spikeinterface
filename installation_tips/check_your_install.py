from pathlib import Path
import platform
import shutil
import argparse
import warnings

warnings.filterwarnings("ignore")


job_kwargs = dict(n_jobs=-1, progress_bar=False, chunk_duration="1s")


def check_import_si():
    import spikeinterface as si


def check_import_si_full():
    import spikeinterface.full as si


def _create_recording(short=False):
    import spikeinterface.full as si

    if short:
        durations = [30.0]
    else:
        durations = [200.0]

    rec, _ = si.generate_ground_truth_recording(
        durations=durations, sampling_frequency=30_000.0, num_channels=16, num_units=10, seed=2205
    )
    rec.save(folder="./toy_example_recording", verbose=False, **job_kwargs)


def _run_one_sorter_and_analyzer(sorter_name):
    import spikeinterface.full as si

    si.set_global_job_kwargs(**job_kwargs)

    recording = si.load("./toy_example_recording")
    sorting = si.run_sorter(sorter_name, recording, folder=f"./sorter_with_{sorter_name}", verbose=False)

    sorting_analyzer = si.create_sorting_analyzer(
        sorting, recording, format="binary_folder", folder=f"./analyzer_with_{sorter_name}"
    )
    sorting_analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
    sorting_analyzer.compute("waveforms")
    sorting_analyzer.compute("templates")
    sorting_analyzer.compute("noise_levels")
    sorting_analyzer.compute("unit_locations", method="monopolar_triangulation")
    sorting_analyzer.compute("correlograms", window_ms=100, bin_ms=5.0)
    sorting_analyzer.compute("principal_components", n_components=3, mode="by_channel_global", whiten=True)
    sorting_analyzer.compute("quality_metrics", metric_names=["snr", "firing_rate"])


def run_tridesclous2():
    _run_one_sorter_and_analyzer("tridesclous2")


def run_kilosort4():
    _run_one_sorter_and_analyzer("kilosort4")


def open_sigui():
    import spikeinterface.full as si
    from spikeinterface_gui import run_mainwindow

    sorter_name = "tridesclous2"
    folder = f"./analyzer_with_{sorter_name}"
    analyzer = si.load_sorting_analyzer(folder)

    win = run_mainwindow(analyzer, start_app=True)


def _clean():
    # clean
    folders = [
        "./toy_example_recording",
        "./sorter_with_tridesclous2",
        "./analyzer_with_tridesclous2",
        "./sorter_with_kilosort4",
        "./analyzer_with_kilosort4",
    ]
    for folder in folders:
        if Path(folder).exists():
            shutil.rmtree(folder)


parser = argparse.ArgumentParser()
# add ci flag so that gui will not be used in ci
# end user can ignore
parser.add_argument("--ci", action="store_false")
parser.add_argument("--short", action="store_false")
parser.add_argument("--skip-kilosort4", action="store_true")


if __name__ == "__main__":

    args = parser.parse_args()

    _clean()
    _create_recording(short=args.short)

    steps = [
        ("Import spikeinterface", check_import_si),
        ("Import spikeinterface.full", check_import_si_full),
        ("Run tridesclous2", run_tridesclous2),
    ]
    if not args.skip_kilosort4:
        steps.append(("Run kilosort4", run_kilosort4))

    # backwards logic because default is True for end-user
    if args.ci:
        steps.append(("Open spikeinterface-gui", open_sigui))

    for label, func in steps:
        try:
            func()
            done = "...OK"
        except Exception as err:
            done = f"...Fail, Error: {err}"
        print(label, done)

    if platform.system() == "Windows":
        pass
    else:
        _clean()
