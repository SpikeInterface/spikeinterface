from pathlib import Path
import platform
import os
import shutil
import argparse


job_kwargs = dict(n_jobs=-1, progress_bar=True, chunk_duration="1s")

def check_import_si():
    import spikeinterface as si

def check_import_si_full():
    import spikeinterface.full as si


def _create_recording():
    import spikeinterface.full as si
    rec, sorting = si.generate_ground_truth_recording(
        durations=[200.],
        sampling_frequency=30_000.,
        num_channels=16,
        num_units=10,
        seed=2205
    )
    rec.save(folder='./toy_example_recording', **job_kwargs)


def _run_one_sorter_and_analyzer(sorter_name):
    import spikeinterface.full as si
    recording = si.load_extractor('./toy_example_recording')
    sorting = si.run_sorter(sorter_name, recording, folder=f'./sorter_with_{sorter_name}', verbose=False)

    sorting_analyzer = si.create_sorting_analyzer(sorting, recording,
                                                format="binary_folder", folder=f"./analyzer_with_{sorter_name}",
                                                **job_kwargs)
    sorting_analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
    sorting_analyzer.compute("waveforms", **job_kwargs)
    sorting_analyzer.compute("templates")
    sorting_analyzer.compute("noise_levels")
    sorting_analyzer.compute("unit_locations", method="monopolar_triangulation")
    sorting_analyzer.compute("correlograms", window_ms=100, bin_ms=5.)
    sorting_analyzer.compute("principal_components", n_components=3, mode='by_channel_global', whiten=True, **job_kwargs)
    sorting_analyzer.compute("quality_metrics", metric_names=["snr", "firing_rate"])


def run_tridesclous2():
    _run_one_sorter_and_analyzer('tridesclous2')

def run_kilosort4():
    _run_one_sorter_and_analyzer('kilosort4')



def open_sigui():
    import spikeinterface.full as si
    import spikeinterface_gui
    app = spikeinterface_gui.mkQApp()

    sorter_name = "tridesclous2"
    folder = f"./analyzer_with_{sorter_name}"
    analyzer = si.load_sorting_analyzer(folder)

    win = spikeinterface_gui.MainWindow(analyzer)
    win.show()
    app.exec_()

def export_to_phy():
    import spikeinterface.full as si
    sorter_name = "tridesclous2"
    folder = f"./analyzer_with_{sorter_name}"
    analyzer = si.load_sorting_analyzer(folder)

    phy_folder = "./phy_example"
    si.export_to_phy(analyzer, output_folder=phy_folder, verbose=False)


def open_phy():
    os.system("phy template-gui ./phy_example/params.py")


def _clean():
    # clean
    folders = [
        "./toy_example_recording",
        "./sorter_with_tridesclous2",
        "./analyzer_with_tridesclous2",
        "./sorter_with_kilosort4",
        "./analyzer_with_kilosort4",
        "./phy_example"
    ]
    for folder in folders:
        if Path(folder).exists():
            shutil.rmtree(folder)

parser = argparse.ArgumentParser()
# add ci flag so that gui will not be used in ci
# end user can ignore
parser.add_argument('--ci', action='store_false')

if __name__ == '__main__':

    args = parser.parse_args()

    _clean()
    _create_recording()

    steps = [
        ('Import spikeinterface', check_import_si),
        ('Import spikeinterface.full', check_import_si_full),
        ('Run tridesclous2', run_tridesclous2),
        ('Run kilosort4', run_kilosort4),
        ]

    # backwards logic because default is True for end-user
    if args.ci:
        steps.append(('Open spikeinterface-gui', open_sigui))

    steps.append(('Export to phy', export_to_phy)),
        # phy is removed from the env because it force a pip install PyQt5
        # which break the conda env
        # ('Open phy', open_phy),

    # if platform.system() == "Windows":
    #     pass
    # elif platform.system() == "Darwin":
    #     pass
    # else:
    #     pass

    for label, func in steps:
        try:
            func()
            done = '...OK'
        except Exception as err:
            done = f'...Fail, Error: {err}'
        print(label, done)

    if platform.system() == "Windows":
        pass
    else:
        _clean()
