from pathlib import Path
import platform
import os
import shutil
import argparse

def check_import_si():
    import spikeinterface as si

def check_import_si_full():
    import spikeinterface.full as si


def _create_recording():
    import spikeinterface.full as si
    rec, sorting = si.toy_example(num_segments=1, duration=200, seed=1, num_channels=16, num_columns=2)
    rec.save(folder='./toy_example_recording')


def _run_one_sorter_and_analyzer(sorter_name):
    job_kwargs = dict(n_jobs=-1, progress_bar=True, chunk_duration="1s")
    import spikeinterface.full as si
    recording = si.load_extractor('./toy_example_recording')
    sorting = si.run_sorter(sorter_name, recording, output_folder=f'./sorter_with_{sorter_name}', verbose=False)

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


def run_tridesclous():
    _run_one_sorter_and_analyzer('tridesclous')

def run_tridesclous2():
    _run_one_sorter_and_analyzer('tridesclous2')



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
        "./sorter_with_tridesclous",
        "./analyzer_with_tridesclous",
        "./sorter_with_tridesclous2",
        "./analyzer_with_tridesclous2",
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
        ('Run tridesclous', run_tridesclous),
        ('Run tridesclous2', run_tridesclous2),
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
