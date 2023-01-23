from pathlib import Path
import platform
import os
import shutil


def check_import_si():
    import spikeinterface as si


def check_import_si_full():
    import spikeinterface.full as si


def _create_recording():
    import spikeinterface.full as si
    rec, sorting = si.toy_example(num_segments=1, duration=200, seed=1, num_channels=16, num_columns=2)
    rec.save(folder='./toy_example_recording')


def _run_one_sorter_and_exctract_wf(sorter_name):
    import spikeinterface.full as si
    rec = si.load_extractor('./toy_example_recording')
    sorting = si.run_sorter(sorter_name, rec, output_folder=f'{sorter_name}_output', verbose=False)
    si.extract_waveforms(rec, sorting, f'{sorter_name}_waveforms',
                         n_jobs=1, total_memory="10M", max_spikes_per_unit=500, return_scaled=False)


def run_tridesclous():
    _run_one_sorter_and_exctract_wf('tridesclous')


def run_spykingcircus():
    _run_one_sorter_and_exctract_wf('spykingcircus')


def run_herdingspikes():
    _run_one_sorter_and_exctract_wf('herdingspikes')


def open_sigui():
    import spikeinterface.full as si
    import spikeinterface_gui
    app = spikeinterface_gui.mkQApp()
    waveform_forlder = 'tridesclous_waveforms'
    we = si.WaveformExtractor.load_from_folder(waveform_forlder)
    pc = si.compute_principal_components(we, n_components=3, mode='by_channel_local', whiten=True, dtype='float32')
    win = spikeinterface_gui.MainWindow(we)
    win.show()
    app.exec_()


def export_to_phy():
    import spikeinterface.full as si
    we = si.WaveformExtractor.load_from_folder("tridesclous_waveforms")
    phy_folder = "./phy_example"
    si.export_to_phy(we, output_folder=phy_folder, verbose=False)


def open_phy():
    os.system("phy template-gui ./phy_example/params.py")


def _clean():
    # clean
    folders = [
        'toy_example_recording',
        "tridesclous_output", "tridesclous_waveforms",
        "spykingcircus_output", "spykingcircus_waveforms",
        "phy_example"
    ]
    for folder in folders:
        if Path(folder).exists():
            shutil.rmtree(folder)


if __name__ == '__main__':

    _clean()
    _create_recording()

    steps = [
        ('Import spikeinterface', check_import_si),
        ('Import spikeinterface.full', check_import_si_full),
        ('Run tridesclous', run_tridesclous),
        ('Open spikeinterface-gui', open_sigui),
        ('Export to phy', export_to_phy),
        # phy is removed from the env because it force a pip install PyQt5
        # which break the conda env
        # ('Open phy', open_phy),
    ]

    if platform.system() == "Windows":
        pass
        # steps.insert(3, ('Run spykingcircus', run_spykingcircus))
    elif platform.system() == "Darwin":
        steps.insert(3, ('Run herdingspikes', run_herdingspikes))
    else:
        steps.insert(3, ('Run spykingcircus', run_spykingcircus))
        steps.insert(4, ('Run herdingspikes', run_herdingspikes))

    for label, func in steps:
        try:
            func()
            done = '...OK'
        except:
            done = '...Fail'
        print(label, done)

    _clean()
