from pathlib import Path
import shutil


def check_import_si():
    import spikeinterface as si

def check_import_si_full():
    import spikeinterface.full as si

def _create_recording():
    import spikeinterface.full as si
    rec, sorting = si.toy_example(num_segments=1, duration=200)
    rec.save(folder = './toy_example_recording')

def _run_one_sorter_and_exctract_wf(sorter_name):
    import spikeinterface.full as si
    rec = si.load_extractor('./toy_example_recording')
    sorting = si.run_sorter(sorter_name, rec, output_folder=f'{sorter_name}_output')
    si.extract_waveforms(rec, sorting, f'{sorter_name}_waveforms',
        n_jobs=1, total_memory="10M", max_spikes_per_unit=500, return_scaled=False)

def run_tridesclous():
    _run_one_sorter_and_exctract_wf('tridesclous')

def run_spykingcircus():
    _run_one_sorter_and_exctract_wf('spykingcircus')

def open_sigui():
    import spikeinterface as si
    import spikeinterface_gui
    app = spikeinterface_gui.mkQApp() 
    waveform_forlder = 'tridesclous_waveforms'
    we = si.WaveformExtractor.load_from_folder(waveform_forlder)
    win = spikeinterface_gui.MainWindow(we)
    win.show()
    app.exec_()


if __name__ == '__main__':

    # clean
    folders = [
        'toy_example_recording',
        "tridesclous_output", "tridesclous_waveforms",
        "spykingcircus_output", "spykingcircus_waveforms"
    ]
    for folder in folders:
        if Path(folder).exists():
            shutil.rmtree(folder)
    
    _create_recording()
    
    # run some checks
    steps = [
        ('Import spikeinterface', check_import_si),
        ('Import spikeinterface.full', check_import_si_full),
        ('Run tridesclous', run_tridesclous),
        ('Import spyeking circus', run_spykingcircus),
        ('Open spikeinterface-gui', open_sigui),
    ]
    
    for label, func in steps:
        try :
            func()
            done = '...OK'
        except:
            done = '...Fail'
        print(label, done)
        