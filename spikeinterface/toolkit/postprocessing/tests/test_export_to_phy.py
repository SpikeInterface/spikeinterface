import unittest
import shutil
from pathlib import Path

import pytest



from spikeinterface import extract_waveforms
from spikeinterface.extractors import toy_example
from spikeinterface.toolkit.postprocessing import export_to_phy


def test_export_to_phy():
    recording, sorting = toy_example(num_segments=1, num_units=10)
    recording = recording.save()
    sorting = sorting.save()
    
    
    waveform_folder = Path('waveforms')
    output_folder= Path('phy_output')
    
    for f in (waveform_folder, output_folder):
        if f.is_dir():
            shutil.rmtree(f)
    
    
    waveform_extractor = extract_waveforms(recording, sorting, waveform_folder)
    print(waveform_extractor)
    export_to_phy(recording, sorting, output_folder, waveform_extractor,
            compute_pc_features=False,
            compute_amplitudes=False)
    
    


if __name__ == '__main__':
    test_export_to_phy()
