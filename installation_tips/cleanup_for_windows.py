from pathlib import Path
import shutil


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