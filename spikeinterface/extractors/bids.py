from pathlib import Path

from .nwbextractors import read_nwb

from probeinterface import read_BIDS_probe

def read_bids_folder(folder_path):
    """
    This read an entire BIDS folder and return a list of recording with
    there attached Probe.
    
    theses files are considered:
      * _channels.tsv
      * _contacts.tsv
      * _ephys.nwb
      * _probes.tsv
    """
    
    folder_path = Path(folder_path)
    
    recordings = []
    for file_path in folder_path.iterdir():
        #~ print(file_path)
        
        bids_name = file_path.stem
        
        if file_path.suffix == '.nwb':
            print('bids_name', bids_name, type)
            rec, = read_nwb(file_path, load_recording=True, load_sorting=False, electrical_series_name=None)
            print(rec)
            rec.annotate(bids_name=bids_name)
            probe = read_BIDS_probe(file_path.parent)
            print(probe)
            
            recordings.append(rec)

    return recordings
    
    
