from pathlib import Path

import pandas as pd

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
            rec, = read_nwb(file_path, load_recording=True, load_sorting=False, electrical_series_name=None)
            rec.annotate(bids_name=bids_name)
            
            probegroup = read_BIDS_probe(file_path.parent)

            # make maps between : channel_id	and contact_id
            # use _channels.tsv
            for probe in probegroup.probes:
                channels_file = file_path.parent / bids_name.replace('_ephys', '_channels.tsv')
                channels = pd.read_csv(channels_file, sep='\t')
                channel_ids = channels['channel_id'].values.astype('U')
                contact_ids = channels['contact_id'].values.astype('U')
                # contact_id > channel_id
                contact_id_to_channel_id = dict(zip(contact_ids, channel_ids))
                
                # contact_id > channel_index
                contact_id_to_channel_index = dict()
                rec_chan_ids = list(rec.channel_ids.astype('U'))
                for contact_id, channel_id in contact_id_to_channel_id.items():
                    channel_index = rec_chan_ids.index(channel_id)
                    contact_id_to_channel_index[contact_id] = channel_index
                
                # vector of channel indices
                # needed for probe wiring
                device_channel_indices = []
                for contact_id in probe.contact_ids:
                    chan_index = contact_id_to_channel_index[contact_id]
                    device_channel_indices.append(chan_index)
                probe.set_device_channel_indices(device_channel_indices)

            rec = rec.set_probegroup(probegroup)
            recordings.append(rec)

    return recordings
