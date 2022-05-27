from pathlib import Path

import numpy as np
import pandas as pd

from .nwbextractors import read_nwb
from .neoextractors import read_nix

from probeinterface import read_BIDS_probe

import neo


def read_bids(folder_path):
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
        # ~ print(file_path)

        bids_name = file_path.stem

        if file_path.suffix == '.nwb':
            rec, = read_nwb(file_path, load_recording=True, load_sorting=False, electrical_series_name=None)
            rec.annotate(bids_name=bids_name)
            rec.extra_requirements.extend('pandas')
            probegroup = _read_probe_group(file_path.parent, bids_name, rec.channel_ids)
            rec = rec.set_probegroup(probegroup)
            recordings.append(rec)

        elif file_path.suffix == '.nix':
            neo_reader = neo.rawio.NIXRawIO(file_path)
            neo_reader.parse_header()
            stream_ids = neo_reader.header['signal_streams']['id']

            for stream_id in stream_ids:
                rec = read_nix(file_path, stream_id=stream_id)
                rec.extra_requirements.extend('pandas')
                probegroup = _read_probe_group(file_path.parent, bids_name, rec.channel_ids)
                rec = rec.set_probegroup(probegroup)
                recordings.append(rec)

    return recordings


def _read_probe_group(folder, bids_name, recording_channel_ids):
    probegroup = read_BIDS_probe(folder)

    # make maps between : channel_id	and contact_id
    # use _channels.tsv
    for probe in probegroup.probes:
        channels_file = folder / bids_name.replace('_ephys', '_channels.tsv')
        channels = pd.read_csv(channels_file, sep='\t')
        channel_ids = channels['channel_id'].values.astype('U')
        channels['contact_id'][channels['contact_id'].isnull()] = -1
        contact_ids = channels['contact_id'].values.astype('int').astype('U')

        keep = np.in1d(channel_ids, recording_channel_ids)
        channel_ids = channel_ids[keep]
        contact_ids = contact_ids[keep]
        channel_indexes = []

        # contact_id > channel_id
        # contact_id_to_channel_id = dict(zip(contact_ids, channel_ids))

        # contact_id > channel_index
        contact_id_to_channel_index = dict()
        rec_chan_ids = list(recording_channel_ids.astype('U'))
        for contact_id, channel_id in zip(contact_ids, channel_ids):
            channel_index = rec_chan_ids.index(channel_id)
            channel_indexes.append(channel_index)

        # vector of channel indices
        # needed for probe wiring
        device_channel_indices = []
        for contact_id in probe.contact_ids:
            if contact_id in contact_ids:
                indexes = np.where(contact_ids == contact_id)[0]
                # TODO: this needs to updated once probeinterface supports multiple channels per contact
                if len(indexes) >= 1:
                    indexes = indexes[0]
                chan_index = channel_indexes[indexes]
                device_channel_indices.append(chan_index)
            else:
                device_channel_indices.append('-1')
        probe.set_device_channel_indices(device_channel_indices)

    return probegroup
