from __future__ import annotations

from pathlib import Path

import numpy as np

import probeinterface

from .nwbextractors import read_nwb
from .neoextractors import read_nix


def read_bids(folder_path):
    """Load a BIDS folder of data into extractor objects.

    The following files are considered:

      * _channels.tsv
      * _contacts.tsv
      * _ephys.nwb
      * _probes.tsv

    Parameters
    ----------
    folder_path : str or Path
        Path to the BIDS folder.

    Returns
    -------
    extractors : list of extractors
        The loaded data, with attached Probes.
    """

    folder_path = Path(folder_path)

    recordings = []
    for file_path in folder_path.iterdir():
        bids_name = file_path.stem

        if file_path.suffix == ".nwb":
            (rec,) = read_nwb(file_path, load_recording=True, load_sorting=False, electrical_series_name=None)
            rec.annotate(bids_name=bids_name)
            rec.extra_requirements.extend("pandas")
            probegroup = _read_probe_group(file_path.parent, bids_name, rec.channel_ids)
            rec = rec.set_probegroup(probegroup)
            recordings.append(rec)

        elif file_path.suffix == ".nix":
            import neo

            neo_reader = neo.rawio.NIXRawIO(file_path)
            neo_reader.parse_header()
            stream_ids = neo_reader.header["signal_streams"]["id"]

            for stream_id in stream_ids:
                rec = read_nix(file_path, stream_id=stream_id)
                rec.extra_requirements.extend("pandas")
                probegroup = _read_probe_group(file_path.parent, bids_name, rec.channel_ids)
                rec = rec.set_probegroup(probegroup)
                recordings.append(rec)

    return recordings


def _read_probe_group(folder, bids_name, recording_channel_ids):
    probegroup = probeinterface.read_BIDS_probe(folder)

    # make maps between : channel_id and contact_id using _channels.tsv
    import pandas as pd

    for probe in probegroup.probes:
        channels_file = folder / bids_name.replace("_ephys", "_channels.tsv")
        channels = pd.read_csv(channels_file, sep="\t", dtype="str")
        # channel_ids are unique
        channel_ids = channels["channel_id"].values.astype("U")
        # contact ids are not unique
        # a single contact can be associated with multiple channels, contact_ids can be n/a
        channels["contact_id"][channels["contact_id"].isnull()] = "unconnected"
        contact_ids = channels["contact_id"].values.astype("U")

        # extracting information of requested channels
        keep = np.isin(channel_ids, recording_channel_ids)
        channel_ids = channel_ids[keep]
        contact_ids = contact_ids[keep]

        rec_chan_ids = list(recording_channel_ids.astype("U"))

        # contact_id > channel_id
        # this overwrites if there's multiple contact_ids = unconnected
        contact_id_to_channel_id = dict(zip(contact_ids, channel_ids))
        # remove unconnected contact entry
        contact_id_to_channel_id.pop("unconnected", None)
        # contact_id > channel_index within recording
        contact_id_to_channel_index = {
            con_id: rec_chan_ids.index(chan_id) for con_id, chan_id in contact_id_to_channel_id.items()
        }

        # vector of channel indices within recording ordered by probe contact_ids
        # needed for probe wiring
        device_channel_indices = []
        for contact_id in probe.contact_ids:
            if contact_id in contact_id_to_channel_index:
                device_channel_indices.append(contact_id_to_channel_index[contact_id])
            else:
                # using -1 for unconnected channels
                device_channel_indices.append(-1)
        probe.set_device_channel_indices(device_channel_indices)

    return probegroup
