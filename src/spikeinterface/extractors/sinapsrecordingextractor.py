from pathlib import Path
import numpy as np

from probeinterface import get_probe

from ..core import BinaryRecordingExtractor, ChannelSliceRecording
from ..core.core_tools import define_function_from_class

class SinapsResearchPlatformRecordingExtractor(ChannelSliceRecording):
    extractor_name = "SinapsResearchPlatform"
    mode = "file"
    name = "sinaps_research_platform"

    def __init__(self, file_path, stream_name="filt"):
        from ..preprocessing import UnsignedToSignedRecording

        file_path = Path(file_path)
        meta_file = file_path.parent / f"metadata_{file_path.stem}.txt"
        meta = parse_sinaps_meta(meta_file)

        num_aux_channels = meta["nbHWAux"] + meta["numberUserAUX"]
        num_total_channels = 2 * meta["nbElectrodes"] + num_aux_channels
        num_electrodes = meta["nbElectrodes"]
        sampling_frequency = meta["samplingFreq"]

        probe_type = meta['probeType']
        # channel_locations = meta["electrodePhysicalPosition"] # will be depricated soon by Sam, switching to probeinterface
        num_shanks = meta["nbShanks"]
        num_electrodes_per_shank = meta["nbElectrodesShank"]
        num_bits = int(np.log2(meta["nbADCLevels"]))

        # channel_groups = []
        # for i in range(num_shanks):
        #     channel_groups.extend([i] * num_electrodes_per_shank)

        gain_ephys = meta["voltageConverter"]
        gain_aux = meta["voltageAUXConverter"]

        recording = BinaryRecordingExtractor(
            file_path, sampling_frequency, dtype="uint16", num_channels=num_total_channels
        )
        recording = UnsignedToSignedRecording(recording, bit_depth=num_bits)

        if stream_name == "raw":
            channel_slice = recording.channel_ids[:num_electrodes]
            renamed_channels = np.arange(num_electrodes)
            # locations = channel_locations
            # groups = channel_groups
            gain = gain_ephys
        elif stream_name == "filt":
            channel_slice = recording.channel_ids[num_electrodes : 2 * num_electrodes]
            renamed_channels = np.arange(num_electrodes)
            # locations = channel_locations
            # groups = channel_groups
            gain = gain_ephys
        elif stream_name == "aux":
            channel_slice = recording.channel_ids[2 * num_electrodes :]
            hw_chans = meta["hwAUXChannelName"][1:-1].split(",")
            user_chans = meta["userAuxName"][1:-1].split(",")
            renamed_channels = hw_chans + user_chans
            # locations = None
            # groups = None
            gain = gain_aux
        else:
            raise ValueError("stream_name must be 'raw', 'filt', or 'aux'")

        ChannelSliceRecording.__init__(self, recording, channel_ids=channel_slice, renamed_channel_ids=renamed_channels)
        # if locations is not None:
            # self.set_channel_locations(locations)
        # if groups is not None:
            # self.set_channel_groups(groups)
        
        self.set_channel_gains(gain)
        self.set_channel_offsets(0)

        if (stream_name == 'filt') | (stream_name == 'raw'):
            if (probe_type == 'p1024s1NHP'):
                probe = get_probe(manufacturer='sinaps',
                                probe_name='SiNAPS-p1024s1NHP')
                # now wire the probe
                channel_indices = np.arange(1024)
                probe.set_device_channel_indices(channel_indices)
                self.set_probe(probe,in_place=True)
            else:
                raise ValueError(f"Unknown probe type: {probe_type}")

read_sinaps_research_platform = define_function_from_class(
    source_class=SinapsResearchPlatformRecordingExtractor, name="read_sinaps_research_platform"
)


def parse_sinaps_meta(meta_file):
    meta_dict = {}
    with open(meta_file) as f:
        lines = f.readlines()
        for l in lines:
            if "**" in l or "=" not in l:
                continue
            else:
                key, val = l.split("=")
                val = val.replace("\n", "")
                try:
                    val = int(val)
                except:
                    pass
                try:
                    val = eval(val)
                except:
                    pass
                meta_dict[key] = val
    return meta_dict
