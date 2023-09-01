import spikeinterface as si
from pathlib import Path
from spikeinterface import BinaryRecordingExtractor
from spikeinterface import load_extractor
import sys
import importlib.util
import runpy
import numpy as np


class KilosortTempWhExtractor(BinaryRecordingExtractor):
    def __init__(self, output_path: Path) -> None:
        self.sorter_output_path = output_path / "sorter_output"
        # TODO: store opts e.g. ntb, Nbatch etc here.

        params = runpy.run_path(self.sorter_output_path / "params.py")

        file_paths = Path(self.sorter_output_path) / "temp_wh.dat"  # some assert
        sampling_frequency = params["sample_rate"]
        dtype = params["dtype"]
        assert dtype == "int16"

        channel_map = np.load(self.sorter_output_path / "channel_map.npy")
        if channel_map.ndim == 2:  # kilosort > 2
            channel_indices = channel_map.ravel()  # TODO: check multiple shanks
        else:
            assert channel_map.ndim == 1
            channel_indices = channel_map

        num_channels = channel_indices.size

        original_recording = load_extractor(output_path / "spikeinterface_recording.json", base_folder=output_path)
        original_channel_ids = original_recording.get_channel_ids()

        if original_recording.has_scaled():
            gain_to_uV = original_recording.get_property("gain_to_uV")[
                channel_indices
            ]  # TODO: check this assumption - does KS change the scale / offset? can check by performing no processing...
            offset_to_uV = original_recording.get_property("offset_to_uV")[channel_indices]
        else:
            gain_to_uV = None
            offset_to_uV = None

        self.original_recording_num_samples = original_recording.get_num_samples()
        new_channel_ids = original_channel_ids[channel_indices]  # TODO: check whether this will erroneously re-order
        # is_filtered = original_recording.is_filtered or ## params was filtering run

        super(KilosortTempWhExtractor, self).__init__(
            file_paths,
            sampling_frequency,
            dtype,
            num_channels=num_channels,
            t_starts=None,
            channel_ids=new_channel_ids,
            time_axis=0,
            file_offset=0,
            gain_to_uV=gain_to_uV,
            offset_to_uV=offset_to_uV,
            is_filtered=None,
            num_chan=None,
        )

        # TODO: check, there must be a probe if sorting was run?
        # change the wiring of the probe
        # TODO: check this carefully, might be completely wrong

        contact_vector = original_recording.get_property("contact_vector")
        contact_vector = contact_vector[channel_indices]
        #      if contact_vector is not None:
        contact_vector["device_channel_indices"] = np.arange(len(new_channel_ids), dtype="int64")
        self.set_property("contact_vector", contact_vector)

        data2 = original_recording.get_traces(start_frame=0, end_frame=75000)
        breakpoint()


#        original_probe = original_recording.get_probe()
# self.set_probe(original_probe)

# 1) figure out metadata and casting for WaveForm Extractor
# 2) check lazyness etc.

# zero padding can just be kept. Check it plays nice with WaveformExtractor...

# TODO: add provenance


#    def get_num_samples(self):
#       """ ignore Kilosort's zero-padding """
#      return self.original_recording.get_num_samples()

path_ = Path(r"X:\neuroinformatics\scratch\jziminski\ephys\code\sorter_output")
data = KilosortTempWhExtractor(path_)

breakpoint()
