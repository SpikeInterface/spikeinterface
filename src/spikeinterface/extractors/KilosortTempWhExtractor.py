from typing import Dict
import spikeinterface as si
from pathlib import Path
from spikeinterface import BinaryRecordingExtractor
from spikeinterface import load_extractor
import sys
import importlib.util
import runpy
import numpy as np
from spikeinterface import WaveformExtractor, extract_waveforms


class KilosortTempWhExtractor(BinaryRecordingExtractor):
    def __init__(self, output_path: Path) -> None:
        # TODO: store opts e.g. ntb, Nbatch etc here.

        if self.has_spikeinterface(output_path):
            self.sorter_output_path = output_path / "sorter_output"

            channel_indices = self.get_channel_indices()

            original_recording = load_extractor(output_path / "spikeinterface_recording.json", base_folder=output_path)
            channel_ids = original_recording.get_channel_ids()

            # TODO: check this assumption - does KS change the scale / offset? can check
            #  by performing no processing...
            if original_recording.has_scaled():
                gain_to_uV = original_recording.get_property("gain_to_uV")[channel_indices]
                offset_to_uV = original_recording.get_property("offset_to_uV")[channel_indices]
            else:
                gain_to_uV = None
                offset_to_uV = None

            channel_locations = original_recording.get_channel_locations()

            # TODO: I think this is safe to assume as if the recording was
            # sorted then it must have a probe attached.
            probe = original_recording.get_probe()

        elif self.has_valid_sorter_output(output_path):
            self.sorter_output_path = output_path

            channel_indices = self.get_channel_indices()
            channel_ids = np.array(channel_indices, dtype=str)

            gain_to_uV = None
            offset_to_uV = None

            channel_locations = np.load(self.sorter_output_path / "channel_positions.npy")
            probe = None

        else:
            raise ValueError("")

        params = self.load_and_check_kilosort_params_file()
        temp_wh_path = Path(self.sorter_output_path) / "temp_wh.dat"

        new_channel_ids = channel_ids[channel_indices]
        new_channel_locations = channel_locations[channel_indices]

        # TODO: need to adjust probe?
        # TODO: check whether this will erroneously re-order
        # is_filtered = original_recording.is_filtered or ## params was filtering run
        super(KilosortTempWhExtractor, self).__init__(
            temp_wh_path,
            params["sample_rate"],
            params["dtype"],
            num_channels=channel_indices.size,
            t_starts=None,
            channel_ids=new_channel_ids,
            time_axis=0,
            file_offset=0,
            gain_to_uV=gain_to_uV,
            offset_to_uV=offset_to_uV,
            is_filtered=None,
            num_chan=None,
        )
        self.set_channel_locations(new_channel_locations)

    #   if probe:
    #      self.set_probe(probe)

    def get_channel_indices(self):
        """"""
        channel_map = np.load(self.sorter_output_path / "channel_map.npy")

        if channel_map.ndim == 2:
            channel_indices = channel_map.ravel()
        else:
            assert channel_map.ndim == 1
            channel_indices = channel_map

        return channel_indices

    def has_spikeinterface(self, path_: Path) -> bool:
        """ """
        sorter_output = path_ / "sorter_output"

        if not (path_ / "spikeinterface_recording.json").is_file() or not sorter_output.is_dir():
            return False

        return self.has_valid_sorter_output(sorter_output)

    def has_valid_sorter_output(self, path_: Path) -> bool:
        """ """
        required_files = ["temp_wh.dat", "channel_map.npy", "channel_positions.npy"]

        for filename in required_files:
            if not (path_ / filename).is_file():
                print(f"The file {filename} cannot be out in {path_}")
                return False
        return True

    def load_and_check_kilosort_params_file(self) -> Dict:
        """ """
        params = runpy.run_path(self.sorter_output_path / "params.py")

        if params["dtype"] != "int16":
            raise ValueError("The dtype in kilosort's params.py is expected" "to be `int16`.")

        return params


#        original_probe = original_recording.get_probe()
# self.set_probe(original_probe) TODO: do we need to adjust the probe? what about contact positions?

# 1) figure out metadata and casting for WaveForm Extractor
# 2) check lazyness etc.

# zero padding can just be kept. Check it plays nice with WaveformExtractor...

# TODO: add provenance
# TODO: what to do about all those zeros?

#    def get_num_samples(self):
#       """ ignore Kilosort's zero-padding """
#      return self.original_recording.get_num_samples()

# TODO: check, there must be a probe if sorting was run?
# change the wiring of the probe
# TODO: check this carefully, might be completely wrong
#      if contact_vector is not None:

# if channel_map.ndim == 2:  # kilosort > 2
#    channel_indices = channel_map.ravel()  # TODO: check multiple shanks

# self.set_channel_locations(new_channel_locations)  # TOOD: check against slice_channels

#             is_filtered=None,  # TODO: need to get from KS provenence?

# In general, do we store the full channel map in channel contacts or do we
# only save the new subset? My guess is subset for contact_positions, but full probe
# for probe. Check against slice_channels.
#             self.set_probe(probe)  # TODO: what does this mean for missing channels?

#         if channel_map.ndim == 2:  # kilosort > 2
# does kilosort > 2 store shanks differently?             channel_indices = channel_map.ravel()

path_ = Path(r"X:\neuroinformatics\scratch\jziminski\ephys\code\sorter_output")  # sorter_output
recording_new = KilosortTempWhExtractor(path_)

from spikeinterface import extractors
from spikeinterface import postprocessing

sorting = extractors.read_kilosort(
    folder_path=(path_ / "sorter_output").as_posix(),
    keep_good_only=False,
)

recording_old = load_extractor(path_ / "spikeinterface_recording.json", base_folder=path_)
folder_old = Path(r"X:\neuroinformatics\scratch\jziminski\ephys\code\waveform_folder_old")
waveforms_old = extract_waveforms(
    recording_old,
    sorting,
    folder_old,
    ms_before=1.5,
    ms_after=2,
    max_spikes_per_unit=500,
    allow_unfiltered=True,
    load_if_exists=True,
)  # match kilosort

folder_new = Path(r"X:\neuroinformatics\scratch\jziminski\ephys\code\waveform_folder_new")
waveforms_new = extract_waveforms(
    recording_new,
    sorting,
    folder_new,
    ms_before=1.5,
    ms_after=2,
    max_spikes_per_unit=500,
    allow_unfiltered=True,
    load_if_exists=True,
)  # match kilosort

breakpoint()

if False:
    import matplotlib.pyplot as plt

    plt.plot(kilosort_waveform)
    plt.show()

    plt.plot(test_waveform)
    plt.show()


# if folder.is_dir():
#   import shutil
#   shutil.rmtree(folder)

# run sorting without kilosort preprocessing
# then, the `temp_wh.dat` should match exactly the original file!
# I think this is a solid way to test. It is not possible to test against
#


if False:
    original_recording = load_extractor(path_ / "spikeinterface_recording.json", base_folder=path_)
    waveforms_old = extract_waveforms(
        original_recording,
        sorting,
        folder,
        ms_before=1.5,
        ms_after=2.0,
        max_spikes_per_unit=500,
        allow_unfiltered=True,
        load_if_exists=True,
    )

    original_recording = load_extractor(path_ / "spikeinterface_recording.json", base_folder=path_)

    # TODO: unit locations don't match kilosort very well, at least in the 1-spike case.
    # But, this could be due to windowing and should average out over many spikes
    breakpoint()

    unit_locations_old = postprocessing.compute_unit_locations(waveforms, method="center_of_mass", outputs="by_unit")
    unit_locations_pandas = pd.DataFrame.from_dict(unit_locations, orient="index", columns=["x", "y"])
    unit_locations_pandas.to_csv(unit_locations_path)

    utils.message_user(f"Unit locations saved to {unit_locations_path}")

    print(we)
