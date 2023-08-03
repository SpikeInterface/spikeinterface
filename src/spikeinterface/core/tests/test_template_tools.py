import pytest
import shutil
from pathlib import Path

from spikeinterface import WaveformExtractor, load_extractor, extract_waveforms, generate_recording, generate_sorting

from spikeinterface.core import (
    get_template_amplitudes,
    get_template_extremum_channel,
    get_template_extremum_channel_peak_shift,
    get_template_extremum_amplitude,
)


if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "core"
else:
    cache_folder = Path("cache_folder") / "core"


def setup_module():
    for folder_name in ("toy_rec", "toy_sort", "toy_waveforms", "toy_waveforms_1"):
        if (cache_folder / folder_name).is_dir():
            shutil.rmtree(cache_folder / folder_name)

    durations = [10.0, 5.0]
    recording = generate_recording(durations=durations, num_channels=4)
    sorting = generate_sorting(durations=durations, num_units=10)

    recording.annotate(is_filtered=True)
    recording.set_channel_groups([0, 0, 1, 1])
    recording = recording.save(folder=cache_folder / "toy_rec")
    sorting.set_property("group", [0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    sorting = sorting.save(folder=cache_folder / "toy_sort")

    we = WaveformExtractor.create(recording, sorting, cache_folder / "toy_waveforms")
    we.set_params(ms_before=3.0, ms_after=4.0, max_spikes_per_unit=500)
    we.run_extract_waveforms(n_jobs=1, chunk_size=30000)


def test_get_template_amplitudes():
    we = WaveformExtractor.load(cache_folder / "toy_waveforms")
    peak_values = get_template_amplitudes(we)
    print(peak_values)


def test_get_template_extremum_channel():
    we = WaveformExtractor.load(cache_folder / "toy_waveforms")
    extremum_channels_ids = get_template_extremum_channel(we, peak_sign="both")
    print(extremum_channels_ids)


def test_get_template_extremum_channel_peak_shift():
    we = WaveformExtractor.load(cache_folder / "toy_waveforms")
    shifts = get_template_extremum_channel_peak_shift(we, peak_sign="neg")
    print(shifts)

    # DEBUG
    # import matplotlib.pyplot as plt
    # extremum_channels_ids = get_template_extremum_channel(we, peak_sign='both')
    # for unit_id in we.unit_ids:
    #     chan_id = extremum_channels_ids[unit_id]
    #     chan_ind = we.recording.id_to_index(chan_id)
    #     template = we.get_template(unit_id)
    #     shift = shifts[unit_id]
    #     fig, ax = plt.subplots()
    #     template_chan = template[:, chan_ind]
    #     ax.plot(template_chan)
    #     ax.axvline(we.nbefore, color='grey')
    #     ax.axvline(we.nbefore + shift, color='red')
    #     plt.show()


def test_get_template_extremum_amplitude():
    we = WaveformExtractor.load(cache_folder / "toy_waveforms")

    extremum_channels_ids = get_template_extremum_amplitude(we, peak_sign="both")
    print(extremum_channels_ids)


if __name__ == "__main__":
    setup_module()

    test_get_template_amplitudes()
    test_get_template_extremum_channel()
    test_get_template_extremum_channel_peak_shift()
    test_get_template_extremum_amplitude()
    test_get_template_channel_sparsity()
