import pytest

from spikeinterface.core import generate_ground_truth_recording, start_sorting_result


from spikeinterface import Templates
from spikeinterface.core import (
    get_template_amplitudes,
    get_template_extremum_channel,
    get_template_extremum_channel_peak_shift,
    get_template_extremum_amplitude,
)


def get_sorting_result():
    recording, sorting = generate_ground_truth_recording(
        durations=[10.0, 5.0],
        sampling_frequency=10_000.0,
        num_channels=4,
        num_units=10,
        noise_kwargs=dict(noise_level=5.0, strategy="tile_pregenerated"),
        seed=2205,
    )
    recording.annotate(is_filtered=True)
    recording.set_channel_groups([0, 0, 1, 1])
    sorting.set_property("group", [0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    sorting_result = start_sorting_result(sorting, recording, format="memory", sparse=False)
    sorting_result.select_random_spikes()
    sorting_result.compute("fast_templates")

    return sorting_result


@pytest.fixture(scope="module")
def sorting_result():
    return get_sorting_result()


def _get_templates_object_from_sorting_result(sorting_result):
    ext = sorting_result.get_extension("fast_templates")
    templates = Templates(
        templates_array=ext.data["average"],
        sampling_frequency=sorting_result.sampling_frequency,
        nbefore=ext.nbefore,
        # this is dense
        sparsity_mask=None,
        channel_ids=sorting_result.channel_ids,
        unit_ids=sorting_result.unit_ids,
    )
    return templates


def test_get_template_amplitudes(sorting_result):
    peak_values = get_template_amplitudes(sorting_result)
    print(peak_values)
    templates = _get_templates_object_from_sorting_result(sorting_result)
    peak_values = get_template_amplitudes(templates)
    print(peak_values)


def test_get_template_extremum_channel(sorting_result):
    extremum_channels_ids = get_template_extremum_channel(sorting_result, peak_sign="both")
    print(extremum_channels_ids)
    templates = _get_templates_object_from_sorting_result(sorting_result)
    extremum_channels_ids = get_template_extremum_channel(templates, peak_sign="both")
    print(extremum_channels_ids)


def test_get_template_extremum_channel_peak_shift(sorting_result):
    shifts = get_template_extremum_channel_peak_shift(sorting_result, peak_sign="neg")
    print(shifts)
    templates = _get_templates_object_from_sorting_result(sorting_result)
    shifts = get_template_extremum_channel_peak_shift(templates, peak_sign="neg")

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


def test_get_template_extremum_amplitude(sorting_result):

    extremum_channels_ids = get_template_extremum_amplitude(sorting_result, peak_sign="both")
    print(extremum_channels_ids)

    templates = _get_templates_object_from_sorting_result(sorting_result)
    extremum_channels_ids = get_template_extremum_amplitude(templates, peak_sign="both")


if __name__ == "__main__":
    # setup_module()

    sorting_result = get_sorting_result()
    print(sorting_result)

    test_get_template_amplitudes(sorting_result)
    test_get_template_extremum_channel(sorting_result)
    test_get_template_extremum_channel_peak_shift(sorting_result)
    test_get_template_extremum_amplitude(sorting_result)
