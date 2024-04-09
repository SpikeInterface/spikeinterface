import pytest

from spikeinterface.core import generate_ground_truth_recording, create_sorting_analyzer


from spikeinterface import Templates
from spikeinterface.core import (
    get_template_amplitudes,
    get_template_extremum_channel,
    get_template_extremum_channel_peak_shift,
    get_template_extremum_amplitude,
)


def get_sorting_analyzer():
    recording, sorting = generate_ground_truth_recording(
        durations=[10.0, 5.0],
        sampling_frequency=10_000.0,
        num_channels=4,
        num_units=10,
        noise_kwargs=dict(noise_levels=5.0, strategy="tile_pregenerated"),
        seed=2205,
    )
    recording.annotate(is_filtered=True)
    recording.set_channel_groups([0, 0, 1, 1])
    sorting.set_property("group", [0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    sorting_analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparse=False)
    sorting_analyzer.compute("random_spikes")
    sorting_analyzer.compute("templates")

    return sorting_analyzer


@pytest.fixture(scope="module")
def sorting_analyzer():
    return get_sorting_analyzer()


def _get_templates_object_from_sorting_analyzer(sorting_analyzer):
    ext = sorting_analyzer.get_extension("templates")
    templates = Templates(
        templates_array=ext.data["average"],
        sampling_frequency=sorting_analyzer.sampling_frequency,
        nbefore=ext.nbefore,
        # this is dense
        sparsity_mask=None,
        channel_ids=sorting_analyzer.channel_ids,
        unit_ids=sorting_analyzer.unit_ids,
    )
    return templates


def test_get_template_amplitudes(sorting_analyzer):
    peak_values = get_template_amplitudes(sorting_analyzer)
    print(peak_values)
    templates = _get_templates_object_from_sorting_analyzer(sorting_analyzer)
    peak_values = get_template_amplitudes(templates)
    print(peak_values)


def test_get_template_extremum_channel(sorting_analyzer):
    extremum_channels_ids = get_template_extremum_channel(sorting_analyzer, peak_sign="both")
    print(extremum_channels_ids)
    templates = _get_templates_object_from_sorting_analyzer(sorting_analyzer)
    extremum_channels_ids = get_template_extremum_channel(templates, peak_sign="both")
    print(extremum_channels_ids)


def test_get_template_extremum_channel_peak_shift(sorting_analyzer):
    shifts = get_template_extremum_channel_peak_shift(sorting_analyzer, peak_sign="neg")
    print(shifts)
    templates = _get_templates_object_from_sorting_analyzer(sorting_analyzer)
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


def test_get_template_extremum_amplitude(sorting_analyzer):

    extremum_channels_ids = get_template_extremum_amplitude(sorting_analyzer, peak_sign="both")
    print(extremum_channels_ids)

    templates = _get_templates_object_from_sorting_analyzer(sorting_analyzer)
    extremum_channels_ids = get_template_extremum_amplitude(templates, peak_sign="both")


if __name__ == "__main__":
    # setup_module()

    sorting_analyzer = get_sorting_analyzer()
    print(sorting_analyzer)

    test_get_template_amplitudes(sorting_analyzer)
    test_get_template_extremum_channel(sorting_analyzer)
    test_get_template_extremum_channel_peak_shift(sorting_analyzer)
    test_get_template_extremum_amplitude(sorting_analyzer)
