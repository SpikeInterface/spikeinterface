from spikeinterface.generation import split_sorting_by_amplitudes, split_sorting_by_times

from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from spikeinterface.core.generate import generate_ground_truth_recording


def test_split_by_times():
    rec, sorting = generate_ground_truth_recording()
    sa = create_sorting_analyzer(sorting, rec)
    new_sorting, splitted_pairs = split_sorting_by_times(sa)
    assert len(new_sorting.unit_ids) == len(sorting.unit_ids) + len(splitted_pairs)
    for pair in splitted_pairs:
        p1 = new_sorting.get_unit_spike_train(pair[0]).mean()
        p2 = new_sorting.get_unit_spike_train(pair[1]).mean()
        assert p1 < p2


def test_split_by_amplitudes():
    rec, sorting = generate_ground_truth_recording()
    sa = create_sorting_analyzer(sorting, rec)
    sa.compute(["random_spikes", "templates", "spike_amplitudes"])
    new_sorting, splitted_pairs = split_sorting_by_amplitudes(sa)
    assert len(new_sorting.unit_ids) == len(sorting.unit_ids) + len(splitted_pairs)


if __name__ == "__main__":
    test_split_by_times()
    test_split_by_amplitudes()
