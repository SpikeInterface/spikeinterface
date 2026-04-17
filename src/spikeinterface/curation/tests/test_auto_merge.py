import pytest

import numpy as np

from spikeinterface.core import create_sorting_analyzer, NumpySorting
from spikeinterface.curation import compute_merge_unit_groups, auto_merge_units
from spikeinterface.generation import split_sorting_by_times


from spikeinterface.curation.tests.common import (
    extensions,
    make_sorting_analyzer,
    sorting_analyzer_for_curation,
    sorting_analyzer_with_splits,
    sorting_analyzer_multi_segment_for_curation,
)


@pytest.mark.parametrize(
    "preset", ["x_contaminations", "feature_neighbors", "temporal_splits", "similarity_correlograms", "slay", None]
)
def test_compute_merge_unit_groups(sorting_analyzer_with_splits, preset):

    sorting_analyzer, num_unit_splitted, other_ids = sorting_analyzer_with_splits

    if preset is not None:
        # do not resolve graph for checking true pairs
        merge_unit_groups, outs = compute_merge_unit_groups(
            sorting_analyzer,
            preset=preset,
            resolve_graph=False,
            # min_spikes=1000,
            # max_distance_um=150.0,
            # contamination_thresh=0.2,
            # corr_diff_thresh=0.16,
            # template_diff_thresh=0.25,
            # censored_period_ms=0.0,
            # refractory_period_ms=4.0,
            # sigma_smooth_ms=0.6,
            # adaptative_window_thresh=0.5,
            # firing_contamination_balance=1.5,
            extra_outputs=True,
        )
        if preset == "x_contaminations":
            assert len(merge_unit_groups) == num_unit_splitted
            for true_pair in other_ids.values():
                true_pair = tuple(true_pair)
                assert true_pair in merge_unit_groups
    else:
        # when preset is None you have to specify the steps
        with pytest.raises(ValueError):
            merge_unit_groups = compute_merge_unit_groups(sorting_analyzer, preset=preset)
        merge_unit_groups = compute_merge_unit_groups(
            sorting_analyzer,
            preset=preset,
            steps=["num_spikes", "snr", "remove_contaminated", "unit_locations"],
        )


@pytest.mark.parametrize(
    "preset", ["x_contaminations", "feature_neighbors", "temporal_splits", "similarity_correlograms", "slay"]
)
def test_compute_merge_unit_groups_multi_segment(sorting_analyzer_multi_segment_for_curation, preset):
    sorting_analyzer = sorting_analyzer_multi_segment_for_curation
    print(sorting_analyzer)

    merge_unit_groups = compute_merge_unit_groups(
        sorting_analyzer,
        preset=preset,
    )


def test_slay_discard_duplicated_spikes(sorting_analyzer_with_splits):
    sorting_analyzer, num_unit_splitted, split_ids = sorting_analyzer_with_splits

    # now for the split units, we add some duplicated spikes
    percent_duplicated = 0.7
    split_units = []
    for split in split_ids:
        split_units.extend(split_ids[split])

    # add unsplit spiketrains untouched
    new_spiketrains = {}
    for unit_id in sorting_analyzer.unit_ids:
        if unit_id in split_ids:
            continue
        new_spiketrains[unit_id] = sorting_analyzer.sorting.get_unit_spike_train(unit_id=unit_id)
    # ad duplicated spikes for split units
    for unit_id in split_ids:
        split_units = split_ids[unit_id]
        spiketrains0 = sorting_analyzer.sorting.get_unit_spike_train(unit_id=split_units[0])
        spiketrains1 = sorting_analyzer.sorting.get_unit_spike_train(unit_id=split_units[1])
        num_duplicated = int(percent_duplicated * min(len(spiketrains0), len(spiketrains1)))
        duplicated_spikes0 = np.random.choice(spiketrains0, size=num_duplicated, replace=False)
        new_spiketrain1 = np.sort(np.concatenate([spiketrains1, duplicated_spikes0]))

        new_spiketrains[split_units[0]] = spiketrains0
        new_spiketrains[split_units[1]] = new_spiketrain1

    sorting_duplicated = NumpySorting.from_unit_dict(
        new_spiketrains, sampling_frequency=sorting_analyzer.sampling_frequency
    )

    sorting_analyzer_duplicated = create_sorting_analyzer(
        sorting_duplicated, sorting_analyzer.recording, format="memory"
    )
    sorting_analyzer_duplicated.compute(extensions)

    # Without censor period the split should not be found because of duplicates.
    merges_no_censor_period, outs_no_censor_period = compute_merge_unit_groups(
        sorting_analyzer_duplicated,
        preset="slay",
        steps_params={"slay_score": {"censored_period_ms": 0.0}},
        extra_outputs=True,
    )
    merges_censor_period, outs_censor_period = compute_merge_unit_groups(
        sorting_analyzer_duplicated,
        preset="slay",
        steps_params={"slay_score": {"censored_period_ms": 0.5}},
        extra_outputs=True,
    )
    assert np.sum(outs_censor_period["slay_eta_ij"]) < np.sum(outs_no_censor_period["slay_eta_ij"])


def test_auto_merge_units(sorting_analyzer_for_curation):
    recording = sorting_analyzer_for_curation.recording
    new_sorting, _ = split_sorting_by_times(sorting_analyzer_for_curation)
    new_sorting_analyzer = create_sorting_analyzer(new_sorting, recording, format="memory")
    merged_analyzer = auto_merge_units(new_sorting_analyzer, presets="x_contaminations")
    assert len(merged_analyzer.unit_ids) < len(new_sorting_analyzer.unit_ids)

    step_merged_analyzer = auto_merge_units(
        new_sorting_analyzer,
        presets=None,
        steps=["num_spikes", "remove_contaminated", "unit_locations", "template_similarity", "quality_score"],
        steps_params={"num_spikes": {"min_spikes": 150}},
    )
    assert len(step_merged_analyzer.unit_ids) < len(new_sorting_analyzer.unit_ids)


def test_auto_merge_units_iterative(sorting_analyzer_for_curation):
    recording = sorting_analyzer_for_curation.recording
    new_sorting, _ = split_sorting_by_times(sorting_analyzer_for_curation)
    new_sorting_analyzer = create_sorting_analyzer(new_sorting, recording, format="memory")
    merged_analyzer = auto_merge_units(new_sorting_analyzer, presets=["x_contaminations", "x_contaminations"])
    assert len(merged_analyzer.unit_ids) < len(new_sorting_analyzer.unit_ids)


if __name__ == "__main__":
    sorting_analyzer = make_sorting_analyzer(sparse=True)
    preset = None
    test_compute_merge_unit_groups(sorting_analyzer, preset=preset)
    test_auto_merge_units(sorting_analyzer)
    test_auto_merge_units_iterative(sorting_analyzer)
