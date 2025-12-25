import pytest


from spikeinterface.core import create_sorting_analyzer
from spikeinterface.curation import compute_merge_unit_groups, auto_merge_units
from spikeinterface.generation import split_sorting_by_times


from spikeinterface.curation.tests.common import (
    make_sorting_analyzer,
    sorting_analyzer_for_curation,
    sorting_analyzer_with_splits,
    sorting_analyzer_multi_segment_for_curation,
)


@pytest.mark.parametrize(
    "preset", ["x_contaminations", "feature_neighbors", "temporal_splits", "similarity_correlograms", "slay", None]
)
def test_compute_merge_unit_groups(sorting_analyzer_with_splits, preset):

    job_kwargs = dict(n_jobs=-1)
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
            **job_kwargs,
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
            **job_kwargs,
        )


@pytest.mark.parametrize(
    "preset", ["x_contaminations", "feature_neighbors", "temporal_splits", "similarity_correlograms", "slay"]
)
def test_compute_merge_unit_groups_multi_segment(sorting_analyzer_multi_segment_for_curation, preset):
    job_kwargs = dict(n_jobs=-1)
    sorting_analyzer = sorting_analyzer_multi_segment_for_curation
    print(sorting_analyzer)

    merge_unit_groups = compute_merge_unit_groups(
        sorting_analyzer,
        preset=preset,
        **job_kwargs,
    )


def test_auto_merge_units(sorting_analyzer_for_curation):
    recording = sorting_analyzer_for_curation.recording
    job_kwargs = dict(n_jobs=-1)
    new_sorting, _ = split_sorting_by_times(sorting_analyzer_for_curation)
    new_sorting_analyzer = create_sorting_analyzer(new_sorting, recording, format="memory")
    merged_analyzer = auto_merge_units(new_sorting_analyzer, presets="x_contaminations", **job_kwargs)
    assert len(merged_analyzer.unit_ids) < len(new_sorting_analyzer.unit_ids)

    step_merged_analyzer = auto_merge_units(
        new_sorting_analyzer,
        presets=None,
        steps=["num_spikes", "remove_contaminated", "unit_locations", "template_similarity", "quality_score"],
        steps_params={"num_spikes": {"min_spikes": 150}},
        **job_kwargs,
    )
    assert len(step_merged_analyzer.unit_ids) < len(new_sorting_analyzer.unit_ids)


def test_auto_merge_units_iterative(sorting_analyzer_for_curation):
    recording = sorting_analyzer_for_curation.recording
    job_kwargs = dict(n_jobs=-1)
    new_sorting, _ = split_sorting_by_times(sorting_analyzer_for_curation)
    new_sorting_analyzer = create_sorting_analyzer(new_sorting, recording, format="memory")
    merged_analyzer = auto_merge_units(
        new_sorting_analyzer, presets=["x_contaminations", "x_contaminations"], **job_kwargs
    )
    assert len(merged_analyzer.unit_ids) < len(new_sorting_analyzer.unit_ids)


if __name__ == "__main__":
    sorting_analyzer = make_sorting_analyzer(sparse=True)
    preset = None
    test_compute_merge_unit_groups(sorting_analyzer, preset=preset)
    test_auto_merge_units(sorting_analyzer)
    test_auto_merge_units_iterative(sorting_analyzer)
