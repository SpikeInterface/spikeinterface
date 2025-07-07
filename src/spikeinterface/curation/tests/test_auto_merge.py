import pytest


from spikeinterface.core import create_sorting_analyzer
from spikeinterface.core.generate import inject_some_split_units
from spikeinterface.curation import compute_merge_unit_groups, auto_merge_units
from spikeinterface.generation import split_sorting_by_times


from spikeinterface.curation.tests.common import make_sorting_analyzer, sorting_analyzer_for_curation


@pytest.mark.parametrize(
    "preset", ["x_contaminations", "feature_neighbors", "temporal_splits", "similarity_correlograms", None]
)
def test_compute_merge_unit_groups(sorting_analyzer_for_curation, preset):

    print(sorting_analyzer_for_curation)
    sorting = sorting_analyzer_for_curation.sorting
    recording = sorting_analyzer_for_curation.recording
    num_unit_splited = 1
    num_split = 2

    split_ids = sorting.unit_ids[:num_unit_splited]
    sorting_with_split, other_ids = inject_some_split_units(
        sorting,
        split_ids=split_ids,
        num_split=num_split,
        output_ids=True,
        seed=42,
    )

    job_kwargs = dict(n_jobs=-1)

    sorting_analyzer = create_sorting_analyzer(sorting_with_split, recording, format="memory")
    sorting_analyzer.compute(
        [
            "random_spikes",
            "waveforms",
            "templates",
            "unit_locations",
            "spike_amplitudes",
            "spike_locations",
            "correlograms",
            "template_similarity",
        ],
        **job_kwargs,
    )

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
            assert len(merge_unit_groups) == num_unit_splited
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


# DEBUG
# import matplotlib.pyplot as plt
# from spikeinterface.curation.auto_merge import normalize_correlogram
# templates_diff = outs['templates_diff']
# correlogram_diff = outs['correlogram_diff']
# bins = outs['bins']
# correlograms_smoothed = outs['correlograms_smoothed']
# correlograms = outs['correlograms']
# win_sizes = outs['win_sizes']

# fig, ax = plt.subplots()
# ax.hist(correlogram_diff.flatten(), bins=np.arange(0, 1, 0.05))

# fig, ax = plt.subplots()
# ax.hist(templates_diff.flatten(), bins=np.arange(0, 1, 0.05))

# m = correlograms.shape[2] // 2

# for unit_id1, unit_id2 in merge_unit_groups[:5]:
#     unit_ind1 = sorting_with_split.id_to_index(unit_id1)
#     unit_ind2 = sorting_with_split.id_to_index(unit_id2)

#     bins2 = bins[:-1] + np.mean(np.diff(bins))
#     fig, axs = plt.subplots(ncols=3)
#     ax = axs[0]
#     ax.plot(bins2, correlograms[unit_ind1, unit_ind1, :], color='b')
#     ax.plot(bins2, correlograms[unit_ind2, unit_ind2, :], color='r')
#     ax.plot(bins2, correlograms_smoothed[unit_ind1, unit_ind1, :], color='b')
#     ax.plot(bins2, correlograms_smoothed[unit_ind2, unit_ind2, :], color='r')

#     ax.set_title(f'{unit_id1} {unit_id2}')
#     ax = axs[1]
#     ax.plot(bins2, correlograms_smoothed[unit_ind1, unit_ind2, :], color='g')

#     auto_corr1 = normalize_correlogram(correlograms_smoothed[unit_ind1, unit_ind1, :])
#     auto_corr2 = normalize_correlogram(correlograms_smoothed[unit_ind2, unit_ind2, :])
#     cross_corr = normalize_correlogram(correlograms_smoothed[unit_ind1, unit_ind2, :])

#     ax = axs[2]
#     ax.plot(bins2, auto_corr1, color='b')
#     ax.plot(bins2, auto_corr2, color='r')
#     ax.plot(bins2, cross_corr, color='g')

#     ax.axvline(bins2[m - win_sizes[unit_ind1]], color='b')
#     ax.axvline(bins2[m + win_sizes[unit_ind1]], color='b')
#     ax.axvline(bins2[m - win_sizes[unit_ind2]], color='r')
#     ax.axvline(bins2[m + win_sizes[unit_ind2]], color='r')

#     ax.set_title(f'corr diff {correlogram_diff[unit_ind1, unit_ind2]} - temp diff {templates_diff[unit_ind1, unit_ind2]}')
#     plt.show()


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
