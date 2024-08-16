import pytest

import time
import numpy as np

from spikeinterface import (
    create_sorting_analyzer,
    generate_ground_truth_recording,
    set_global_job_kwargs,
    get_template_extremum_amplitude,
)
from spikeinterface.core.generate import inject_some_split_units


def get_dataset():
    recording, sorting = generate_ground_truth_recording(
        durations=[30.0],
        sampling_frequency=16000.0,
        num_channels=10,
        num_units=10,
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_levels=5.0, strategy="tile_pregenerated"),
        seed=2205,
    )

    # since templates are going to be averaged and this might be a problem for amplitude scaling
    # we select the 3 units with the largest templates to split
    analyzer_raw = create_sorting_analyzer(sorting, recording, format="memory", sparse=False)
    analyzer_raw.compute(["random_spikes", "templates"])
    # select 3 largest templates to split
    sort_by_amp = np.argsort(list(get_template_extremum_amplitude(analyzer_raw).values()))[::-1]
    split_ids = sorting.unit_ids[sort_by_amp][:3]

    sorting_with_splits, other_ids = inject_some_split_units(
        sorting, num_split=3, split_ids=split_ids, output_ids=True, seed=0
    )
    return recording, sorting_with_splits, other_ids


@pytest.fixture(scope="module")
def dataset():
    return get_dataset()


@pytest.mark.parametrize("sparse", [False, True])
def test_SortingAnalyzer_merge_all_extensions(dataset, sparse):
    set_global_job_kwargs(n_jobs=1)

    recording, sorting, other_ids = dataset

    sorting_analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparse=sparse)

    # we apply the merges according to the artificial splits
    merges = [list(v) for v in other_ids.values()]
    split_unit_ids = np.ravel(merges)
    unmerged_unit_ids = sorting_analyzer.unit_ids[~np.isin(sorting_analyzer.unit_ids, split_unit_ids)]

    # even if this is in postprocessing, we make an extension for quality metrics
    extension_dict = {
        "noise_levels": dict(),
        "random_spikes": dict(),
        "waveforms": dict(),
        "templates": dict(),
        "principal_components": dict(),
        "spike_amplitudes": dict(),
        "template_similarity": dict(),
        "correlograms": dict(),
        "isi_histograms": dict(),
        "amplitude_scalings": dict(handle_collisions=False),  # otherwise hard mode could fail due to dropped spikes
        "spike_locations": dict(method="center_of_mass"),  # trick to avoid UserWarning
        "unit_locations": dict(),
        "template_metrics": dict(),
        "quality_metrics": dict(metric_names=["firing_rate", "isi_violation", "snr"]),
    }
    extension_data_type = {
        "noise_levels": None,
        "templates": "unit",
        "isi_histograms": "unit",
        "unit_locations": "unit",
        "spike_amplitudes": "spike",
        "amplitude_scalings": "spike",
        "spike_locations": "spike",
        "quality_metrics": "pandas",
        "template_metrics": "pandas",
        "correlograms": "matrix",
        "template_similarity": "matrix",
        "principal_components": "random",
        "waveforms": "random",
        "random_spikes": "random_spikes",
    }
    data_with_miltiple_returns = ["isi_histograms", "correlograms"]

    # due to incremental PCA, hard computation could result in different results for PCA
    # the model is differents always
    random_computation = ["principal_components"]

    sorting_analyzer.compute(extension_dict, n_jobs=1)

    # TODO: still some UserWarnings for n_jobs, where from?
    t0 = time.perf_counter()
    analyzer_merged_hard = sorting_analyzer.merge_units(
        merge_unit_groups=merges, censor_ms=2, merging_mode="hard", n_jobs=1
    )
    t_hard = time.perf_counter() - t0

    t0 = time.perf_counter()
    analyzer_merged_soft = sorting_analyzer.merge_units(
        merge_unit_groups=merges, censor_ms=2, merging_mode="soft", sparsity_overlap=0.0, n_jobs=1
    )
    t_soft = time.perf_counter() - t0

    # soft must faster
    assert t_soft < t_hard
    np.testing.assert_array_equal(analyzer_merged_hard.unit_ids, analyzer_merged_soft.unit_ids)
    new_unit_ids = list(np.arange(max(split_unit_ids) + 1, max(split_unit_ids) + 1 + len(merges)))
    np.testing.assert_array_equal(analyzer_merged_hard.unit_ids, list(unmerged_unit_ids) + new_unit_ids)

    for ext in extension_dict:
        # 1. check that data are exactly the same for unchanged units between hard/soft/original
        data_original = sorting_analyzer.get_extension(ext).get_data()
        data_hard = analyzer_merged_hard.get_extension(ext).get_data()
        data_soft = analyzer_merged_soft.get_extension(ext).get_data()
        if ext in data_with_miltiple_returns:
            data_original = data_original[0]
            data_hard = data_hard[0]
            data_soft = data_soft[0]
        data_original_unmerged = get_extension_data_for_units(
            sorting_analyzer, data_original, unmerged_unit_ids, extension_data_type[ext]
        )
        data_hard_unmerged = get_extension_data_for_units(
            analyzer_merged_hard, data_hard, unmerged_unit_ids, extension_data_type[ext]
        )
        data_soft_unmerged = get_extension_data_for_units(
            analyzer_merged_soft, data_soft, unmerged_unit_ids, extension_data_type[ext]
        )

        np.testing.assert_array_equal(data_original_unmerged, data_soft_unmerged)

        if ext not in random_computation:
            np.testing.assert_array_equal(data_original_unmerged, data_hard_unmerged)
        else:
            print(f"Skipping hard test for {ext} due to randomness in computation")

        # 2. check that soft/hard data are similar for merged units
        data_hard_merged = get_extension_data_for_units(
            analyzer_merged_hard, data_hard, new_unit_ids, extension_data_type[ext]
        )
        data_soft_merged = get_extension_data_for_units(
            analyzer_merged_soft, data_soft, new_unit_ids, extension_data_type[ext]
        )

        if ext not in random_computation:
            if extension_data_type[ext] == "pandas":
                data_hard_merged = data_hard_merged.dropna().to_numpy().astype("float")
                data_soft_merged = data_soft_merged.dropna().to_numpy().astype("float")
            if data_hard_merged.dtype.fields is None:
                assert np.allclose(data_hard_merged, data_soft_merged, rtol=0.1)
            else:
                for f in data_hard_merged.dtype.fields:
                    assert np.allclose(data_hard_merged[f], data_soft_merged[f], rtol=0.1)


def get_extension_data_for_units(sorting_analyzer, data, unit_ids, ext_data_type):
    unit_indices = sorting_analyzer.sorting.ids_to_indices(unit_ids)
    spike_vector = sorting_analyzer.sorting.to_spike_vector()
    if ext_data_type is None:
        return data
    elif ext_data_type == "random_spikes":
        random_spikes = spike_vector[data]
        unit_mask = np.isin(random_spikes["unit_index"], unit_indices)
        # since merging could scramble unit ids and drop spikes, we need to get the original unit ids
        return sorting_analyzer.unit_ids[random_spikes[unit_mask]["unit_index"]]
    elif ext_data_type == "random":
        random_indices = sorting_analyzer.get_extension("random_spikes").get_data()
        unit_mask = np.isin(spike_vector[random_indices]["unit_index"], unit_indices)
        return data[unit_mask]
    elif ext_data_type == "matrix":
        return data[unit_indices][:, unit_indices]
    elif ext_data_type == "unit":
        return data[unit_indices]
    elif ext_data_type == "spike":
        unit_mask = np.isin(spike_vector["unit_index"], unit_indices)
        return data[unit_mask]
    elif ext_data_type == "pandas":
        return data.loc[unit_ids].dropna()


if __name__ == "__main__":
    dataset = get_dataset()
    test_SortingAnalyzer_merge_all_extensions(dataset, False)
