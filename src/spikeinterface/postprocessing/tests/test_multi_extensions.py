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
# for some extensions (templates, amplitude_scalings), since the templates slightly change for merges/splits
# we allow a relative tolerance
# (amplitud_scalings are the moste sensitive!)
extensions_with_rel_tolerance_merge = {
    "amplitude_scalings": 1e-1,
    "templates": 1e-3,
    "template_similarity": 1e-3,
    "unit_locations": 1e-3,
    "template_metrics": 1e-3,
    "quality_metrics": 1e-3,
}
extensions_with_rel_tolerance_splits = {"amplitude_scalings": 1e-1}


def get_dataset_to_merge():
    # generate a dataset with some split units to minimize merge errors
    recording, sorting = generate_ground_truth_recording(
        durations=[30.0],
        sampling_frequency=16000.0,
        num_channels=10,
        num_units=10,
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_levels=5.0, strategy="tile_pregenerated"),
        generate_unit_locations_kwargs=dict(margin_um=10.0, minimum_z=2.0, maximum_z=15.0, minimum_distance=20),
        seed=2205,
    )

    channel_ids_as_integers = [id for id in range(recording.get_num_channels())]
    unit_ids_as_integers = [id for id in range(sorting.get_num_units())]
    recording = recording.rename_channels(new_channel_ids=channel_ids_as_integers)
    sorting = sorting.rename_units(new_unit_ids=unit_ids_as_integers)

    # since templates are going to be averaged and this might be a problem for amplitude scaling
    # we select the 3 units with the largest templates to split
    analyzer_raw = create_sorting_analyzer(sorting, recording, format="memory", sparse=False)
    analyzer_raw.compute(["random_spikes", "templates"])
    # select 3 largest templates to split
    sort_by_amp = np.argsort(list(get_template_extremum_amplitude(analyzer_raw).values()))[::-1]
    split_ids = sorting.unit_ids[sort_by_amp][:3]

    sorting_with_splits, split_unit_ids = inject_some_split_units(
        sorting, num_split=3, split_ids=split_ids, output_ids=True, seed=0
    )
    return recording, sorting_with_splits, split_unit_ids


def get_dataset_to_split():
    # generate a dataset and return large unit to split to minimize split errors
    recording, sorting = generate_ground_truth_recording(
        durations=[30.0],
        sampling_frequency=16000.0,
        num_channels=10,
        num_units=10,
        generate_sorting_kwargs=dict(firing_rates=10.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_levels=5.0, strategy="tile_pregenerated"),
        seed=2205,
    )

    channel_ids_as_integers = [id for id in range(recording.get_num_channels())]
    unit_ids_as_integers = [id for id in range(sorting.get_num_units())]
    recording = recording.rename_channels(new_channel_ids=channel_ids_as_integers)
    sorting = sorting.rename_units(new_unit_ids=unit_ids_as_integers)

    # since templates are going to be averaged and this might be a problem for amplitude scaling
    # we select the 3 units with the largest templates to split
    analyzer_raw = create_sorting_analyzer(sorting, recording, format="memory", sparse=False)
    analyzer_raw.compute(["random_spikes", "templates"])
    # select 3 largest templates to split
    sort_by_amp = np.argsort(list(get_template_extremum_amplitude(analyzer_raw).values()))[::-1]
    large_units = sorting.unit_ids[sort_by_amp][:2]

    return recording, sorting, large_units


@pytest.fixture(scope="module")
def dataset_to_merge():
    return get_dataset_to_merge()


@pytest.fixture(scope="module")
def dataset_to_split():
    return get_dataset_to_split()


@pytest.mark.parametrize("sparse", [False, True])
def test_SortingAnalyzer_merge_all_extensions(dataset_to_merge, sparse):
    set_global_job_kwargs(n_jobs=1)

    recording, sorting, other_ids = dataset_to_merge

    sorting_analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparse=sparse)
    extension_dict_merge = extension_dict.copy()

    # we apply the merges according to the artificial splits
    merges = [list(v) for v in other_ids.values()]
    split_unit_ids = np.ravel(merges)
    unmerged_unit_ids = sorting_analyzer.unit_ids[~np.isin(sorting_analyzer.unit_ids, split_unit_ids)]

    sorting_analyzer.compute(extension_dict_merge, n_jobs=1)

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
            if ext in extensions_with_rel_tolerance_merge:
                rtol = extensions_with_rel_tolerance_merge[ext]
            else:
                rtol = 0
            if extension_data_type[ext] == "pandas":
                data_hard_merged = data_hard_merged.dropna().to_numpy().astype("float")
                data_soft_merged = data_soft_merged.dropna().to_numpy().astype("float")
            if data_hard_merged.dtype.fields is None:
                if not np.allclose(data_hard_merged, data_soft_merged, rtol=rtol):
                    max_error = np.max(np.abs(data_hard_merged - data_soft_merged))
                    raise Exception(f"Failed for {ext} - max error {max_error}")
            else:
                for f in data_hard_merged.dtype.fields:
                    if not np.allclose(data_hard_merged[f], data_soft_merged[f], rtol=rtol):
                        max_error = np.max(np.abs(data_hard_merged[f] - data_soft_merged[f]))
                        raise Exception(f"Failed for {ext} - field {f} - max error {max_error}")


@pytest.mark.parametrize("sparse", [False, True])
def test_SortingAnalyzer_split_all_extensions(dataset_to_split, sparse):
    set_global_job_kwargs(n_jobs=1)

    recording, sorting, units_to_split = dataset_to_split

    sorting_analyzer = create_sorting_analyzer(sorting, recording, format="memory", sparse=sparse)
    extension_dict_split = extension_dict.copy()
    sorting_analyzer.compute(extension_dict, n_jobs=1)

    # we randomly apply splits (at half of spiketrain)
    num_spikes = sorting.count_num_spikes_per_unit()

    unsplit_unit_ids = sorting_analyzer.unit_ids[~np.isin(sorting_analyzer.unit_ids, units_to_split)]
    splits = {}
    for unit in units_to_split:
        splits[unit] = [np.arange(num_spikes[unit] // 2), np.arange(num_spikes[unit] // 2, num_spikes[unit])]

    analyzer_split, split_unit_ids = sorting_analyzer.split_units(split_units=splits, return_new_unit_ids=True)
    split_unit_ids = list(np.concatenate(split_unit_ids))

    # also do a full recopute
    analyzer_hard = create_sorting_analyzer(analyzer_split.sorting, recording, format="memory", sparse=sparse)
    # we propagate random spikes to avoid random spikes to be recomputed
    extension_dict_ = extension_dict_split.copy()
    extension_dict_.pop("random_spikes")
    analyzer_hard.extensions["random_spikes"] = analyzer_split.extensions["random_spikes"]
    analyzer_hard.compute(extension_dict_, n_jobs=1)

    for ext in extension_dict:
        # 1. check that data are exactly the same for unchanged units between original/split
        data_original = sorting_analyzer.get_extension(ext).get_data()
        data_split = analyzer_split.get_extension(ext).get_data()
        data_recompute = analyzer_hard.get_extension(ext).get_data()
        if ext in data_with_miltiple_returns:
            data_original = data_original[0]
            data_split = data_split[0]
            data_recompute = data_recompute[0]
        data_original_unsplit = get_extension_data_for_units(
            sorting_analyzer, data_original, unsplit_unit_ids, extension_data_type[ext]
        )
        data_split_unsplit = get_extension_data_for_units(
            analyzer_split, data_split, unsplit_unit_ids, extension_data_type[ext]
        )

        np.testing.assert_array_equal(data_original_unsplit, data_split_unsplit)

        # 2. check that split data are the same for extension split and recompute
        data_split_soft = get_extension_data_for_units(
            analyzer_split, data_split, split_unit_ids, extension_data_type[ext]
        )
        data_split_hard = get_extension_data_for_units(
            analyzer_hard, data_recompute, split_unit_ids, extension_data_type[ext]
        )
        if ext not in random_computation:
            if ext in extensions_with_rel_tolerance_splits:
                rtol = extensions_with_rel_tolerance_splits[ext]
            else:
                rtol = 0
            if extension_data_type[ext] == "pandas":
                data_split_soft = data_split_soft.dropna().to_numpy().astype("float")
                data_split_hard = data_split_hard.dropna().to_numpy().astype("float")
            if data_split_hard.dtype.fields is None:
                if not np.allclose(data_split_hard, data_split_soft, rtol=rtol):
                    max_error = np.max(np.abs(data_split_hard - data_split_soft))
                    raise Exception(f"Failed for {ext} - max error {max_error}")
            else:
                for f in data_split_hard.dtype.fields:
                    if not np.allclose(data_split_hard[f], data_split_soft[f], rtol=rtol):
                        max_error = np.max(np.abs(data_split_hard[f] - data_split_soft[f]))
                        raise Exception(f"Failed for {ext} - field {f} - max error {max_error}")


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
    dataset = get_dataset_to_merge()
    test_SortingAnalyzer_merge_all_extensions(dataset, False)
