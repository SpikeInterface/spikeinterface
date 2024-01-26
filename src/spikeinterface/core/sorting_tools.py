from __future__ import annotations
from .basesorting import BaseSorting
import numpy as np


def spike_vector_to_spike_trains(spike_vector: list[np.array], unit_ids: np.array) -> dict[dict[str, np.array]]:
    """
    Computes all spike trains for all units/segments from a spike vector list.

    Internally calls numba if numba is installed.

    Parameters
    ----------
    spike_vector: list[np.ndarray]
        List of spike vectors optained with sorting.to_spike_vector(concatenated=False)
    unit_ids: np.array
        Unit ids

    Returns
    -------
    spike_trains: dict[dict]:
        A dict containing, for each segment, the spike trains of all units
        (as a dict: unit_id --> spike_train).
    """

    try:
        import numba

        HAVE_NUMBA = True
    except:
        HAVE_NUMBA = False

    if HAVE_NUMBA:
        # the trick here is to have a function getter
        vector_to_list_of_spiketrain = get_numba_vector_to_list_of_spiketrain()
    else:
        vector_to_list_of_spiketrain = vector_to_list_of_spiketrain_numpy

    num_units = unit_ids.size
    spike_trains = {}
    for segment_index, spikes in enumerate(spike_vector):
        sample_indices = np.array(spikes["sample_index"]).astype(np.int64, copy=False)
        unit_indices = np.array(spikes["unit_index"]).astype(np.int64, copy=False)
        list_of_spiketrains = vector_to_list_of_spiketrain(sample_indices, unit_indices, num_units)
        spike_trains[segment_index] = dict(zip(unit_ids, list_of_spiketrains))

    return spike_trains


def vector_to_list_of_spiketrain_numpy(sample_indices, unit_indices, num_units):
    """
    Slower implementation of vetor_to_dict using numpy boolean mask.
    This is for one segment.
    """
    spike_trains = []
    for u in range(num_units):
        spike_trains.append(sample_indices[unit_indices == u])
    return spike_trains


def get_numba_vector_to_list_of_spiketrain():
    if hasattr(get_numba_vector_to_list_of_spiketrain, "_cached_numba_function"):
        return get_numba_vector_to_list_of_spiketrain._cached_numba_function

    import numba

    @numba.jit((numba.int64[::1], numba.int64[::1], numba.int64), nopython=True, nogil=True, cache=True)
    def vector_to_list_of_spiketrain_numba(sample_indices, unit_indices, num_units):
        """
        Fast implementation of vector_to_dict using numba loop.
        This is for one segment.
        """
        num_spikes = sample_indices.size
        num_spike_per_units = np.zeros(num_units, dtype=np.int32)
        for s in range(num_spikes):
            num_spike_per_units[unit_indices[s]] += 1

        spike_trains = []
        for u in range(num_units):
            spike_trains.append(np.empty(num_spike_per_units[u], dtype=np.int64))

        current_x = np.zeros(num_units, dtype=np.int64)
        for s in range(num_spikes):
            unit_index = unit_indices[s]
            spike_trains[unit_index][current_x[unit_index]] = sample_indices[s]
            current_x[unit_index] += 1

        return spike_trains

    # Cache the compiled function
    get_numba_vector_to_list_of_spiketrain._cached_numba_function = vector_to_list_of_spiketrain_numba

    return vector_to_list_of_spiketrain_numba


# TODO later : implement other method like "maximum_rate", "by_percent", ...
def random_spikes_selection(
    sorting: BaseSorting,
    num_samples: int,
    method: str = "uniform",
    max_spikes_per_unit: int = 500,
    margin_size: int | None = None,
    seed: int | None = None,
):
    """
    This replaces `select_random_spikes_uniformly()`.
    Random spikes selection of spike across per units.
    Can optionally avoid spikes on segment borders if
    margin_size is not None.

    Parameters
    ----------
    sorting: BaseSorting
        The sorting object
    num_samples: list of int
        The number of samples per segment.
        Can be retrieved from recording with
        num_samples = [recording.get_num_samples(seg_index) for seg_index in range(recording.get_num_segments())]
    method: "uniform", default: "uniform"
        The method to use. Only "uniform" is implemented for now
    max_spikes_per_unit: int, default: 500
        The number of spikes per units
    margin_size: None | int, default: None
        A margin on each border of segments to avoid spikes
    seed: None | int, default: None
        A seed for random generator

    Returns
    -------
    random_spikes_indices: np.array
        Selected spike indices coresponding to the sorting spike vector.
    """
    rng = np.random.default_rng(seed=seed)
    spikes = sorting.to_spike_vector()

    random_spikes_indices = []
    for unit_index, unit_id in enumerate(sorting.unit_ids):
        all_unit_indices = np.flatnonzero(unit_index == spikes["unit_index"])

        if method == "uniform":
            selected_unit_indices = rng.choice(
                all_unit_indices, size=min(max_spikes_per_unit, all_unit_indices.size), replace=False, shuffle=False
            )
        else:
            raise ValueError(f"random_spikes_selection wrong method {method}, currently only 'uniform' can be used.")

        if margin_size is not None:
            margin_size = int(margin_size)
            keep = np.ones(selected_unit_indices.size, dtype=bool)
            # left margin
            keep[selected_unit_indices < margin_size] = False
            # right margin
            for segment_index in range(sorting.get_num_segments()):
                remove_mask = np.flatnonzero(
                    (spikes[selected_unit_indices]["segment_index"] == segment_index)
                    & (spikes[selected_unit_indices]["sample_index"] >= (num_samples[segment_index] - margin_size))
                )
                keep[remove_mask] = False
            selected_unit_indices = selected_unit_indices[keep]

        random_spikes_indices.append(selected_unit_indices)

    random_spikes_indices = np.concatenate(random_spikes_indices)
    random_spikes_indices = np.sort(random_spikes_indices)

    return random_spikes_indices
