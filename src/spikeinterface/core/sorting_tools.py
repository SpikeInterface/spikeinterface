from __future__ import annotations
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
