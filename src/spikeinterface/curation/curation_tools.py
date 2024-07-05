from __future__ import annotations
from typing import Optional
import numpy as np


try:
    import numba

    HAVE_NUMBA = True
except ModuleNotFoundError as err:
    HAVE_NUMBA = False

_methods = ("keep_first", "random", "keep_last", "keep_first_iterative", "keep_last_iterative")
_methods_numpy = ("keep_first", "random", "keep_last")


def _find_duplicated_spikes_numpy(
    spike_train: np.ndarray,
    censored_period: int,
    seed: Optional[int] = None,
    method: "keep_first" | "random" | "keep_last" = "keep_first",
) -> np.ndarray:
    (indices_of_duplicates,) = np.where(np.diff(spike_train) <= censored_period)

    if method == "keep_first":
        indices_of_duplicates += 1
    elif method == "random":
        rand_state = np.random.get_state()
        np.random.seed(seed)
        mask = np.ones(len(spike_train), dtype=bool)

        while np.sum(np.diff(spike_train[mask]) <= censored_period) > 0:
            shift = np.random.randint(low=0, high=2, size=len(indices_of_duplicates))
            mask[indices_of_duplicates + shift] = False
        np.random.set_state(rand_state)

        (indices_of_duplicates,) = np.where(~mask)
    elif method != "keep_last":
        raise ValueError(
            f"Method '{method}' isn't a valid method for _find_duplicated_spikes_numpy use one of {_methods_numpy}."
        )

    return indices_of_duplicates


def _find_duplicated_spikes_random(spike_train: np.ndarray, censored_period: int, seed: int) -> np.ndarray:
    # random seed
    rng = np.random.RandomState(seed=seed)

    indices_of_duplicates = []
    while not np.all(np.diff(np.delete(spike_train, indices_of_duplicates)) > censored_period):
        (duplicates,) = np.where(np.diff(spike_train) <= censored_period)
        duplicates = np.unique(np.concatenate((duplicates, duplicates + 1)))
        duplicate = rng.choice(duplicates)
        indices_of_duplicates.append(duplicate)

    return np.array(indices_of_duplicates, dtype=np.int64)


if HAVE_NUMBA:

    @numba.jit(nopython=True, nogil=True, cache=False)
    def _find_duplicated_spikes_keep_first_iterative(spike_train, censored_period):
        indices_of_duplicates = numba.typed.List()
        N = len(spike_train)

        for i in range(N - 1):
            if i in indices_of_duplicates:
                continue

            for j in range(i + 1, N):
                if spike_train[j] - spike_train[i] > censored_period:
                    break
                indices_of_duplicates.append(j)

        return np.asarray(indices_of_duplicates)

    @numba.jit(nopython=True, nogil=True, cache=True)
    def _find_duplicated_spikes_keep_last_iterative(spike_train, censored_period):
        indices_of_duplicates = numba.typed.List()
        N = len(spike_train)

        for i in range(N - 1, 0, -1):
            if i in indices_of_duplicates:
                continue

            for j in range(i - 1, -1, -1):
                if spike_train[i] - spike_train[j] > censored_period:
                    break
                indices_of_duplicates.append(j)

        return np.asarray(indices_of_duplicates)


def find_duplicated_spikes(
    spike_train,
    censored_period: int,
    method: "keep_first" | "keep_last" | "keep_first_iterative" | "keep_last_iterative" | "random" = "random",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Finds the indices where spikes should be considered duplicates.
    When two spikes are closer together than the censored period,
    one of them is taken out based on the method provided.

    Parameters
    ----------
    spike_train: np.ndarray
        The spike train on which to look for duplicated spikes.
    censored_period: int
        The censored period for duplicates (in sample time).
    method: "keep_first" |"keep_last" | "keep_first_iterative" | "keep_last_iterative" |random", default: "random"
        Method used to remove the duplicated spikes.
    seed: int | None
        The seed to use if method="random".

    Returns
    -------
    indices_of_duplicates: np.ndarray
        The indices of spikes considered to be duplicates.
    """

    if method in ("keep_first", "keep_last"):
        return _find_duplicated_spikes_numpy(spike_train, censored_period, method=method)
    elif method == "random":
        assert seed is not None, "The 'seed' has to be provided if method=='random'"
        return _find_duplicated_spikes_random(spike_train, censored_period, seed)
    elif method == "keep_first_iterative":
        assert HAVE_NUMBA, "'keep_first' method requires numba. Install it with >>> pip install numba"
        return _find_duplicated_spikes_keep_first_iterative(spike_train.astype(np.int64), censored_period)
    elif method == "keep_last_iterative":
        assert HAVE_NUMBA, "'keep_last' method requires numba. Install it with >>> pip install numba"
        return _find_duplicated_spikes_keep_last_iterative(spike_train.astype(np.int64), censored_period)
    else:
        raise ValueError(f"Method '{method}' isn't a valid method for find_duplicated_spikes. Use one of {_methods}")
