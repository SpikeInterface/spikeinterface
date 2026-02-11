"""Sorting components: peak selection"""

from __future__ import annotations


import numpy as np


def select_peaks(
    peaks, recording=None, method="uniform", seed=None, return_indices=False, margin=None, **method_kwargs
):
    """
    Method to select a subset of peaks from a set of peaks.
    Usually use for reducing computational foorptint of downstream methods.
    Parameters
    ----------
    peaks: the peaks that have been found
    method: "uniform", "uniform_locations", "smart_sampling_amplitudes", "smart_sampling_locations",
    "smart_sampling_locations_and_time"
        Method to use. Options:
            * "uniform": a random subset is selected from all the peaks, on a per channel basis by default
            * "smart_sampling_amplitudes": peaks are selected via monte-carlo rejection probabilities
                based on peak amplitudes, on a per channel basis
            * "smart_sampling_locations": peaks are selection via monte-carlo rejections probabilities
                based on peak locations, on a per area region basis-
            * "smart_sampling_locations_and_time": peaks are selection via monte-carlo rejections probabilities
                based on peak locations and time positions, assuming everything is independent

    seed: int
        The seed for random generations
    return_indices: bool
        If True, return the indices of selection such that selected_peaks = peaks[selected_indices]
    margin : Margin in timesteps. default: None. Otherwise should be a tuple (nbefore, nafter)
        preventing peaks to be selected at the borders of the segments. A recording should be provided to get the duration
        of the segments

    method_kwargs: dict of kwargs method
        Keyword arguments for the chosen method:
            "uniform":
                * select_per_channel: bool, default: False
                    If True, the selection is done on a per channel basis
                * n_peaks: int
                    If select_per_channel is True, this is the number of peaks per channels,
                    otherwise this is the total number of peaks
            "smart_sampling_amplitudes":
                * noise_levels : array
                    The noise levels used while detecting the peaks
                * n_peaks: int
                    If select_per_channel is True, this is the number of peaks per channels,
                    otherwise this is the total number of peaks
                * select_per_channel: bool, default: False
                    If True, the selection is done on a per channel basis
            "smart_sampling_locations":
                * n_peaks: int
                    Total number of peaks to select
                * peaks_locations: array
                    The locations of all the peaks, computed via localize_peaks
            "smart_sampling_locations_and_time":
                * n_peaks: int
                    Total number of peaks to select
                * peaks_locations: array
                    The locations of all the peaks, computed via localize_peaks

    {}
    Returns
    -------
    selected_peaks: array
        Selected peaks.
    selected_indices: array
        indices of peak selection such that selected_peaks = peaks[selected_indices].  Only returned when
        return_indices is True.
    """

    if margin is not None:
        assert recording is not None, "recording should be provided if margin is not None"

    selected_indices = select_peak_indices(peaks, method=method, seed=seed, **method_kwargs)
    selected_peaks = peaks[selected_indices]
    num_segments = len(np.unique(selected_peaks["segment_index"]))

    if margin is not None:
        to_keep = np.zeros(len(selected_peaks), dtype=bool)
        for segment_index in range(num_segments):
            num_samples_in_segment = recording.get_num_samples(segment_index)
            i0, i1 = np.searchsorted(selected_peaks["segment_index"], [segment_index, segment_index + 1])
            while selected_peaks["sample_index"][i0] <= margin[0]:
                i0 += 1
            while selected_peaks["sample_index"][i1 - 1] >= (num_samples_in_segment - margin[1]):
                i1 -= 1
            to_keep[i0:i1] = True
        selected_indices = selected_indices[to_keep]
        selected_peaks = peaks[selected_indices]

    if return_indices:
        return selected_peaks, selected_indices
    else:
        return selected_peaks


def select_peak_indices(peaks, method, seed, **method_kwargs):
    """
    Method to subsample all the found peaks before clustering.  Returns selected_indices.

    This function is wrapped by select_peaks -- see

    :py:func:`spikeinterface.sortingcomponents.peak_selection.select_peaks` for detailed documentation.
    """
    from sklearn.preprocessing import QuantileTransformer

    selected_indices = []

    seed = seed if seed else None
    rng = np.random.default_rng(seed=seed)

    if method == "uniform":
        params = {"select_per_channel": False, "n_peaks": None}

        params.update(method_kwargs)

        assert params["n_peaks"] is not None, "n_peaks should be defined!"

        if params["select_per_channel"]:
            ## This method will randomly select max_peaks_per_channel peaks per channels
            for channel in np.unique(peaks["channel_index"]):
                peaks_indices = np.where(peaks["channel_index"] == channel)[0]
                max_peaks = min(peaks_indices.size, params["n_peaks"])
                selected_indices += [rng.choice(peaks_indices, size=max_peaks, replace=False)]
        else:
            num_peaks = min(peaks.size, params["n_peaks"])
            selected_indices = [rng.choice(peaks.size, size=num_peaks, replace=False)]

    elif method in ["smart_sampling_amplitudes", "smart_sampling_locations", "smart_sampling_locations_and_time"]:
        if method == "smart_sampling_amplitudes":
            ## This method will try to select around n_peaks per channel but in a non uniform manner
            ## First, it will look at the distribution of the peaks amplitudes, per channel.
            ## Once this distribution is known, it will sample from the peaks with a rejection probability
            ## such that the final distribution of the amplitudes, for the selected peaks, will be as
            ## uniform as possible. In a nutshell, the method will try to sample as homogenously as possible
            ## from the space of all the peaks, using the amplitude as a discriminative criteria
            ## To do so, one must provide the noise_levels, detect_threshold used to detect the peaks, the
            ## sign of the peaks, and the number of bins for the probability density histogram

            params = {"n_peaks": None, "noise_levels": None, "select_per_channel": False}

            params.update(method_kwargs)

            assert params["n_peaks"] is not None, "n_peaks should be defined!"
            assert params["noise_levels"] is not None, "Noise levels should be provided"

            if params["select_per_channel"]:
                for channel in np.unique(peaks["channel_index"]):
                    peaks_indices = np.where(peaks["channel_index"] == channel)[0]
                    if params["n_peaks"] > peaks_indices.size:
                        selected_indices += [peaks_indices]
                    else:
                        sub_peaks = peaks[peaks_indices]
                        snrs = sub_peaks["amplitude"] / params["noise_levels"][channel]
                        preprocessing = QuantileTransformer(
                            output_distribution="uniform", n_quantiles=min(100, len(snrs))
                        )
                        snrs = preprocessing.fit_transform(snrs[:, np.newaxis])

                        my_selection = np.zeros(0, dtype=np.int32)
                        all_index = np.arange(len(snrs))
                        while my_selection.size < params["n_peaks"]:
                            candidates = all_index[np.logical_not(np.isin(all_index, my_selection))]
                            probabilities = rng.random(size=len(candidates))
                            valid = candidates[np.where(snrs[candidates, 0] < probabilities)[0]]
                            my_selection = np.concatenate((my_selection, valid))

                        selected_indices += [peaks_indices[rng.permutation(my_selection)[: params["n_peaks"]]]]

            else:
                if params["n_peaks"] > peaks.size:
                    selected_indices += [np.arange(peaks.size)]
                else:
                    snrs = peaks["amplitude"] / params["noise_levels"][peaks["channel_index"]]
                    preprocessing = QuantileTransformer(output_distribution="uniform", n_quantiles=min(100, len(snrs)))
                    snrs = preprocessing.fit_transform(snrs[:, np.newaxis])

                    my_selection = np.zeros(0, dtype=np.int32)
                    all_index = np.arange(len(snrs))
                    while my_selection.size < params["n_peaks"]:
                        candidates = all_index[np.logical_not(np.isin(all_index, my_selection))]
                        probabilities = rng.random(size=len(candidates))
                        valid = candidates[np.where(snrs[candidates, 0] < probabilities)[0]]
                        my_selection = np.concatenate((my_selection, valid))

                    selected_indices = [rng.permutation(my_selection)[: params["n_peaks"]]]

        elif method == "smart_sampling_locations":
            ## This method will try to select around n_peaksbut in a non uniform manner
            ## First, it will look at the distribution of the positions.
            ## Once this distribution is known, it will sample from the peaks with a rejection probability
            ## such that the final distribution of the amplitudes, for the selected peaks, will be as
            ## uniform as possible. In a nutshell, the method will try to sample as homogenously as possible
            ## from the space of all the peaks, using the locations as a discriminative criteria
            ## To do so, one must provide the peaks locations, and the number of bins for the
            ## probability density histogram

            params = {"peaks_locations": None, "n_peaks": None}

            params.update(method_kwargs)

            assert params["n_peaks"] is not None, "n_peaks should be defined!"
            assert params["peaks_locations"] is not None, "peaks_locations should be defined!"

            nb_spikes = len(params["peaks_locations"]["x"])

            if params["n_peaks"] > nb_spikes:
                selected_indices += [np.arange(peaks.size)]
            else:
                preprocessing = QuantileTransformer(output_distribution="uniform", n_quantiles=min(100, nb_spikes))
                data = np.array([params["peaks_locations"]["x"], params["peaks_locations"]["y"]]).T
                data = preprocessing.fit_transform(data)

                my_selection = np.zeros(0, dtype=np.int32)
                all_index = np.arange(peaks.size)
                while my_selection.size < params["n_peaks"]:
                    candidates = all_index[np.logical_not(np.isin(all_index, my_selection))]

                    probabilities = rng.random(size=len(candidates))
                    data_x = data[:, 0] < probabilities

                    probabilities = rng.random(size=len(candidates))
                    data_y = data[:, 1] < probabilities

                    valid = candidates[np.where(data_x * data_y)[0]]
                    my_selection = np.concatenate((my_selection, valid))

                selected_indices = [rng.permutation(my_selection)[: params["n_peaks"]]]

        elif method == "smart_sampling_locations_and_time":
            ## This method will try to select around n_peaksbut in a non uniform manner
            ## First, it will look at the distribution of the positions.
            ## Once this distribution is known, it will sample from the peaks with a rejection probability
            ## such that the final distribution of the amplitudes, for the selected peaks, will be as
            ## uniform as possible. In a nutshell, the method will try to sample as homogenously as possible
            ## from the space of all the peaks, using the locations as a discriminative criteria
            ## To do so, one must provide the peaks locations, and the number of bins for the
            ## probability density histogram

            params = {"peaks_locations": None, "n_peaks": None}

            params.update(method_kwargs)

            assert params["n_peaks"] is not None, "n_peaks should be defined!"
            assert params["peaks_locations"] is not None, "peaks_locations should be defined!"

            nb_spikes = len(params["peaks_locations"]["x"])

            if params["n_peaks"] > nb_spikes:
                selected_indices += [np.arange(peaks.size)]
            else:
                preprocessing = QuantileTransformer(output_distribution="uniform", n_quantiles=min(100, nb_spikes))
                data = np.array(
                    [params["peaks_locations"]["x"], params["peaks_locations"]["y"], peaks["sample_index"]]
                ).T
                data = preprocessing.fit_transform(data)

                my_selection = np.zeros(0, dtype=np.int32)
                all_index = np.arange(peaks.size)
                while my_selection.size < params["n_peaks"]:
                    candidates = all_index[np.logical_not(np.isin(all_index, my_selection))]

                    probabilities = rng.random(size=len(candidates))
                    data_x = data[:, 0] < probabilities

                    probabilities = rng.random(size=len(candidates))
                    data_y = data[:, 1] < probabilities

                    probabilities = rng.random(size=len(candidates))
                    data_t = data[:, 2] < probabilities

                    valid = candidates[np.where(data_x * data_y * data_t)[0]]
                    my_selection = np.concatenate((my_selection, valid))

                selected_indices = [rng.permutation(my_selection)[: params["n_peaks"]]]

    else:
        raise NotImplementedError(
            f"The 'method' {method} does not exist for peaks selection." f" possible methods are {_possible_methods}"
        )

    selected_indices = np.concatenate(selected_indices)
    selected_indices = selected_indices[
        np.lexsort((peaks[selected_indices]["sample_index"], peaks[selected_indices]["segment_index"]))
    ]
    return selected_indices


_possible_methods = (
    "uniform",
    "uniform_locations",
    "smart_sampling_amplitudes",
    "smart_sampling_locations",
    "smart_sampling_locations_and_time",
)
