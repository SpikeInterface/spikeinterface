"""Sorting components: peak selection"""

import numpy as np
import scipy
from sklearn.preprocessing import QuantileTransformer


def select_peaks(peaks, method='uniform_amplitudes', seed=None, **method_kwargs):

    """Method to subsample all the found peaks before clustering
    Parameters
    ----------
    peaks: the peaks that have been found
    method: 'uniform', 'uniform_locations', 'smart_sampling_amplitudes', 'smart_sampling_locations', 
    'smart_sampling_locations_and_time'
        Method to use. Options:
            * 'uniform': a random subset is selected from all the peaks, on a per channel basis by default
            * 'uniform_locations': a random subset is selected from all the peaks, to cover uniformly the space
            * 'smart_sampling_amplitudes': peaks are selected via monte-carlo rejection probabilities 
                based on peak amplitudes, on a per channel basis
            * 'smart_sampling_locations': peaks are selection via monte-carlo rejections probabilities 
                based on peak locations, on a per area region basis
            * 'smart_sampling_locations_and_time': peaks are selection via monte-carlo rejections probabilities 
                based on peak locations and time positions, assuming everything is independent
    
    seed: int
        The seed for random generations
    
    method_kwargs: dict of kwargs method
        Keyword arguments for the chosen method:
            'uniform':
                * select_per_channel: bool
                    If True, the selection is done on a per channel basis (False by default)
                * n_peaks: int
                    If select_per_channel is True, this is the number of peaks per channels,
                    otherwise this is the total number of peaks
            'uniform_locations':
                * peaks_locations: array
                    The locations of all the peaks, computed via localize_peaks
                * n_peaks: int
                    The number of peaks to select in a given region of the space, in a uniform manner
                * n_bins: tuple
                    The number of bins used to delimit the space in (x, y) dimensions [default (10, 10)]
            'smart_sampling_amplitudes':
                * noise_levels : array
                    The noise levels used while detecting the peaks
                * n_peaks: int
                    If select_per_channel is True, this is the number of peaks per channels,
                    otherwise this is the total number of peaks
                * select_per_channel: bool
                    If True, the selection is done on a per channel basis (False by default)
            'smart_sampling_locations':
                * n_peaks: int
                    Total number of peaks to select
                * peaks_locations: array
                    The locations of all the peaks, computed via localize_peaks
            'smart_sampling_locations_and_time':
                * n_peaks: int
                    Total number of peaks to select
                * peaks_locations: array
                    The locations of all the peaks, computed via localize_peaks
    
    {}
    Returns
    -------
    peaks: array
        Selected peaks.
    """

    selected_peaks = []
    
    if seed is not None:
        np.random.seed(seed)

    if method == 'uniform':

        params = {'select_per_channel' : False, 
                  'n_peaks' : None}

        params.update(method_kwargs)

        assert params['n_peaks'] is not None, "n_peaks should be defined!"

        if params['select_per_channel']:

            ## This method will randomly select max_peaks_per_channel peaks per channels
            for channel in np.unique(peaks['channel_ind']):
                peaks_indices = np.where(peaks['channel_ind'] == channel)[0]
                max_peaks = min(peaks_indices.size, params['n_peaks'])
                selected_peaks += [np.random.choice(peaks_indices, size=max_peaks, replace=False)]
        else:
            num_peaks = min(peaks.size, params['n_peaks'])
            selected_peaks = [np.random.choice(peaks.size, size=num_peaks, replace=False)]

    elif method == 'uniform_locations':

        params = {'peaks_locations' : None, 
                  'n_bins' : (50, 50),
                  'n_peaks' : None}

        params.update(method_kwargs)

        assert params['peaks_locations'] is not None, "peaks_locations should be defined!"
        assert params['n_peaks'] is not None, "n_peaks should be defined!"

        xmin, xmax = np.min(params['peaks_locations']['x']), np.max(params['peaks_locations']['x'])
        ymin, ymax = np.min(params['peaks_locations']['y']), np.max(params['peaks_locations']['y'])

        x_grid = np.linspace(xmin, xmax, params['n_bins'][0])
        y_grid = np.linspace(ymin, ymax, params['n_bins'][1])

        x_idx = np.searchsorted(x_grid, params['peaks_locations']['x'])
        y_idx = np.searchsorted(y_grid, params['peaks_locations']['y'])

        for i in range(params['n_bins'][0]):
            for j in range(params['n_bins'][1]):
                peaks_indices = np.where((x_idx == i) & (y_idx == j))[0]
                max_peaks = min(peaks_indices.size, params['n_peaks'])
                selected_peaks += [np.random.choice(peaks_indices, size=max_peaks, replace=False)]


    elif method in ['smart_sampling_amplitudes', 'smart_sampling_locations', 'smart_sampling_locations_and_time']:

        if method == 'smart_sampling_amplitudes':

            ## This method will try to select around n_peaks per channel but in a non uniform manner
            ## First, it will look at the distribution of the peaks amplitudes, per channel. 
            ## Once this distribution is known, it will sample from the peaks with a rejection probability
            ## such that the final distribution of the amplitudes, for the selected peaks, will be as
            ## uniform as possible. In a nutshell, the method will try to sample as homogenously as possible 
            ## from the space of all the peaks, using the amplitude as a discriminative criteria
            ## To do so, one must provide the noise_levels, detect_threshold used to detect the peaks, the 
            ## sign of the peaks, and the number of bins for the probability density histogram

            params = {'n_peaks' : None, 
                      'noise_levels' : None,
                      'select_per_channel' : False}

            params.update(method_kwargs)

            assert params['n_peaks'] is not None, "n_peaks should be defined!"
            assert params['noise_levels'] is not None, "Noise levels should be provided"

            histograms = {}

            if params['select_per_channel']:
                for channel in np.unique(peaks['channel_ind']):

                    peaks_indices = np.where(peaks['channel_ind'] == channel)[0]
                    sub_peaks = peaks[peaks_indices]
                    snrs = sub_peaks['amplitude'] / params['noise_levels'][channel]
                    max_peaks = min(peaks_indices.size, params['n_peaks'])
                    preprocessing = QuantileTransformer(output_distribution='uniform', n_quantiles = min(100, len(snrs)))
                    snrs = preprocessing.fit_transform(snrs[:, np.newaxis])
                    probabilities = np.random.rand(len(snrs))
                    selected_peaks += [np.random.permutation(np.where(snrs[:,0] < probabilities)[0])[:max_peaks]]

            else:
                snrs = peaks['amplitude'] / params['noise_levels'][peaks['channel_ind']]
                preprocessing = QuantileTransformer(output_distribution='uniform', n_quantiles=min(100, len(snrs)))
                snrs = preprocessing.fit_transform(snrs[:, np.newaxis])
                num_peaks = min(peaks.size, params['n_peaks'])
                probabilities = np.random.rand(len(snrs))
                selected_peaks = [np.random.permutation(np.where(snrs[:,0] < probabilities)[0])[:num_peaks]]

        elif method == 'smart_sampling_locations':

            ## This method will try to select around n_peaksbut in a non uniform manner
            ## First, it will look at the distribution of the positions. 
            ## Once this distribution is known, it will sample from the peaks with a rejection probability
            ## such that the final distribution of the amplitudes, for the selected peaks, will be as
            ## uniform as possible. In a nutshell, the method will try to sample as homogenously as possible 
            ## from the space of all the peaks, using the locations as a discriminative criteria
            ## To do so, one must provide the peaks locations, and the number of bins for the 
            ## probability density histogram

            params = {'peaks_locations' : None,
                      'n_peaks' : None}

            params.update(method_kwargs)

            assert params['n_peaks'] is not None, "n_peaks should be defined!"
            assert params['peaks_locations'] is not None, "peaks_locations should be d96efined!"

            nb_spikes = len(params['peaks_locations']['x'])
            preprocessing = QuantileTransformer(output_distribution='uniform', n_quantiles=min(100, nb_spikes))
            
            data = np.array([params['peaks_locations']['x'], params['peaks_locations']['y']]).T
            data = preprocessing.fit_transform(data)

            max_peaks = min(params['peaks_locations'].size, params['n_peaks'])

            probabilities = np.random.rand(len(data))
            data_x = data[:, 0] < probabilities

            probabilities = np.random.rand(len(data))
            data_y = data[:, 1] < probabilities

            selection = np.where(data_x * data_y)[0]

            selected_peaks = [np.random.permutation(selection)[:max_peaks]]

        elif method == 'smart_sampling_locations_and_time':

            ## This method will try to select around n_peaksbut in a non uniform manner
            ## First, it will look at the distribution of the positions. 
            ## Once this distribution is known, it will sample from the peaks with a rejection probability
            ## such that the final distribution of the amplitudes, for the selected peaks, will be as
            ## uniform as possible. In a nutshell, the method will try to sample as homogenously as possible 
            ## from the space of all the peaks, using the locations as a discriminative criteria
            ## To do so, one must provide the peaks locations, and the number of bins for the 
            ## probability density histogram

            params = {'peaks_locations' : None,
                      'n_peaks' : None}

            params.update(method_kwargs)

            assert params['n_peaks'] is not None, "n_peaks should be defined!"
            assert params['peaks_locations'] is not None, "peaks_locations should be defined!"

            nb_spikes = len(params['peaks_locations']['x'])
            preprocessing = QuantileTransformer(output_distribution='uniform', n_quantiles=min(100, nb_spikes))
            data = np.array([params['peaks_locations']['x'], params['peaks_locations']['y'], peaks['sample_ind']]).T
            data = preprocessing.fit_transform(data)

            max_peaks = min(params['peaks_locations'].size, params['n_peaks'])

            probabilities = np.random.rand(len(data))
            data_x = data[:, 0] < probabilities

            probabilities = np.random.rand(len(data))
            data_y = data[:, 1] < probabilities

            probabilities = np.random.rand(len(data))
            data_t = data[:, 2] < probabilities

            selection = np.where(data_x * data_y * data_t)[0]

            selected_peaks = [np.random.permutation(selection)[:max_peaks]]

    else:

        raise NotImplementedError(f"No method {method} for peaks selection")

    selected_peaks = peaks[np.concatenate(selected_peaks)]
    selected_peaks = selected_peaks[np.argsort(selected_peaks['sample_ind'])]

    return selected_peaks