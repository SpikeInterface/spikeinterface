"""Sorting components: peak selection"""

import numpy as np
import scipy



def select_peaks(peaks, method='uniform', seed=None, **method_kwargs):

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
                    If True, the selection is done on a per channel basis (default)
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
                * detect_threshold : int
                    The detection threshold 
                * peak_sign: string
                    If the peaks are detected as negative, positive, or both 
                * n_bins: int
                    The number of bins used to estimate the distributions at each channel
                * n_peaks: int
                    If select_per_channel is True, this is the number of peaks per channels,
                    otherwise this is the total number of peaks
                * select_per_channel: bool
                    If True, the selection is done on a per channel basis (default)
            'smart_sampling_locations':
                * n_bins: int
                    The number of bins used to estimate the distributions on each dimensions
                * n_peaks: int
                    Total number of peaks to select
                * peaks_locations: array
                    The locations of all the peaks, computed via localize_peaks
            'smart_sampling_locations_and_time':
                * n_bins: int
                    The number of bins used to estimate the distributions on each dimensions
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

        params = {'select_per_channel' : True, 
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

        def reject_rate(x, d, a, target, n_bins):
            return (np.mean(n_bins*a*np.clip(1 - d*x, 0, 1)) - target)**2

        def get_valid_indices(params, snrs, exponent=1, n_bins=None):
            if n_bins is None:
                n_bins = params['n_bins']
            bins = np.linspace(snrs.min(), snrs.max(), n_bins)
            x, y = np.histogram(snrs, bins=bins)
            histograms = {'probability' : x/x.sum(), 'snrs' : y[1:]}
            indices = np.searchsorted(histograms['snrs'], snrs)

            probabilities = histograms['probability']
            z = probabilities[probabilities > 0]
            c = 1.0 / np.min(z)
            d = np.ones(len(probabilities))
            d[probabilities > 0] = 1. / (c * z)
            d = np.minimum(1, d)
            d /= np.sum(d)
            twist = np.sum(probabilities * d)
            factor = twist * c

            target_rejection = (1 - max(0, params['n_peaks']/len(indices)))**exponent
            res = scipy.optimize.fmin(reject_rate, factor, args=(d, probabilities, target_rejection, n_bins), disp=False)
            rejection_curve = np.clip(1 - d*res[0], 0, 1)

            acceptation_threshold = rejection_curve[indices]
            valid_indices = acceptation_threshold < np.random.rand(len(indices))

            return valid_indices

        if method == 'smart_sampling_amplitudes':

            ## This method will try to select around n_peaks per channel but in a non uniform manner
            ## First, it will look at the distribution of the peaks amplitudes, per channel. 
            ## Once this distribution is known, it will sample from the peaks with a rejection probability
            ## such that the final distribution of the amplitudes, for the selected peaks, will be as
            ## uniform as possible. In a nutshell, the method will try to sample as homogenously as possible 
            ## from the space of all the peaks, using the amplitude as a discriminative criteria
            ## To do so, one must provide the noise_levels, detect_threshold used to detect the peaks, the 
            ## sign of the peaks, and the number of bins for the probability density histogram

            params = {'detect_threshold' : 5, 
                      'peak_sign' : 'neg',
                      'n_bins' : 50, 
                      'n_peaks' : None, 
                      'noise_levels' : None,
                      'select_per_channel' : True}

            params.update(method_kwargs)

            assert params['n_peaks'] is not None, "n_peaks should be defined!"
            assert params['noise_levels'] is not None, "Noise levels should be provided"

            histograms = {}

            if params['select_per_channel']:
                for channel in np.unique(peaks['channel_ind']):

                    peaks_indices = np.where(peaks['channel_ind'] == channel)[0]
                    sub_peaks = peaks[peaks_indices]
                    snrs = sub_peaks['amplitude'] / params['noise_levels'][channel]
                    valid_indices = get_valid_indices(params, snrs)
                    selected_peaks += [peaks_indices[valid_indices]]
            else:

                snrs = peaks['amplitude'] / params['noise_levels'][peaks['channel_ind']]
                valid_indices = get_valid_indices(params, snrs)
                valid_indices,  = np.where(valid_indices)
                selected_peaks = [valid_indices]

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
                      'n_bins' : (50, 50),
                      'n_peaks' : None}

            params.update(method_kwargs)

            assert params['n_peaks'] is not None, "n_peaks should be defined!"
            assert params['peaks_locations'] is not None, "peaks_locations should be defined!"

            def f(x):
                valid_indices_x = get_valid_indices(params, params['peaks_locations']['x'], n_bins=params['n_bins'][0], exponent=x)
                valid_indices_y = get_valid_indices(params, params['peaks_locations']['y'], n_bins=params['n_bins'][1], exponent=x)
                valid_indices,  = np.where(valid_indices_x*valid_indices_y)
                return np.abs(1-len(valid_indices)/params['n_peaks'])**2

            exponents = np.arange(1, 100)
            data = [f(exponent) for exponent in exponents]
            best_exponent = exponents[np.argmin(data)]

            valid_indices_x = get_valid_indices(params, params['peaks_locations']['x'], n_bins=params['n_bins'][0], exponent=best_exponent)
            valid_indices_y = get_valid_indices(params, params['peaks_locations']['y'], n_bins=params['n_bins'][1], exponent=best_exponent)

            valid_indices,  = np.where(valid_indices_x*valid_indices_y)
            selected_peaks = [valid_indices]

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
                      'n_bins' : (50, 50, 50),
                      'n_peaks' : None}

            params.update(method_kwargs)

            assert params['n_peaks'] is not None, "n_peaks should be defined!"
            assert params['peaks_locations'] is not None, "peaks_locations should be defined!"

            def f(x):
                valid_indices_x = get_valid_indices(params, params['peaks_locations']['x'], n_bins=params['n_bins'][0], exponent=x)
                valid_indices_y = get_valid_indices(params, params['peaks_locations']['y'], n_bins=params['n_bins'][1], exponent=x)
                valid_indices_time = get_valid_indices(params, peaks['sample_ind'], n_bins=params['n_bins'][2], exponent=x)
                valid_indices,  = np.where(valid_indices_x*valid_indices_y*valid_indices_time)
                return np.abs(1-len(valid_indices)/params['n_peaks'])**2

            exponents = np.arange(1, 100)
            data = [f(exponent) for exponent in exponents]
            best_exponent = exponents[np.argmin(data)]

            valid_indices_x = get_valid_indices(params, params['peaks_locations']['x'], n_bins=params['n_bins'][0], exponent=best_exponent)
            valid_indices_y = get_valid_indices(params, params['peaks_locations']['y'], n_bins=params['n_bins'][1], exponent=best_exponent)
            valid_indices_time = get_valid_indices(params, peaks['sample_ind'], n_bins=params['n_bins'][2], exponent=best_exponent)

            valid_indices,  = np.where(valid_indices_x*valid_indices_y*valid_indices_time)
            selected_peaks = [valid_indices]

    else:

        raise NotImplementedError(f"No method {method} for peaks selection")

    selected_peaks = peaks[np.concatenate(selected_peaks)]
    selected_peaks = selected_peaks[np.argsort(selected_peaks['sample_ind'])]

    return selected_peaks