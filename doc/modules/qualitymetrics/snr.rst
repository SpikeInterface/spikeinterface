Signal-to-noise ratio (:code:`snr`)
===================================

Calculation
-----------

- :math:`A_{\mu s}` : maximum amplitude of the mean spike waverform (on the best channel).
- :math:`\sigma_b` : standard deviation of the background noise on the same channel (usually computed via the `median absolute deviation <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_).

.. math::
    \textrm{SNR} = \frac{A_{\mu s}}{\sigma_b}

Expectation and use
-------------------

A high SNR unit has a signal which is greater in amplitude than the background noise and is likely to correspond to a neuron [Jackson]_, [Lemon]_.
A low SNR value (close to 0) suggests that the unit is highly contaminated by noise (type I error).


Example code
------------

Without SpikeInterface:

.. code-block:: python

    import numpy as np
    import scipy.stats

    data        # The data from your recording in shape (channel, time)
    mean_wvf    # The mean waveform of your unit in shape (channel, time)
    # If your data is filtered, then both data and mean_wvf need to be filtered the same.

    best_channel = np.argmax(np.max(np.abs(mean_wvf), axis=1))
    noise_level = scipy.stats.median_abs_deviation(data[best_channel], scale="normal")
    amplitude = np.max(np.abs(mean_wvf))

    SNR = amplitude / noise_level

With SpikeInterface:

.. code-block:: python

    import spikeinterface.qualitymetrics as sqm

    # Combining sorting and recording into a sorting_analzyer
    SNRs = sqm.compute_snrs(sorting_analzyer=sorting_analzyer)
    # SNRs is a dict containing the unit IDs as keys and their SNRs as values.

Links to original implementations
---------------------------------

* From the `AllenSDK <https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html#SNR>`_


References
----------

.. autofunction:: spikeinterface.qualitymetrics.misc_metrics.compute_snrs

Literature
----------

Presented by [Lemon]_ and useful initial discussion by [Jackson]_.
