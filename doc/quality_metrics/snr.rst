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

    import spikeinterface.toolkit as st

    # Make recording, sorting and wvf_extractor object for your data.

    SNRs = st.compute_snrs(wvf_extractor)
    # SNRs is a dict containing the units' ID as keys and their SNR as values.

Links to source code
--------------------

From `SpikeInterface <https://github.com/SpikeInterface/spikeinterface/blob/85244cd686bfe2a80649246ea1e29120930cb9c7/spikeinterface/toolkit/qualitymetrics/misc_metrics.py#L130>`_

From the `AllenSDK <https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html#SNR>`_


References
----------

.. automodule:: spikeinterface.toolkit.qualitymetrics.misc_metrics

    .. autofunction:: compute_snrs

Literature
----------

Presented by Lemon_ and useful initial discussion by Jackson_.

Citations
---------

.. [Jackson] Jadin Jackson, Neil Schmitzer-Torbert, K.D. Harris, and A.D. Redish. Quantitative assessment of extracellular multichannel recording quality using measures of cluster separation. Soc Neurosci Abstr, 518, 01 2005.

.. [Lemon] R. Lemon. Methods for neuronal recording in conscious animals. IBRO Handbook Series, 4:56–60, 1984.