Inter-spike-interval (ISI) violations
=====================================



Calculation
-----------

Neurons have a refractory period after a spiking event during which they cannot fire again.
Inter-spike-interval (ISI) violations refers to the rate of refractory period violations Hill_.

The calculation works under the assumption that the contaminant events happen randomly, or comes from another neuron that is not correlated with our unit. A correlation will lead to an over-estimation of the contamination, whereas an anti-correlation will lead to an under-estimation.

Different formulas were developped over the years.


Calculation from the Hill_ paper
--------------------------------

The following quantities are required:

- :math:`ISI_t` : biological threshold for ISI violation.
- :math:`ISI_{min}`: minimum ISI threshold enforced by the data recording system used.
- :math:`ISI_s$` : the array of ISI violations which are observed in the unit's spike train.
- :math:`\#`: denote count.

The threshold for ISI violations is the biological ISI threshold, :math:`ISI_t`, minus the minimum ISI threshold, :math:`ISI_{min}` enforced by the data recording system used.
The array of inter-spike-intervals observed in the unit's spike train, :math:`ISI_s$`, is used to identify the count (:math:`\#`) of observed ISI's below this threshold.
For a recording with a duration of :math:`T_r` seconds, and a unit with :math:`N_s` spikes, the rate of ISI violations is:

.. math::

    \textrm{ISI violations} = \frac{ \#( ISI_s < ISI_t) T_r  }{ 2  N_s^2  (ISI_t - ISI_{min}) }


Calculation from the Llobet_ paper
----------------------------------

The following quantities are required:

- :math:`T` the duration of the recording.
- :math:`N` the number of spikes in the unit's spike train.
- :math:`t_r` the duration of the unit's refractory period.
- :math:`n_v` the number of violations of the refractory period.

The estimated contamination :math:`C` can be calculated with 2 extreme scenarios. In the first one, the contaminant spikes are completely random (or come from an infinite number of other neurons). In the second one, the contaminant spikes come from a single other neuron:

.. math::

    C = \frac{FP}{TP + FP} \approx \begin{cases}
        1 - \sqrt{1 - \frac{n_v T}{N^2 t_r}} \text{ for the case of random contamination} \\
        \frac{1}{2} \left( 1 - \sqrt{1 - \frac{2 n_v T}{N^2 t_r}} \right) \text{ for the case of 1 contaminant neuron}
    \end{cases}

Where :math:`TP` is the number of true positives (detected spikes that come from the neuron) and :math:`FP` is the number of false positives (detected spikes that don't come from the neuron).

Expectation and use
-------------------

ISI violations identifies unit contamination - a high value indicates a highly contaminated unit.
Despite being a ratio, ISI violations can exceed 1 (or become a complex number in the Llobet_ formula). This is usually due to the contaminant events being correlated with our neuron, and their number are greater than a purely random spike train.

Example code
------------

Without SpikeInterface:

.. code-block:: python

    spike_train = ...   # The spike train of our unit
    t_r = ...           # The refractory period of our unit
    n_v = np.sum(np.diff(spike_train) < t_r)

    # Use the formula you want here

With SpikeInterface:

.. code-block:: python

    import spikeinterface.toolkit as st

    # Make recording, sorting and wvf_extractor object for your data.

    isi_violations_ratio, isi_violations_rate, isi_violations_count = st.compute_isi_violations(wvf_extractor, isi_threshold_ms=1.0)

Links to source code
--------------------

From `SpikeInterface <https://github.com/SpikeInterface/spikeinterface/blob/ae679aff788a6dd4d8e7783e1f72ec7e550c1bf9/spikeinterface/toolkit/qualitymetrics/misc_metrics.py#L169>`_

From `Lussac <https://github.com/BarbourLab/lussac/blob/main/postprocessing/utils.pyx#L365>`_

From the `AllenSDK <https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html#ISI-violations>`_


Literature
----------

Introduced by [Hill]_ (2011).

.. [Hill] Hill, Daniel N., Samar B. Mehta, and David Kleinfeld. “Quality Metrics to Accompany Spike Sorting of Extracellular Signals.” The Journal of neuroscience 31.24 (2011): 8699–8705. Web.


Also described by [Llobet]_ (2022)

.. [Llobet] Llobet Victor, Wyngaard Aurélien and Barbour Boris. “Automatic post-processing and merging of multiple spike-sorting analyses with Lussac“. BioRxiv (2022).
