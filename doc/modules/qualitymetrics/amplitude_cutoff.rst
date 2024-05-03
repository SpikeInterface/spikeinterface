.. _amp_cutoff:

Amplitude cutoff (:code:`amplitude_cutoff`)
===========================================

Calculation
-----------

First, all spike amplitudes for one unit are collapsed across time to create a histogram, then the histogram is smoothed using a 1D Gaussian filter. For some units, the amplitude histogram will not fall off gracefully to zero at the lower end, indicating that some spikes were likely missed by the sorter. To calculate the approximate fraction of missing spikes, we look at the height of the lowest amplitude bin, and count the number of spikes at the upper end of the distribution with amplitudes above a bin of similar height. The ratio of this count to the total number of spikes yields the amplitude cutoff.

Expectation and use
-------------------

This metric can be considered an estimate of the fraction of false negatives (missed spikes) during the intervals over which a unit is successfully detected. Therefore, smaller amplitude cutoff values tend to indicate higher quality units. However, the actual fraction of missed spikes may be much higher if the unit is not tracked for a portion of the recording session. For this reason, amplitude cutoff should be combined with a metric such as presence ratio to more accurately estimate each unit's overall completeness.

The calculation works best when the amplitude distribution is *symmetric* and has a *single peak*. Importantly, these assumptions likely do not hold if there is substantial drift. In this case, amplitude cutoff can be computed separately for different chunks of the recording, yielding a more accurate estimate of the fraction of spikes missing from each chunk.

**IMPORTANT:** When applying this metric, it is critical to know how the spike amplitudes were calculated. Template-based sorting algorithms such as Kilosort output the template scaling factors that are used to fit particular templates to each spike waveform. By multiplying the template amplitude by the scaling factor, one can approximate the original amplitude of each spike. However, this amplitude is not guaranteed to match the true amplitude. Amplitude cutoffs computed from the template scaling factors (`amplitudes.npy` in the Kilosort output) tend to be much higher than when using actual spike amplitudes extracted from the raw data. SpikeInterface uses amplitudes calculated from the raw data, but several large-scale electrophysiology surveys (such as those from the Allen Institute) use the template scaling factors. It's important to know which method was used in order to compare amplitude cutoff values across studies.

Example code
------------

.. code-block:: python

	import spikeinterface.qualitymetrics as sqm

	# Combine sorting and recording into a sorting_analyzer
	# It is also recommended to run sorting_analyzer.compute(input="spike_amplitudes")
	# in order to use amplitudes from all spikes
	fraction_missing = sqm.compute_amplitude_cutoffs(sorting_analyzer=sorting_analyzer, peak_sign="neg")
	# fraction_missing is a dict containing the unit IDs as keys,
	# and their estimated fraction of missing spikes as values.

Reference
---------

.. autofunction:: spikeinterface.qualitymetrics.misc_metrics.compute_amplitude_cutoffs


Links to original implementations
---------------------------------

* From the `AllenInstitute <https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/metrics.py#L219/>`_

* From the `AllenSDK <https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html#Amplitude-cutoff>`_

Literature
----------

A version of this metric was first suggested by [Hill]_. This paper suggested fitting a Gaussian distribution to the amplitude histogram to calculate the fraction of missing spikes, but this is not part of the SpikeInterface implementation. Instead, the upper end of the amplitude distribution is used to estimate the size of the lower end of the distribution.
