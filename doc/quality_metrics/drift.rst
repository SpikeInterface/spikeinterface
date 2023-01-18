Drift metrics (:code:`drift_ptp`, :code:`drift_std`, :code:`drift_mad`)
=======================================================================

Calculation
-----------

Geometric positions and times of spikes within the cluster are estimated.
Over the duration of the recording, the drift observed in positions of spikes is calculated in intervals, with respect 
to the overall median positions over the entire recording. These are referred to as "drift signals".

* The drift_ptp is the peak-to-peak of the drift signal for each unit.
* The drift_std is the standard deviation of the drift signal for each unit.
* The drift_mad is the median absolute deviation of the drift signal for each unit.


The SpikeInterface implementation differes from the original Allen because it uses spike location estimates 
(using :py:func:`~spikeinterface.postprocessing.compute_spike_locations()` - either center of mass or monopolar 
triangulation), instead of the center of mass of the first PC projection.
In addition the Allen Institute implementation assumes linear and equally spaced arrangement of channels.

Finally, the original "cumulative_drift" and "max_drift" metrics have been refactored/modified 
for the following reasons:
- "max_drift" is calculated with the the peak-to-peak, so it's been renamed "drift_ptp"
- | "cumulative_drift" sums the absolute value of the drift signal for each interval. This makes it very sensitive to 
  | the number of bins (and hence the recording duration)! The "drift_std" and "drift_mad", instead, are measures of 
  | the dispersion of the drift signal and are insensitive to the recording duration.


Expectation and use
-------------------

Drift metrics represents how much, in um, a unit has moved over the recording.
Larger values indicate more "drifty" units, possibly of lower quality.

Example code
------------

.. code-block:: python

	import spikeinterface.qualitymetrics as qm

	# Make recording, sorting and wvf_extractor objects for your data.
	# It is also required to run `compute_spike_locations(wvf_extractor)`
    # (if missing, values will be NaN)

	drift_ptps, drift_stds, drift_mads = qm.compute_amplitudes_cutoff(wvf_extractor, peak_sign="neg")
	# drift_ptps, drift_stds, and drift_mads are dict containing the units' ID as keys,
	# and their metrics as values.


Reference
---------

.. automodule:: spikeinterface.qualitymetrics.misc_metrics

	.. autofunction:: compute_drift_metrics

Links to source code
--------------------

From the `AllenInstitute <https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/metrics.py#L548/>`_

From `SpikeInterface <https://github.com/SpikeInterface/spikeinterface/blob/master/spikeinterface/qualitymetrics/misc_metrics.py#L259/>`_


Literature
----------

First introduced in Siegle_ and modified by the SpikeInterface Team.

Citations
---------

.. [Siegle] Siegle, Joshua H., et al. “Survey of spiking in the mouse visual system reveals functional hierarchy.” Nature 592.7852 (2021): 86-92.
