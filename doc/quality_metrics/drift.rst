Drift metrics (:code:`maximum_drift`, :code:`cumulative_drift`)
===============================================================

Calculation
-----------

Geometric positions and times of spikes within the cluster are estimated.
Over the duration of the recording, the drift observed in positions of spikes is calculated in intervals, with respect 
to the median positions in the first interval.
Drifts between intervals is summed and this is the cumulative drift.
The maximum drift is the estimated peak-to-peak of the drift.

The SpikeInterface implementation differes from the original Allen because it uses spike location estimates 
(using :py:func:`~spikeinterface.postprocessing.compute_spike_locations()` - either center of mass or monopolar 
triangulation), instead of the center of mass of the first PC projection.
In addition the Allen Institute implementation assumes linear and equally spaced arrangement of channels.


Expectation and use
-------------------

Both maximum and cumulative drift represents how much, in um, a unit has moved over the recording.
Larger values indicate more "drifty" units, possibly of lower quality.

Example code
------------

.. code-block:: python

	import spikeinterface.qualitymetrics as qm

	# Make recording, sorting and wvf_extractor objects for your data.
	# It is also required to run `compute_spike_locations(wvf_extractor)`
    # (if missing, values will be NaN)

	maximum_drift, cumulative_drift = qm.compute_amplitudes_cutoff(wvf_extractor, peak_sign="neg")
	# maximum_drift and cumulative_drift are dict containing the units' ID as keys,
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

First introduced in Siegle_.

Citations
---------

.. [Siegle] Siegle, Joshua H., et al. “Survey of spiking in the mouse visual system reveals functional hierarchy.” Nature 592.7852 (2021): 86-92.
