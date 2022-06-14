Cumulative drift (not yet implemented)
======================================

Calculation
-----------

Geometric positions and times of spikes within the cluster are estimated.
The Allen Institute implementation assumes linear and equally spaced arrangement of channels.
Over the duration of the recording, the drift observed in positions of spikes is calculated in intervals.
Drifts between intervals is summed and this is the cumulative drift.

This metric is not yet implemented in the current SpikeInterface version. 

Expectation and use
-------------------

Reference
---------

Links to source code
--------------------

From the `AllenInstitute <https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/quality_metrics/metrics.py#L465/>`_

Literature
----------


Citations
---------
