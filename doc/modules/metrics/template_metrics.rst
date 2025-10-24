Template Metrics
================

This extension computes commonly used waveform/template metrics.
By default, the following metrics are computed:

* "peak_to_valley": duration in :math:`s` between negative and positive peaks
* "halfwidth": duration in :math:`s` at 50% of the amplitude
* "peak_to_trough_ratio": ratio between negative and positive peaks
* "recovery_slope": speed to recover from the negative peak to 0
* "repolarization_slope": speed to repolarize from the positive peak to 0
* "num_positive_peaks": the number of positive peaks
* "num_negative_peaks": the number of negative peaks

The units of :code:`recovery_slope` and :code:`repolarization_slope` depend on the
input. Voltages are based on the units of the template. By default this is :math:`\mu V`
but can be the raw output from the recording device (this depends on the
:code:`return_in_uV` parameter, read more here: :ref:`modules/core:SortingAnalyzer`).
Distances are in :math:`\mu m` and times are in seconds. So, for example, if the
templates are in units of :math:`\mu V` then: :code:`repolarization_slope` is in
:math:`mV / s`; :code:`peak_to_trough_ratio` is in :math:`\mu m` and the
:code:`halfwidth` is in :math:`s`.

Optionally, the following multi-channel metrics can be computed by setting:
:code:`include_multi_channel_metrics=True`

* "velocity_above": the velocity in :math:`\mu m/s` above the max channel of the template
* "velocity_below": the velocity in :math:`\mu m/s` below the max channel of the template
* "exp_decay": the exponential decay in :math:`\mu m` of the template amplitude over distance
* "spread": the spread in :math:`\mu m` of the template amplitude over distance

.. figure:: ../../images/1d_waveform_features.png

    Visualization of template metrics. Image from `ecephys_spike_sorting <https://github.com/AllenInstitute/ecephys_spike_sorting/tree/v0.2/ecephys_spike_sorting/modules/mean_waveforms>`_
    from the Allen Institute.


.. code-block:: python

    tm = sorting_analyzer.compute(input="template_metrics", include_multi_channel_metrics=True)


For more information, see :py:func:`~spikeinterface.postprocessing.compute_template_metrics`
