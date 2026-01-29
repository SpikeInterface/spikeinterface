Metrics module
==============


The :py:mod:`~spikeinterface.metrics` module includes functions to compute various metrics related to spike sorting.

Currently, it contains the following submodules:

- **template metrics**: Computes commonly used waveform/template metrics.
- **quality metrics**: Computes a variety of quality metrics to assess the goodness of spike sorting outputs.
- **spiketrain metrics**: Computes metrics based on spike train statistics and correlogram shapes.


All metrics extensions inherit from the :py:class:`~spikeinterface.core.analyzer_extension_core.BaseMetricExtension`
base class, which provides a common interface for computing and retrieving metrics and has convenience method to access
metric information. For example, you can get the list of available metrics using the and their descriptions with:

.. code-block:: python

    from spikeinterface.metrics import ComputeTemplateMetrics
    # ComputeTemplateMetrics inherits from BaseMetricExtension

    available_metric_columns = ComputeTemplateMetrics.get_metric_columns()
    print(f"Available metric columns: ")
    print(available_metric_columns)


.. code-block::

    Available metric columns:
    [
        'peak_to_trough_duration',
        'half_width',
        'repolarization_slope',
        'recovery_slope',
        'num_positive_peaks',
        'num_negative_peaks',
        'main_to_next_peak_duration',
        'peak_before_to_trough_ratio',
        'peak_after_to_trough_ratio',
        'peak_before_to_peak_after_ratio',
        'main_peak_to_trough_ratio',
        'trough_width',
        'peak_before_width',
        'peak_after_width',
        'waveform_baseline_flatness',
        'velocity_above',
        'velocity_below',
        'exp_decay',
        'spread'
    ]


.. code-block:: python

    metric_descriptions = ComputeTemplateMetrics.get_metric_column_descriptions()
    print("Metric descriptions: ")
    print(metric_descriptions)


.. code-block:: bash

    Metric descriptions:
    {
        'peak_to_trough_duration': 'Duration in seconds between the trough (minimum) and the peak (maximum) of the spike waveform.',
        'half_width': 'Duration in s at half the amplitude of the trough (minimum) of the spike waveform.',
        'repolarization_slope': 'Slope of the repolarization phase of the spike waveform, between the trough (minimum) and return to baseline in uV/s.',
        'recovery_slope': 'Slope of the recovery phase of the spike waveform, after the peak (maximum) returning to baseline in uV/s.',
        'num_positive_peaks': 'Number of positive peaks in the template',
        'num_negative_peaks': 'Number of negative peaks (troughs) in the template',
        'main_to_next_peak_duration': 'Duration in seconds from main extremum to next extremum.',
        'peak_before_to_trough_ratio': 'Ratio of peak before amplitude to trough amplitude',
        'peak_after_to_trough_ratio': 'Ratio of peak after amplitude to trough amplitude',
        'peak_before_to_peak_after_ratio': 'Ratio of peak before amplitude to peak after amplitude',
        'main_peak_to_trough_ratio': 'Ratio of main peak amplitude to trough amplitude',
        'trough_width': 'Width of the main trough in seconds',
        'peak_before_width': 'Width of the main peak before trough in seconds',
        'peak_after_width': 'Width of the main peak after trough in seconds',
        'waveform_baseline_flatness': 'Ratio of max baseline amplitude to max waveform amplitude. Lower = flatter baseline.',
        'velocity_above': 'Velocity of the spike propagation above the max channel in um/ms',
        'velocity_below': 'Velocity of the spike propagation below the max channel in um/ms',
        'exp_decay': 'Spatial decay of the template amplitude over distance from the extremum channel (1/um). Uses exponential or linear fit based on linear_fit parameter.',
        'spread': 'Spread of the template amplitude in um, calculated as the distance between channels whose templates exceed the spread_threshold.'
    }




.. toctree::
    :caption: Metrics submodules
    :maxdepth: 1

    metrics/template_metrics
    metrics/quality_metrics
    metrics/spiketrain_metrics
