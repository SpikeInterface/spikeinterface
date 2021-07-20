API
===

Module :mod:`spikeinterface.core`
---------------------------------
.. automodule:: spikeinterface.core

    .. autofunction:: load_extractor

    .. autoclass:: BaseRecording

    .. autoclass:: BaseSorting

    .. autoclass:: BaseEvent

    .. autoclass:: BinaryRecordingExtractor

    .. autoclass:: NpzSortingExtractor

    .. autoclass:: NumpyRecording

    .. autoclass:: NumpySorting


    .. autofunction:: set_global_tmp_folder

    .. autofunction:: set_global_dataset_folder




Module :mod:`spikeinterface.extractors`
---------------------------------------
.. automodule:: spikeinterface.extractors

    .. autofunction:: toy_example


Module :mod:`spikeinterface.toolkit`
--------------------------------------

Preprocessing
~~~~~~~~~~~~~
.. automodule:: spikeinterface.toolkit.preprocessing

    .. autofunction:: filter
    .. autofunction:: bandpass_filter
    .. autofunction:: notch_filter
    .. autofunction:: normalize_by_quantile
    .. autofunction:: scale
    .. autofunction:: center
    .. autofunction:: whiten
    .. autofunction:: rectify
    .. autofunction:: blank_staturation
    .. autofunction:: clip
    .. autofunction:: common_reference
    .. autofunction:: remove_artifacts
    .. autofunction:: remove_bad_channels


Postprocessing
~~~~~~~~~~~~~~
.. automodule:: spikeinterface.toolkit.postprocessing

    .. autofunction:: get_template_amplitudes
    .. autofunction:: get_template_extremum_channel
    .. autofunction:: get_template_extremum_channel_peak_shift
    .. autofunction:: get_template_extremum_amplitude
    .. autofunction:: get_template_best_channels
    .. autofunction:: compute_unit_centers_of_mass
    .. autofunction:: calculate_template_metrics
    .. autofunction:: export_to_phy
    .. autofunction:: compute_principal_components
    .. autofunction:: get_unit_amplitudes


Quality metrics
~~~~~~~~~~~~~~~
.. automodule:: spikeinterface.toolkit.qualitymetrics

    .. autofunction:: compute_quality_metrics




Module :mod:`spikeinterface.sorters`
--------------------------------------
.. automodule:: spikeinterface.sorters

    .. autofunction:: available_sorters
    .. autofunction:: installed_sorters
    .. autofunction:: get_default_params
    .. autofunction:: print_sorter_versions
    .. autofunction:: get_sorter_description
    .. autofunction:: run_sorter
    .. autofunction:: run_sorters


Module :mod:`spikeinterface.comparison`
---------------------------------------
.. automodule:: spikeinterface.comparison

    .. autofunction:: compare_two_sorters
    .. autofunction:: compare_multiple_sorters
    .. autofunction:: compare_sorter_to_ground_truth

    .. autoclass:: GroundTruthComparison
        :members:

    .. autoclass:: SymmetricSortingComparison
        :members:
        :undoc-members:

    .. autoclass:: GroundTruthStudy
        :members:
        :undoc-members:


Module :mod:`spikeinterface.widgets`
--------------------------------------
.. automodule:: spikeinterface.widgets

    .. autofunction:: plot_timeseries
    .. autofunction:: plot_rasters
    .. autofunction:: plot_probe_map
    .. autofunction:: plot_unit_waveforms
    .. autofunction:: plot_unit_templates
    .. autofunction:: plot_amplitudes_timeseries
    .. autofunction:: plot_amplitudes_distribution
    .. autofunction:: plot_principal_component
    .. autofunction:: plot_confusion_matrix
    .. autofunction:: plot_agreement_matrix
    .. autofunction:: plot_multicomp_graph
    .. autofunction:: plot_multicomp_agreement
    .. autofunction:: plot_multicomp_agreement_by_sorter
    .. autofunction:: plot_comparison_collision_pair_by_pair
    .. autofunction:: plot_comparison_collision_by_similarity
    .. autofunction:: plot_sorting_performance


Module :mod:`spikeinterface.exporters`
--------------------------------------
.. automodule:: spikeinterface.exporters

    .. autofunction:: export_to_phy
    .. autofunction:: export_report


Module :mod:`spikeinterface.sortingcomponents`
--------------------------------------
.. automodule:: spikeinterface.sortingcomponents

    .. autofunction:: detect_peaks
    .. autofunction:: localize_peaks
