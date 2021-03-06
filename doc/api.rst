API
===

Module :mod:`spikeinterface.extractors`
---------------------------------------
.. automodule:: spikeextractors

    .. autoclass:: spikeextractors.baseextractor.BaseExtractor
        :members:
    .. autoclass:: RecordingExtractor
        :members:
    .. autoclass:: SortingExtractor
        :members:

    .. autoclass:: SubRecordingExtractor
        :members:

    .. autoclass:: SubSortingExtractor
        :members:

    .. autoclass:: MultiRecordingChannelExtractor
        :members:

    .. autoclass:: MultiRecordingTimeExtractor
        :members:

    .. autoclass:: MultiSortingExtractor
        :members:

    .. autofunction:: load_extractor_from_dict
    .. autofunction:: load_extractor_from_json
    .. autofunction:: load_extractor_from_pickle
    .. autofunction:: load_probe_file
    .. autofunction:: save_to_probe_file
    .. autofunction:: write_to_binary_dat_format
    .. autofunction:: get_sub_extractors_by_property



Module :mod:`spikeinterface.toolkit`
--------------------------------------

Preprocessing
~~~~~~~~~~~~~
.. automodule:: spiketoolkit.preprocessing

    .. autofunction:: bandpass_filter
    .. autofunction:: blank_saturation
    .. autofunction:: clip
    .. autofunction:: normalize_by_quantile
    .. autofunction:: notch_filter
    .. autofunction:: rectify
    .. autofunction:: remove_artifacts
    .. autofunction:: remove_bad_channels
    .. autofunction:: resample
    .. autofunction:: transform
    .. autofunction:: whiten
    .. autofunction:: common_reference


Postprocessing
~~~~~~~~~~~~~~
.. automodule:: spiketoolkit.postprocessing

    .. autofunction:: get_unit_waveforms
    .. autofunction:: get_unit_templates
    .. autofunction:: get_unit_amplitudes
    .. autofunction:: get_unit_max_channels
    .. autofunction:: set_unit_properties_by_max_channel_properties
    .. autofunction:: compute_unit_pca_scores
    .. autofunction:: export_to_phy
    .. autofunction:: compute_unit_template_features

Validation
~~~~~~~~~~~~~~
.. automodule:: spiketoolkit.validation

    .. autofunction:: compute_isolation_distances
    .. autofunction:: compute_isi_violations
    .. autofunction:: compute_snrs
    .. autofunction:: compute_amplitude_cutoffs
    .. autofunction:: compute_d_primes
    .. autofunction:: compute_drift_metrics
    .. autofunction:: compute_firing_rates
    .. autofunction:: compute_l_ratios
    .. autofunction:: compute_nn_metrics
    .. autofunction:: compute_num_spikes
    .. autofunction:: compute_presence_ratios
    .. autofunction:: compute_silhouette_scores
    .. autofunction:: compute_quality_metrics


Curation
~~~~~~~~~~~~~~
.. automodule:: spiketoolkit.curation

    .. autofunction:: threshold_amplitude_cutoffs
    .. autofunction:: threshold_d_primes
    .. autofunction:: threshold_drift_metrics
    .. autofunction:: threshold_firing_rates
    .. autofunction:: threshold_isi_violations
    .. autofunction:: threshold_isolation_distances
    .. autofunction:: threshold_l_ratios
    .. autofunction:: threshold_nn_metrics
    .. autofunction:: threshold_num_spikes
    .. autofunction:: threshold_presence_ratios
    .. autofunction:: threshold_silhouette_scores
    .. autofunction:: threshold_snrs

    .. autoclass:: CurationSortingExtractor
        :members:




Module :mod:`spikeinterface.sorters`
--------------------------------------
.. automodule:: spikesorters

    .. autofunction:: available_sorters
    .. autofunction:: get_default_params
    .. autofunction:: run_sorter
    .. autofunction:: run_sorters



Module :mod:`spikeinterface.comparison`
---------------------------------------
.. automodule:: spikecomparison

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
.. automodule:: spikewidgets

    .. autofunction:: plot_timeseries
    .. autofunction:: plot_electrode_geometry
    .. autofunction:: plot_spectrum
    .. autofunction:: plot_spectrogram
    .. autofunction:: plot_activity_map
    .. autofunction:: plot_rasters
    .. autofunction:: plot_autocorrelograms
    .. autofunction:: plot_crosscorrelograms
    .. autofunction:: plot_isi_distribution
    .. autofunction:: plot_unit_waveforms
    .. autofunction:: plot_unit_templates
    .. autofunction:: plot_unit_template_maps
    .. autofunction:: plot_amplitudes_distribution
    .. autofunction:: plot_amplitudes_timeseries
    .. autofunction:: plot_confusion_matrix
    .. autofunction:: plot_agreement_matrix
    .. autofunction:: plot_sorting_performance
    .. autofunction:: plot_multicomp_graph
    .. autofunction:: plot_multicomp_agreement
    .. autofunction:: plot_multicomp_agreement_by_sorter
