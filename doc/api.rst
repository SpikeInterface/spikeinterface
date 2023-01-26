API
===

spikeinterface.core
-------------------
.. automodule:: spikeinterface.core

    .. autofunction:: load_extractor
    .. autoclass:: BaseRecording
        :members:
    .. autoclass:: BaseSorting
        :members:
    .. autoclass:: BaseSortingSegment
        :members:
    .. autoclass:: BaseEvent
    .. autoclass:: BinaryRecordingExtractor
    .. autofunction:: read_binary
    .. autoclass:: NpzSortingExtractor
    .. autoclass:: NumpyRecording
    .. autoclass:: NumpySorting
    .. autoclass:: ChannelSliceRecording
    .. autoclass:: UnitsSelectionSorting
    .. autoclass:: FrameSliceRecording
    .. autofunction:: append_recordings
    .. autofunction:: concatenate_recordings
    .. autofunction:: append_sortings
    .. autofunction:: load_extractor
    .. autofunction:: extract_waveforms
    .. autofunction:: load_waveforms
    .. autoclass:: WaveformExtractor
        .. automethod:: set_params
    .. autofunction:: download_dataset
    .. autofunction:: write_binary_recording
    .. autofunction:: set_global_tmp_folder
    .. autofunction:: set_global_dataset_folder
    .. autoclass:: ChunkRecordingExecutor
    .. autofunction:: get_random_data_chunks
    .. autofunction:: get_channel_distances
    .. autofunction:: get_closest_channels
    .. autofunction:: get_noise_levels
    .. autofunction:: get_chunk_with_margin
    .. autofunction:: get_template_amplitudes
    .. autofunction:: get_template_extremum_channel
    .. autofunction:: get_template_extremum_channel_peak_shift
    .. autofunction:: get_template_extremum_amplitude
    .. autofunction:: get_template_channel_sparsity
    


spikeinterface.extractors
-------------------------

NEO-based
~~~~~~~~~
.. automodule:: spikeinterface.extractors

    .. autofunction:: read_alphaomega
    .. autofunction:: read_alphaomega_event
    .. autofunction:: read_axona
    .. autofunction:: read_biocam
    .. autofunction:: read_blackrock
    .. autofunction:: read_ced
    .. autofunction:: read_intan
    .. autofunction:: read_kilosort
    .. autofunction:: read_maxwell
    .. autofunction:: read_mearec
    .. autofunction:: read_mcsraw
    .. autofunction:: read_neuralynx
    .. autofunction:: read_neuroscope
    .. autofunction:: read_nix
    .. autofunction:: read_openephys
    .. autofunction:: read_openephys_event
    .. autofunction:: read_plexon
    .. autofunction:: read_spike2
    .. autofunction:: read_spikegadgets
    .. autofunction:: read_spikeglx
    .. autofunction:: read_tdt


Non-NEO-based
~~~~~~~~~~~~~
.. automodule:: spikeinterface.extractors

    .. autofunction:: read_alf_sorting
    .. autofunction:: read_bids
    .. autofunction:: read_cbin_ibl
    .. autofunction:: read_combinato
    .. autofunction:: read_hdsort
    .. autofunction:: read_herdingspikes
    .. autofunction:: read_klusta
    .. autofunction:: read_mcsh5
    .. autofunction:: read_mda_recording
    .. autofunction:: read_mda_sorting
    .. autofunction:: read_nwb
    .. autofunction:: read_phy
    .. autofunction:: read_shybrid_recording
    .. autofunction:: read_shybrid_sorting
    .. autofunction:: read_spykingcircus
    .. autofunction:: toy_example
    .. autofunction:: read_tridesclous
    .. autofunction:: read_waveclust
    .. autofunction:: read_yass

Low-level classes
~~~~~~~~~~~~~~~~~

.. automodule:: spikeinterface.extractors

    .. autoclass:: AlphaOmegaRecordingExtractor
    .. autoclass:: AlphaOmegaEventExtractor
    .. autoclass:: AxonaRecordingExtractor
    .. autoclass:: BiocamRecordingExtractor
    .. autoclass:: BlackrockRecordingExtractor
    .. autoclass:: BlackrockSortingExtractor
    .. autoclass:: CedRecordingExtractor
    .. autoclass:: EDFRecordingExtractor
    .. autoclass:: IntanRecordingExtractor
    .. autoclass:: MaxwellRecordingExtractor
    .. autoclass:: MaxwellEventExtractor
    .. autoclass:: MaxwellEventSegment
    .. autoclass:: MEArecRecordingExtractor
    .. autoclass:: MEArecSortingExtractor
    .. autoclass:: MCSRawRecordingExtractor
    .. autoclass:: NeuralynxRecordingExtractor
    .. autoclass:: NeuralynxSortingExtractor
    .. autoclass:: NeuroScopeRecordingExtractor
    .. autoclass:: NeuroScopeSortingExtractor
    .. autoclass:: NixRecordingExtractor
    .. autoclass:: OpenEphysLegacyRecordingExtractor
    .. autoclass:: OpenEphysBinaryRecordingExtractor
    .. autoclass:: OpenEphysBinaryEventExtractor
    .. autoclass:: PlexonRecordingExtractor
    .. autoclass:: Spike2RecordingExtractor
    .. autoclass:: SpikeGadgetsRecordingExtractor
    .. autoclass:: SpikeGLXRecordingExtractor
    .. autoclass:: TdtRecordingExtractor

spikeinterface.preprocessing
----------------------------


.. automodule:: spikeinterface.preprocessing

    .. autofunction:: bandpass_filter
    .. autofunction:: blank_staturation
    .. autofunction:: center
    .. autofunction:: clip
    .. autofunction:: common_reference
    .. autofunction:: filter
    .. autofunction:: normalize_by_quantile
    .. autofunction:: notch_filter
    .. autofunction:: rectify
    .. autofunction:: remove_artifacts
    .. autofunction:: remove_bad_channels
    .. autofunction:: scale
    .. autofunction:: whiten



spikeinterface.postprocessing
-----------------------------

.. automodule:: spikeinterface.postprocessing

    .. autofunction:: localize_units
    .. autofunction:: get_template_metric_names
    .. autofunction:: compute_template_metrics
    .. autofunction:: compute_principal_components
    .. autofunction:: compute_spike_amplitudes
    .. autofunction:: compute_unit_locations
    .. autofunction:: compute_spike_locations
    .. autofunction:: compute_correlograms
    .. autofunction:: compute_template_similarity
    .. autofunction:: compute_correlograms
    .. autofunction:: compute_isi_histograms
    .. autofunction:: align_sorting


spikeinterface.qualitymetrics
-----------------------------

.. automodule:: spikeinterface.qualitymetrics

    .. autofunction:: compute_quality_metrics
    .. autofunction:: get_quality_metric_list


spikeinterface.sorters
----------------------
.. automodule:: spikeinterface.sorters

    .. autofunction:: available_sorters
    .. autofunction:: installed_sorters
    .. autofunction:: get_default_params
    .. autofunction:: print_sorter_versions
    .. autofunction:: get_sorter_description
    .. autofunction:: run_sorter
    .. autofunction:: run_sorters
    .. autofunction:: run_sorter_by_property

Low level
~~~~~~~~~
.. automodule:: spikeinterface.sorters

    .. autoclass:: BaseSorter


spikeinterface.comparison
-------------------------
.. automodule:: spikeinterface.comparison

    .. autofunction:: compare_two_sorters
    .. autofunction:: compare_multiple_sorters
    .. autofunction:: compare_sorter_to_ground_truth
    .. autofunction:: compare_templates
    .. autofunction:: compare_multiple_templates
    .. autofunction:: aggregate_performances_table
    .. autofunction:: create_hybrid_units_recording
    .. autofunction:: create_hybrid_spikes_recording

    .. autoclass:: GroundTruthComparison
        :members:

    .. autoclass:: SymmetricSortingComparison
        :members:
        :undoc-members:

    .. autoclass:: GroundTruthStudy
        :members:
        :undoc-members:

    .. autoclass:: MultiSortingComparison
        :members:


spikeinterface.widgets
----------------------
.. automodule:: spikeinterface.widgets

    .. autofunction:: plot_timeseries
    .. autofunction:: plot_rasters
    .. autofunction:: plot_probe_map
    .. autofunction:: plot_isi_distribution
    .. autofunction:: plot_crosscorrelograms
    .. autofunction:: plot_autocorrelograms
    .. autofunction:: plot_drift_over_time
    .. autofunction:: plot_peak_activity_map
    .. autofunction:: plot_unit_waveforms
    .. autofunction:: plot_unit_templates
    .. autofunction:: plot_unit_waveforms_density_map
    .. autofunction:: plot_amplitudes
    .. autofunction:: plot_all_amplitudes_distributions
    .. autofunction:: plot_principal_component
    .. autofunction:: plot_unit_locations
    .. autofunction:: plot_unit_probe_map
    .. autofunction:: plot_unit_depths
    .. autofunction:: plot_confusion_matrix
    .. autofunction:: plot_agreement_matrix
    .. autofunction:: plot_multicomp_graph
    .. autofunction:: plot_multicomp_agreement
    .. autofunction:: plot_multicomp_agreement_by_sorter
    .. autofunction:: plot_comparison_collision_pair_by_pair
    .. autofunction:: plot_comparison_collision_by_similarity
    .. autofunction:: plot_sorting_performance
    .. autofunction:: plot_unit_summary
    .. autofunction:: plot_sorting_summary


spikeinterface.exporters
------------------------
.. automodule:: spikeinterface.exporters

    .. autofunction:: export_to_phy
    .. autofunction:: export_report


spikeinterface.sortingcomponents
--------------------------------

Peak Localization
~~~~~~~~~~~~~~~~~
.. automodule:: spikeinterface.sortingcomponents.peak_localization

    .. autofunction:: localize_peaks

Peak Detection
~~~~~~~~~~~~~~
.. automodule:: spikeinterface.sortingcomponents.peak_detection

    .. autofunction:: detect_peaks

Motion Correction
~~~~~~~~~~~~~~~~~
.. automodule:: spikeinterface.sortingcomponents.motion_correction

    .. autoclass:: CorrectMotionRecording

Clustering
~~~~~~~~~~
.. automodule:: spikeinterface.sortingcomponents.clustering

    .. autofunction:: find_cluster_from_peaks

Template Matching
~~~~~~~~~~~~~~~~~
.. automodule:: spikeinterface.sortingcomponents.matching

    .. autofunction:: find_spikes_from_templates

