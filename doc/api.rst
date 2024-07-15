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
    .. autoclass:: BaseSnippets
        :members:
    .. autoclass:: BaseEvent
        :members:
    .. autoclass:: SortingAnalyzer
        :members:
    .. autofunction:: create_sorting_analyzer
    .. autofunction:: load_sorting_analyzer
    .. autofunction:: compute_sparsity
    .. autofunction:: estimate_sparsity
    .. autoclass:: ChannelSparsity
        :members:
    .. autoclass:: BinaryRecordingExtractor
    .. autoclass:: ZarrRecordingExtractor
    .. autoclass:: BinaryFolderRecording
    .. autoclass:: NpzFolderSorting
    .. autoclass:: NpyFolderSnippets
    .. autoclass:: NumpyRecording
    .. autoclass:: NumpySorting
    .. autoclass:: NumpySnippets
    .. autoclass:: AppendSegmentRecording
    .. autoclass:: ConcatenateSegmentRecording
    .. autoclass:: SelectSegmentRecording
    .. autoclass:: AppendSegmentSorting
    .. autoclass:: SplitSegmentSorting
    .. autoclass:: SelectSegmentSorting
    .. autofunction:: download_dataset
    .. autofunction:: write_binary_recording
    .. autofunction:: set_global_tmp_folder
    .. autofunction:: set_global_dataset_folder
    .. autofunction:: set_global_job_kwargs
    .. autofunction:: get_random_data_chunks
    .. autofunction:: get_channel_distances
    .. autofunction:: get_closest_channels
    .. autofunction:: get_noise_levels
    .. autofunction:: get_chunk_with_margin
    .. autofunction:: order_channels_by_depth
    .. autofunction:: get_template_amplitudes
    .. autofunction:: get_template_extremum_channel
    .. autofunction:: get_template_extremum_channel_peak_shift
    .. autofunction:: get_template_extremum_amplitude
    .. autofunction:: append_recordings
    .. autofunction:: concatenate_recordings
    .. autofunction:: split_recording
    .. autofunction:: select_segment_recording
    .. autofunction:: append_sortings
    .. autofunction:: split_sorting
    .. autofunction:: select_segment_sorting
    .. autofunction:: read_binary
    .. autofunction:: read_zarr
    .. autofunction:: apply_merges_to_sorting
    .. autofunction:: spike_vector_to_spike_trains
    .. autofunction:: random_spikes_selection


Low-level
~~~~~~~~~

.. automodule:: spikeinterface.core
    :noindex:

    .. autoclass:: ChunkRecordingExecutor

spikeinterface.extractors
-------------------------


NEO-based
~~~~~~~~~

.. automodule:: spikeinterface.extractors

    .. autofunction:: read_alphaomega
    .. autofunction:: read_alphaomega_event
    .. autofunction:: read_axona
    .. autofunction:: read_biocam
    .. autofunction:: read_binary
    .. autofunction:: read_blackrock
    .. autofunction:: read_ced
    .. autofunction:: read_intan
    .. autofunction:: read_maxwell
    .. autofunction:: read_mearec
    .. autofunction:: read_mcsraw
    .. autofunction:: read_neuralynx
    .. autofunction:: read_neuralynx_sorting
    .. autofunction:: read_neuroexplorer
    .. autofunction:: read_neuroscope
    .. autofunction:: read_nix
    .. autofunction:: read_openephys
    .. autofunction:: read_openephys_event
    .. autofunction:: read_plexon
    .. autofunction:: read_plexon_sorting
    .. autofunction:: read_plexon2
    .. autofunction:: read_plexon2_sorting
    .. autofunction:: read_spike2
    .. autofunction:: read_spikegadgets
    .. autofunction:: read_spikeglx
    .. autofunction:: read_tdt
    .. autofunction:: read_zarr


Non-NEO-based
~~~~~~~~~~~~~
.. automodule:: spikeinterface.extractors
    :noindex:

    .. autofunction:: read_alf_sorting
    .. autofunction:: read_bids
    .. autofunction:: read_cbin_ibl
    .. autofunction:: read_combinato
    .. autofunction:: read_ibl_recording
    .. autofunction:: read_hdsort
    .. autofunction:: read_herdingspikes
    .. autofunction:: read_kilosort
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
    .. autofunction:: read_waveclus
    .. autofunction:: read_yass




.. _api_preprocessing:


spikeinterface.preprocessing
----------------------------


.. automodule:: spikeinterface.preprocessing

    .. autofunction:: astype
    .. autofunction:: average_across_direction
    .. autofunction:: bandpass_filter
    .. autofunction:: blank_staturation
    .. autofunction:: center
    .. autofunction:: clip
    .. autofunction:: common_reference
    .. autofunction:: correct_lsb
    .. autofunction:: correct_motion
    .. autofunction:: depth_order
    .. autofunction:: detect_bad_channels
    .. autofunction:: directional_derivative
    .. autofunction:: filter
    .. autofunction:: gaussian_filter
    .. autofunction:: highpass_filter
    .. autofunction:: highpass_spatial_filter
    .. autofunction:: interpolate_bad_channels
    .. autofunction:: normalize_by_quantile
    .. autofunction:: notch_filter
    .. autofunction:: phase_shift
    .. autofunction:: rectify
    .. autofunction:: remove_artifacts
    .. autofunction:: resample
    .. autofunction:: scale
    .. autofunction:: silence_periods
    .. autofunction:: unsigned_to_signed
    .. autofunction:: whiten
    .. autofunction:: zero_channel_pad
    .. autofunction:: zscore


spikeinterface.postprocessing
-----------------------------

.. automodule:: spikeinterface.postprocessing

    .. autofunction:: compute_noise_levels
    .. autofunction:: compute_template_metrics
    .. autofunction:: compute_principal_components
    .. autofunction:: compute_spike_amplitudes
    .. autofunction:: compute_unit_locations
    .. autofunction:: compute_spike_locations
    .. autofunction:: compute_template_similarity
    .. autofunction:: compute_correlograms
    .. autofunction:: compute_isi_histograms
    .. autofunction:: get_template_metric_names
    .. autofunction:: align_sorting


spikeinterface.qualitymetrics
-----------------------------

.. automodule:: spikeinterface.qualitymetrics

    .. autofunction:: compute_quality_metrics
    .. autofunction:: get_quality_metric_list
    .. autofunction:: get_quality_pca_metric_list
    .. autofunction:: get_default_qm_params


spikeinterface.sorters
----------------------
.. automodule:: spikeinterface.sorters

    .. autofunction:: available_sorters
    .. autofunction:: installed_sorters
    .. autofunction:: get_default_sorter_params
    .. autofunction:: get_sorter_params_description
    .. autofunction:: print_sorter_versions
    .. autofunction:: get_sorter_description
    .. autofunction:: run_sorter
    .. autofunction:: run_sorter_jobs
    .. autofunction:: run_sorter_by_property
    .. autofunction:: read_sorter_folder

Low level
~~~~~~~~~
.. automodule:: spikeinterface.sorters
    :noindex:

    .. autoclass:: BaseSorter


spikeinterface.comparison
-------------------------
.. automodule:: spikeinterface.comparison

    .. autofunction:: compare_two_sorters
    .. autofunction:: compare_multiple_sorters
    .. autofunction:: compare_sorter_to_ground_truth
    .. autofunction:: compare_templates
    .. autofunction:: compare_multiple_templates
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

    .. autoclass:: CollisionGTComparison
    .. autoclass:: CorrelogramGTComparison
    .. autoclass:: CollisionGTStudy
    .. autoclass:: CorrelogramGTStudy



spikeinterface.widgets
----------------------

.. automodule:: spikeinterface.widgets

    .. autofunction:: set_default_plotter_backend
    .. autofunction:: get_default_plotter_backend

    .. autofunction:: plot_agreement_matrix
    .. autofunction:: plot_all_amplitudes_distributions
    .. autofunction:: plot_amplitudes
    .. autofunction:: plot_autocorrelograms
    .. autofunction:: plot_confusion_matrix
    .. autofunction:: plot_comparison_collision_by_similarity
    .. autofunction:: plot_crosscorrelograms
    .. autofunction:: plot_isi_distribution
    .. autofunction:: plot_motion
    .. autofunction:: plot_multicomparison_agreement
    .. autofunction:: plot_multicomparison_agreement_by_sorter
    .. autofunction:: plot_multicomparison_graph
    .. autofunction:: plot_peak_activity
    .. autofunction:: plot_probe_map
    .. autofunction:: plot_quality_metrics
    .. autofunction:: plot_rasters
    .. autofunction:: plot_sorting_summary
    .. autofunction:: plot_spike_locations
    .. autofunction:: plot_spikes_on_traces
    .. autofunction:: plot_template_metrics
    .. autofunction:: plot_template_similarity
    .. autofunction:: plot_traces
    .. autofunction:: plot_unit_depths
    .. autofunction:: plot_unit_locations
    .. autofunction:: plot_unit_presence
    .. autofunction:: plot_unit_probe_map
    .. autofunction:: plot_unit_summary
    .. autofunction:: plot_unit_templates
    .. autofunction:: plot_unit_waveforms_density_map
    .. autofunction:: plot_unit_waveforms
    .. autofunction:: plot_study_run_times
    .. autofunction:: plot_study_unit_counts
    .. autofunction:: plot_study_performances
    .. autofunction:: plot_study_agreement_matrix
    .. autofunction:: plot_study_summary
    .. autofunction:: plot_study_comparison_collision_by_similarity


spikeinterface.exporters
------------------------
.. automodule:: spikeinterface.exporters

    .. autofunction:: export_to_phy
    .. autofunction:: export_report


spikeinterface.curation
------------------------
.. automodule:: spikeinterface.curation

    .. autoclass:: CurationSorting
    .. autoclass:: MergeUnitsSorting
    .. autoclass:: SplitUnitSorting
    .. autofunction:: get_potential_auto_merge
    .. autofunction:: find_redundant_units
    .. autofunction:: remove_redundant_units
    .. autofunction:: remove_duplicated_spikes
    .. autofunction:: remove_excess_spikes
    .. autofunction:: apply_sortingview_curation


spikeinterface.generation
-------------------------

.. currentmodule:: spikeinterface.generation

Core
~~~~


.. autofunction:: generate_recording
.. autofunction:: generate_sorting
.. autofunction:: generate_snippets
.. autofunction:: generate_templates
.. autofunction:: generate_recording_by_size
.. autofunction:: generate_ground_truth_recording
.. autofunction:: add_synchrony_to_sorting
.. autofunction:: synthesize_random_firings
.. autofunction:: inject_some_duplicate_units
.. autofunction:: inject_some_split_units
.. autofunction:: synthetize_spike_train_bad_isi
.. autofunction:: inject_templates
.. autofunction:: noise_generator_recording
.. autoclass:: InjectTemplatesRecording
.. autoclass:: NoiseGeneratorRecording

Drift
~~~~~

.. autofunction:: generate_drifting_recording
.. autofunction:: generate_displacement_vector
.. autofunction:: make_one_displacement_vector
.. autofunction:: make_linear_displacement
.. autofunction:: move_dense_templates
.. autofunction:: interpolate_templates
.. autoclass:: DriftingTemplates
.. autoclass:: InjectDriftingTemplatesRecording

Hybrid
~~~~~~

.. autofunction:: generate_hybrid_recording
.. autofunction:: estimate_templates_from_recording
.. autofunction:: select_templates
.. autofunction:: scale_template_to_range
.. autofunction:: relocate_templates
.. autofunction:: fetch_template_object_from_database
.. autofunction:: fetch_templates_database_info
.. autofunction:: list_available_datasets_in_template_database
.. autofunction:: query_templates_from_database


Noise
~~~~~

.. autofunction:: generate_noise


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

Clustering
~~~~~~~~~~
.. automodule:: spikeinterface.sortingcomponents.clustering

    .. autofunction:: find_cluster_from_peaks

Template Matching
~~~~~~~~~~~~~~~~~
.. automodule:: spikeinterface.sortingcomponents.matching

    .. autofunction:: find_spikes_from_templates

Motion Correction
~~~~~~~~~~~~~~~~~
.. automodule:: spikeinterface.sortingcomponents.motion

    .. autoclass:: Motion
    .. autofunction:: estimate_motion
    .. autofunction:: interpolate_motion
    .. autofunction:: correct_motion_on_peaks
    .. autofunction:: interpolate_motion_on_traces
    .. autofunction:: clean_motion_vector
    .. autoclass:: InterpolateMotionRecording
