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
    .. autoclass:: WaveformExtractor
        :members:
    .. autofunction:: extract_waveforms
    .. autofunction:: load_waveforms
    .. autofunction:: compute_sparsity
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

..
    .. autofunction:: read_binary
    .. autofunction:: read_zarr
    .. autofunction:: append_recordings
    .. autofunction:: concatenate_recordings
    .. autofunction:: split_recording
    .. autofunction:: select_segment_recording
    .. autofunction:: append_sortings
    .. autofunction:: split_sorting
    .. autofunction:: select_segment_sorting

Low-level
~~~~~~~~~

.. automodule:: spikeinterface.core
    :noindex:

    .. autoclass:: BaseWaveformExtractorExtension
    .. autoclass:: ChannelSparsity
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
    .. autofunction:: read_blackrock
    .. autofunction:: read_ced
    .. autofunction:: read_intan
    .. autofunction:: read_maxwell
    .. autofunction:: read_mearec
    .. autofunction:: read_mcsraw
    .. autofunction:: read_neuralynx
    .. autofunction:: read_neuralynx_sorting
    .. autofunction:: read_neuroscope
    .. autofunction:: read_nix
    .. autofunction:: read_openephys
    .. autofunction:: read_openephys_event
    .. autofunction:: read_plexon
    .. autofunction:: read_plexon_sorting
    .. autofunction:: read_spike2
    .. autofunction:: read_spikegadgets
    .. autofunction:: read_spikeglx
    .. autofunction:: read_tdt

Non-NEO-based
~~~~~~~~~~~~~
.. automodule:: spikeinterface.extractors
    :noindex:

    .. autofunction:: read_alf_sorting
    .. autofunction:: read_bids
    .. autofunction:: read_cbin_ibl
    .. autofunction:: read_combinato
    .. autofunction:: read_ibl_streaming_recording
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
    .. autofunction:: gaussian_bandpass_filter
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
    .. autofunction:: run_sorters
    .. autofunction:: run_sorter_by_property

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

    .. autoclass:: CollisionGTComparison
    .. autoclass:: CorrelogramGTComparison
    .. autoclass:: CollisionGTStudy
    .. autoclass:: CorrelogramGTStudy



spikeinterface.widgets
----------------------

.. automodule:: spikeinterface.widgets

    .. autofunction:: set_default_plotter_backend
    .. autofunction:: get_default_plotter_backend

    .. autofunction:: plot_all_amplitudes_distributions
    .. autofunction:: plot_amplitudes
    .. autofunction:: plot_autocorrelograms
    .. autofunction:: plot_crosscorrelograms
    .. autofunction:: plot_quality_metrics
    .. autofunction:: plot_sorting_summary
    .. autofunction:: plot_spike_locations
    .. autofunction:: plot_spikes_on_traces
    .. autofunction:: plot_template_metrics
    .. autofunction:: plot_template_similarity
    .. autofunction:: plot_timeseries
    .. autofunction:: plot_unit_depths
    .. autofunction:: plot_unit_locations
    .. autofunction:: plot_unit_summary
    .. autofunction:: plot_unit_templates
    .. autofunction:: plot_unit_waveforms_density_map
    .. autofunction:: plot_unit_waveforms


Legacy widgets
~~~~~~~~~~~~~~

These widgets are only available with the "matplotlib" backend

.. automodule:: spikeinterface.widgets
    :noindex:

    .. autofunction:: plot_rasters
    .. autofunction:: plot_probe_map
    .. autofunction:: plot_isi_distribution
    .. autofunction:: plot_peak_activity_map
    .. autofunction:: plot_principal_component
    .. autofunction:: plot_unit_probe_map
    .. autofunction:: plot_confusion_matrix
    .. autofunction:: plot_agreement_matrix
    .. autofunction:: plot_multicomp_graph
    .. autofunction:: plot_multicomp_agreement
    .. autofunction:: plot_multicomp_agreement_by_sorter
    .. autofunction:: plot_comparison_collision_pair_by_pair
    .. autofunction:: plot_comparison_collision_by_similarity
    .. autofunction:: plot_sorting_performance


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
.. automodule:: spikeinterface.sortingcomponents.motion_interpolation

    .. autoclass:: InterpolateMotionRecording

Clustering
~~~~~~~~~~
.. automodule:: spikeinterface.sortingcomponents.clustering

    .. autofunction:: find_cluster_from_peaks

Template Matching
~~~~~~~~~~~~~~~~~
.. automodule:: spikeinterface.sortingcomponents.matching

    .. autofunction:: find_spikes_from_templates
