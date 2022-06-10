API
===

spikeinterface.core
-------------------
.. automodule:: spikeinterface.core

    .. autofunction:: load_extractor
    .. autoclass:: BaseRecording
    .. autoclass:: BaseSorting
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
    .. autofunction:: extract_waveforms
    .. autoclass:: WaveformExtractor
    .. autofunction:: download_dataset
    .. autofunction:: write_binary_recording
    .. autofunction:: set_global_tmp_folder
    .. autofunction:: set_global_dataset_folder
    .. autoclass:: ChunkRecordingExecutor



spikeinterface.extractors
-------------------------
.. automodule:: spikeinterface.extractors

    .. autofunction:: toy_example
    .. autofunction:: read_bids_folder
    .. autofunction:: read_mearec
    .. autofunction:: read_spikeglx
    .. autofunction:: read_openephys
    .. autofunction:: read_openephys_event
    .. autofunction:: read_intan
    .. autofunction:: read_neuroscope
    .. autofunction:: read_plexon
    .. autofunction:: read_neuralynx
    .. autofunction:: read_blackrock
    .. autofunction:: read_mcsraw
    .. autofunction:: read_kilosort
    .. autofunction:: read_spike2
    .. autofunction:: read_ced
    .. autofunction:: read_maxwell
    .. autofunction:: read_nix
    .. autofunction:: read_spikegadgets
    .. autofunction:: read_klusta
    .. autofunction:: read_hdsort
    .. autofunction:: read_waveclust
    .. autofunction:: read_yass
    .. autofunction:: read_combinato
    .. autofunction:: read_tridesclous
    .. autofunction:: read_spykingcircus
    .. autofunction:: read_herdingspikes
    .. autofunction:: read_mda_recording
    .. autofunction:: read_mda_sorting
    .. autofunction:: read_shybrid_recording
    .. autofunction:: read_shybrid_sorting
    .. autofunction:: read_alf_sorting
    .. autofunction:: read_alphaomega
    .. autofunction:: read_alphaomega_event


spikeinterface.toolkit
----------------------

toolkit.utils
~~~~~~~~~~~~~


.. automodule:: spikeinterface.toolkit

    .. autofunction:: get_random_data_chunks
    .. autofunction:: get_channel_distances
    .. autofunction:: get_closest_channels
    .. autofunction:: get_noise_levels


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
    .. autofunction:: get_template_channel_sparsity
    .. autofunction:: compute_unit_centers_of_mass
    .. autofunction:: calculate_template_metrics
    .. autofunction:: get_template_metric_names
    .. autofunction:: compute_principal_components
    .. autofunction:: get_spike_amplitudes
    .. autofunction:: compute_correlograms


Quality metrics
~~~~~~~~~~~~~~~
.. automodule:: spikeinterface.toolkit.qualitymetrics

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

    .. autoclass:: GroundTruthComparison
        :members:

    .. autoclass:: SymmetricSortingComparison
        :members:
        :undoc-members:

    .. autoclass:: GroundTruthStudy
        :members:
        :undoc-members:


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
    .. autofunction:: plot_unit_waveform_density_map
    .. autofunction:: plot_amplitudes_timeseries
    .. autofunction:: plot_amplitudes_distribution
    .. autofunction:: plot_principal_component
    .. autofunction:: plot_unit_localization
    .. autofunction:: plot_unit_probe_map
    .. autofunction:: plot_units_depth_vs_amplitude
    .. autofunction:: plot_confusion_matrix
    .. autofunction:: plot_agreement_matrix
    .. autofunction:: plot_multicomp_graph
    .. autofunction:: plot_multicomp_agreement
    .. autofunction:: plot_multicomp_agreement_by_sorter
    .. autofunction:: plot_comparison_collision_pair_by_pair
    .. autofunction:: plot_comparison_collision_by_similarity
    .. autofunction:: plot_sorting_performance
    .. autofunction:: plot_unit_summary


spikeinterface.exporters
------------------------
.. automodule:: spikeinterface.exporters

    .. autofunction:: export_to_phy
    .. autofunction:: export_report


spikeinterface.sortingcomponents
-----------------

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
.. automodule::

    .. autofunction:: find_cluster_from_peaks

