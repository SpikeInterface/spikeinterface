.. _releasenotes:

=============
Release notes
=============


.. toctree::
  :maxdepth: 1

  releases/0.101.0.rst
  releases/0.100.8.rst
  releases/0.100.7.rst
  releases/0.100.6.rst
  releases/0.100.5.rst
  releases/0.100.4.rst
  releases/0.100.3.rst
  releases/0.100.2.rst
  releases/0.100.1.rst
  releases/0.100.0.rst
  releases/0.99.1.rst
  releases/0.99.0.rst
  releases/0.98.2.rst
  releases/0.98.1.rst
  releases/0.98.0.rst
  releases/0.97.1.rst
  releases/0.97.0.rst
  releases/0.96.1.rst
  releases/0.96.0.rst
  releases/0.95.1.rst
  releases/0.95.0.rst
  releases/0.94.0.rst
  releases/0.93.0.rst
  releases/0.92.0.rst
  releases/0.91.0.rst
  releases/0.90.1.rst
  releases/0.90.0.rst
  releases/0.13.0.rst
  releases/0.12.0.rst
  releases/0.11.0.rst
  releases/0.10.0.rst
  releases/0.9.9.rst
  releases/0.9.1.rst


Version 0.101.0
===============

* Implementation of `SortingAnalyzer` (#2398)
* Improved auto-merging functions and enable `SortingAnalyzer` to merge units and extensions (#3043, #3154, #3203)
* Added framework for hybrid recording generation (#2436, #2769, #2857)
* Refactored motion correction with the `Motion` class and the DREDGE AP and LFP methods (#2915, #3062)
* Extendeded benchmarking of `sortingcomponents` (#2501, #2518, #2586, #2811, #2959)
* Added a powerful drift generator module (#2683)

Version 0.101.0rc1
==================

* Release candidate for version 0.101.0 (15th July 2024)

Version 0.101.0rc0
==================

* Release candidate for version 0.101.0 (8th June 2024)

Version 0.100.8
===============

* Minor release with bug fixes

Version 0.100.7
===============

* Minor release with bug fixes


Version 0.100.6
===============

* Minor release with bug fixes


Version 0.100.5
===============

* Minor release with bug fixes


Version 0.100.4
===============

* Minor release with extended compression capability for Zarr


Version 0.100.3
===============

* Minor release with bug fixes for Zarr compressor and NWB in container


Version 0.100.2
===============

* Minor release with fix for running Kilosort4 with GPU support in container


Version 0.100.1
===============

* Minor release with some bug fixes and Kilosort4 support


Version 0.100.0
===============

* Several improvements and bug fixes for Windows users
* Important refactoring of NWB extractors:
  * implemented direct backend implementation (to avoid using `pynwb`)
  * sped up streaming using `remfile`
  * added support for `zarr` backend
* Removed `joblib` dependency in favor of `ParallelProcessExecutor`
* Improved flexibility when running sorters in containers by adding several options for installing `spikeinterface`
* Add `Templates` class to core, which handles unit templates and sparsity (#1982)
* Added Zarr-backend to `Sorting` objects (`sorting.save(folder="...", format="zarr")`) (#2403)
* Added `SharedmemRecording` for shared memory recordings (#2365)
* Added machinery for moving/interpolating templates for generating hybrid recordings with drift (#2291)
* Added new fast method for unit/spike/peak localization: `grid_convolution` (#2172)


Version 0.99.1
==============

* Minor release with some bug fixes


Version 0.99.0
==============

* Refactored :code:`core.generate` module to create high-quality lazy ground-truth data
* Added DeepInterpolation training, transfer, and inference to preprocessing
* Added new quality_metrics (synchrony, firing_range, amplitude_cv)
* Added new template metrics (num positive/negative peaks + multi-channel metrics: velocities, spread, exp. decay)
* Important improvements to internal sorters (especially Spyking Circus 2)
* Ported all widgets to "new" API and dismantled legacy widgets
* Extensive clean-up of docstrings, code style, and documentation


Version 0.98.2
==============

* Minor release with some bug fixes


Version 0.98.1
==============

* Minor release with some bug fixes


Version 0.98.0
==============

* Implemented motion correction framework in SpikeInterface
* Added "grid convolution" as localization method
* Added amplitude_scaling implementation in postprocessing
* Improved container support for docker and singularity
* Improved implementation and performance of quality metrics module
* Extensive additions to :code:`sortingcomponents`` module (pipeline nodes, waveform denoising, sparsity)
* :code:`_ind` to :code:`_index` in complex dtype representation for peak and spikes

Version 0.97.1
==============

* Minor release with some bug fixes and minor new features


Version 0.97.0
==============

* Implemented "sparse" waveforms in core and propagated to all modules:

  * Add waveform extractor :code:`save()` function and Zarr backend
  * Dense waveforms can be saved as sparse with the :code:`save()` function
  * Sparse waveforms can be extracted directly with :code:`sparse=True`
* Added IBL preprocessing tools for Neuropixels:

  * Bad channel detection
  * Channel interpolation
  * High-pass spatial filter
* | Add subfolder for sorter output ("output_folder/sorter_output") and :code:`sorting_info` field to Sorting,
  | so that "sorter_output" can be deleted without deleting metadata and :code:`sorting_info`
* Refactor Quality Metrics module:

  * Improve parameter handling
  * Unified naming and conventions
  * Add missing metrics: drift, amp medians, sliding_rp_violations
* Extended Motion estimation and motion correction, which is now ready to use
* Removed deprecated functions/modules:

  * :code:`toolkit` module
  * :code:`symmetrize` argument from compute_correlograms
  * :code:`localize_units` from postprocessing

* Major rewrite of the docs, with extensive module-specific documentation


Version 0.96.1
==============

* Bump up probeinterface version to 0.2.14 to fix bug in reading SpikeGLX contact locations

Version 0.96.0
==============

* Change setup.py to pyproject.toml
* Enable loading waveform extractor without recording and with pre-existing sorting
* Implement NpzFolder for saving Sorting object
* Implement in-memory waveform extensions
* Improve NEO event handling
* Modify pyks for IBL version
* Implement sparse PCA
* Extend comparison to multi-segment sortings
* Various improvements to widgets modules

Version 0.95.1
==============

* Widgets: Fix creation of 2D axes when figure is passed
* Widgets: Add check for matplotlib backend when using ipywidgets

Version 0.95.0
==============

* Add `BaseSnippets` object to handle waveform cutouts
* Sacrificed `toolkit` in favor of: `preprocessing`, `postprocessing`, `qualitymetrics`, and `curation` modules
* Major refactoring widget module to allow for multiple backends
  * `matplotlib`
  * `ipywidgets`
  * `sortingview`
* First implementations of (experimental) `sortingcomponents`-based sorters:
  * `spykingcircus2`
  * `tridesclous2`

Version 0.94.0
==============

* Refactor WaveformExtractor with waveform_tools
* Implement Zarr backend for save()
* Read IBL compressed files
* Phase shift (destripe) preprocessor
* Test are run partially : faster GH actions
* Many improvement in sorting compnents: template matching, select_peaks, motion_estimation, motion_correction

Version 0.93.0
==============

* add WaveformExtractorExtension (PC, qualitymetrics, spike amplitudes)
  to automatically store and retrieve processed data waveforms folder
* add singularity integration in run_sorter
* add a link to the originating recording to the sorting object
* new framework for collision benchmark
* refactor comparison module and add TemplateComparison
* add template_matching module (experimental)
* add motion_correction module (experimental)


Version 0.92.0
==============

* many improvements in toolkit module
* added spike unit localization
* handle time vector in base recording

Version 0.91.0
==============

* Major improvements and bug-fixes.
* Improvements for spikeinterface-gui.

Version 0.90.1
==============

* Minor release  - bug fixes

Version 0.90.0
==============

* Major release:

  * many API modifications : no backward compatibility
  * contains all subpackages
  * get_traces() has transposed shape (time x channels)
  * handles multi segment
  * new WaveformExtractor object to handle waveforms computation
  * new Event object to handle epochs and events


LEGACY API
----------

Version 0.13.0
==============

  * Final release of version 0.1X - bug fixes

Version 0.12.0
==============

  * Major update: API change for get_traces to enable return_scaled

Version 0.11.0
==============

  * Bug fixes and improvements on efficiency

Version 0.10.0
==============

  * Minor updates and bug fixes for biorXiv preprint

Version 0.9.9
==============

  * Major updates and bug fixes to all packages - pre-release

Version 0.9.1
==============

  * First SpikeInterface pre-release
