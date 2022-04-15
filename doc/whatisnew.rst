=============
Release notes
=============


.. toctree::
  :maxdepth: 1

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

NEW API
-------

Version 0.94.0
==============

* Refactor waveformextractor with waveform_tools.
* Implement Zarr backend for save()
* Read IBL compress files. 
* Phase shift (destripe) preprocessor
* Test are run partially : faster GH actions
* Many improvement in sorting compnents: template matching, select_peak, motion_estimation, motion_correction

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
