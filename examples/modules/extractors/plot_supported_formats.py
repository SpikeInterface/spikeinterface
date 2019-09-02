'''
Supported File Formats
========================

Currently, we support many popular file formats for both raw and sorted extracellular datasets. Given the standardized,
modular design of our recording and sorting extractors, adding new file formats is straightforward so we expect this
list to grow in future versions.
'''

##############################################################################
# Recording Data Formats
# ----------------------
#
# For raw data formats, we currently support:
#
# * **Binary** - BinDatRecordingExtractor
# * **Biocam HDF5** - BiocamRecordingExtractor
# * **Experimental Directory Structure (Exdir)** - ExdirRecordingExtractor
# * **Intan** - IntanRecordingExtractor
# * **Klusta** - KlustaRecordingExtractor
# * **MaxOne** - MaxOneRecordingExtractor
# * **MCSH5** - MCSH5RecordingExtractor
# * **MEArec** - MEArecRecordingExtractor
# * **Mountainsort MDA** - MdaRecordingExtractor
# * **Neurodata Without Borders** - NwbRecordingExtractor
# * **Open Ephys** - OpenEphysRecordingExtractor
# * **Phy/Kilosort** - PhyRecordingExtractor/KilosortRecordingExtractor
# * **SpikeGLX** - SpikeGLXRecordingExtractor
# * **Spyking Circus** - SpykingCircusRecordingExtractor

##############################################################################
# Sorted Data Formats
# ----------------------
#
# For sorted data formats, we currently support:
#
# * **Experimental Directory Structure (Exdir)** - ExdirSortingExtractor
# * **HerdingSpikes2** - HS2SortingExtractor
# * **Kilosort/Kilosort2** - KiloSortSortingExtractor
# * **Klusta** - KlustaSortingExtractor
# * **MEArec** - MEArecSortingExtractor
# * **Mountainsort MDA** - MdaSortingExtractor
# * **Neurodata Without Borders** - NwbSortingExtractor
# * **NPZ (numpy)** - NpzSortingExtractor
# * **Open Ephys** - OpenEphysSortingExtractor
# * **Spyking Circus** - SpykingCircusSortingExtractor
# * **Trideclous** - TridesclousSortingExtractor

##############################################################################
# Installed Extractors
# --------------------
#
# To check which extractors are useable in a given python environment, one can print the installed recording extractor
# list and the installed sorting extractor list. An example from a newly installed miniconda3 environment is shown
# below.
#
# First, import the spikeextractors package:

import spikeinterface.extractors as se

##############################################################################
# Then you can check the installed RecordingExtractor list:

print(se.installed_recording_extractor_list)

##############################################################################
# and the installed SortingExtractors list:

print(se.installed_sorting_extractor_list)

##############################################################################
# When trying to use an extractor that has not been installed in your environment, an installation message will appear
# indicating which python packages must be installed as a prerequisite to using the extractor:

exdir_file = 'path_to_exdir_file'
try:
    recording = se.ExdirRecordingExtractor(exdir_file)
except Exception as e:
    print(e)

##############################################################################
# So to use either of the Exdir extractors, you must install the python package :code:`exdir`.
# The python packages that are required to use of all the extractors can be installed as below:
#
# :code:`pip install exdir h5py pyintan MEArec pyopenephys pyintan tridesclous pynwb`








