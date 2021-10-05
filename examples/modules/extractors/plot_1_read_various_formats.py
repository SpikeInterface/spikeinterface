'''
Read various format into SI
============================

:code:`spikeinterface` can read various format of "recording" (traces) and "sorting" (spike train) data.

Internally, to read different formats, :code:`spikeinterface` either uses:
  * a wrapper to the `neo <https://github.com/NeuralEnsemble/python-neo>`_ rawio classes
  * or a direct implementation

Note that:

  * file formats contain a "recording", a "sorting",  or "both"
  * file formats can be file-based (NWB, ...)  or folder based (SpikeGLX, OpenEphys, ...)

In this example we demonstrate how to read different file formats into SI
'''

import matplotlib.pyplot as plt

import spikeinterface as si
import spikeinterface.extractors as se

##############################################################################
# Let's download some datasets in different formats from the
# `ephy_testing_data <https://gin.g-node.org/NeuralEnsemble/ephy_testing_data>`_ repo:
#   * MEArec: an simulator format which is hdf5-based. It contains both a "recording" and a "sorting" in the same file.
#   * Spike2: file from spike2 devices. It contains "recording" information only.


spike2_file_path = si.download_dataset(remote_path='spike2/130322-1LY.smr')
print(spike2_file_path)

mearec_folder_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
print(mearec_folder_path)

##############################################################################
# Now that we have downloaded the files let's load them into SI.
#
# The :code:`read_spike2` function returns one object, a :code:`RecordingExtractor`.
#
# Note that internally this file contains 2 data streams ('0' and '1'), so we need to specify which one we
# want to retrieve ('0' in our case):

recording = se.read_spike2(spike2_file_path, stream_id='0')
print(recording)
print(type(recording))
print(isinstance(recording, si.BaseRecording))

##############################################################################
# The :code:`read_spike2` function is equivalent to instantiating a
# :code:`Spike2RecordingExtractor` object:
#

recording = se.Spike2RecordingExtractor(spike2_file_path, stream_id='0')
print(recording)

##############################################################################
# The :code:`read_mearec` function returns two objects,
# a :code:`RecordingExtractor` and a :code:`SortingExtractor`:

recording, sorting = se.read_mearec(mearec_folder_path)
print(recording)
print(type(recording))
print()
print(sorting)
print(type(sorting))

##############################################################################
#  The :code:`read_mearec` function is equivalent to:

recording = se.MEArecRecordingExtractor(mearec_folder_path)
sorting = se.MEArecSortingExtractor(mearec_folder_path)

##############################################################################
# SI objects (:code:`RecordingExtractor` and :code:`SortingExtractor`) object
# can be plotted quickly with the :code:`widgets` submodule:

import spikeinterface.widgets as sw

w_ts = sw.plot_timeseries(recording, time_range=(0, 5))
w_rs = sw.plot_rasters(sorting, time_range=(0, 5))

plt.show()
