'''
Read various format
============================

spikeinterface can read various format of "recording" (traces) and "sorting" (spiketrain)

Internally, to read diffrents format, spikeinterface either use:
  * `neo <https://github.com/NeuralEnsemble/python-neo>`_ 
  * or a direct implementation

Note that:

  * some format contain "recording" or "sorting " or "both"
  * some format are file based (mearec, nwb ...) some other are folder based (spikeglx, openephys, ...)

'''


import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.extractors as se

import spikeinterface as si
import spikeinterface.extractors as se


##############################################################################
# Lets download some dataset in diferents formats:
#   * mearec : a simulator format is hdf5 based. contain recording and sorting. file based
#   * spike2: file from spike2 device. contain recording only. file based


spike2_file_path = si.download_dataset(remote_path='spike2/130322-1LY.smr')
print(spike2_file_path)


mearec_folder_path = si.download_dataset(remote_path='mearec/mearec_test_10s.h5')
print(mearec_folder_path)


##############################################################################
# Read spike2 give one object
#
#  Note that internally this file contain 2 stream ('0' and '1') so we need to specify which stream.

recording = se.read_spike2(spike2_file_path, stream_id='0')
print(recording)
print(type(recording))
print(isinstance(recording, si.BaseRecording))


##############################################################################
# This equivalent to do this
#
# The "old" (<=0.12) spikeinterface API use to have this "class approach reading"

recording = se.Spike2RecordingExtractor(spike2_file_path, stream_id='0')
print(recording)


##############################################################################
# Read MEArec give 2 object

recording, sorting = se.read_mearec(mearec_folder_path)
print(recording)
print(type(recording))
print()
print(sorting)
print(type(sorting))


##############################################################################
# This equivalent to do this

recording = se.MEArecRecordingExtractor(mearec_folder_path)
sorting = se.MEArecSortingExtractor(mearec_folder_path)

##############################################################################
# recording and sorting object can be plot quickly with the widgets submodule

import spikeinterface.widgets as sw

w_ts = sw.plot_timeseries(recording, time_range=(0, 5))
w_rs = sw.plot_rasters(sorting, time_range=(0, 5))

plt.show()





