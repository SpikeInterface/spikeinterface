from __future__ import annotations

from pathlib import Path

import numpy as np

from spikeinterface.core import BaseRecording, BaseRecordingSegment
from spikeinterface.core.core_tools import define_function_from_class


class MCSH5RecordingExtractor(BaseRecording):
    """Load a MCS H5 file as a recording extractor.

    Parameters
    ----------
    file_path : str or Path
        The path to the MCS h5 file.
    stream_id : int, default: 0
        The stream ID to load.

    Returns
    -------
    recording : MCSH5RecordingExtractor
        The loaded data.
    """

    installation_mesg = "To use the MCSH5RecordingExtractor install h5py: \n\n pip install h5py\n\n"

    def __init__(self, file_path, stream_id=0):

        try:
            import h5py
        except ImportError:
            raise ImportError(self.installation_mesg)

        self._file_path = file_path

        mcs_info = openMCSH5File(self._file_path, stream_id)
        self._rf = mcs_info["filehandle"]

        BaseRecording.__init__(
            self,
            sampling_frequency=mcs_info["sampling_frequency"],
            channel_ids=mcs_info["channel_ids"],
            dtype=mcs_info["dtype"],
        )

        self.extra_requirements.append("h5py")

        recording_segment = MCSH5RecordingSegment(
            self._rf, stream_id, mcs_info["num_frames"], sampling_frequency=mcs_info["sampling_frequency"]
        )
        self.add_recording_segment(recording_segment)

        # set gain
        self.set_channel_gains(mcs_info["gain"])

        # set offsets
        self.set_channel_offsets(mcs_info["offset"])

        # set other properties
        self.set_property("electrode_labels", mcs_info["electrode_labels"])

        self._kwargs = {"file_path": str(Path(file_path).absolute()), "stream_id": stream_id}

    def __del__(self):
        self._rf.close()


class MCSH5RecordingSegment(BaseRecordingSegment):
    def __init__(self, rf, stream_id, num_frames, sampling_frequency):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self._rf = rf
        self._stream_id = stream_id
        self._num_samples = int(num_frames)
        self._stream = self._rf.require_group("/Data/Recording_0/AnalogStream/Stream_" + str(self._stream_id))

    def get_num_samples(self):
        return self._num_samples

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        if isinstance(channel_indices, slice):
            traces = self._stream.get("ChannelData")[channel_indices, start_frame:end_frame].T
        else:
            # channel_indices is np.ndarray
            if np.array(channel_indices).size > 1 and np.any(np.diff(channel_indices) < 0):
                # get around h5py constraint that it does not allow datasets
                # to be indexed out of order
                sorted_channel_indices = np.sort(channel_indices)
                resorted_indices = np.array([list(sorted_channel_indices).index(ch) for ch in channel_indices])
                recordings = self._stream.get("ChannelData")[sorted_channel_indices, start_frame:end_frame].T
                traces = recordings[:, resorted_indices]
            else:
                traces = self._stream.get("ChannelData")[channel_indices, start_frame:end_frame].T

        return traces


def openMCSH5File(filename, stream_id):
    """Open an MCS hdf5 file, read and return the recording info.
    Specs can be found online
    https://www.multichannelsystems.com/downloads/documentation?page=3
    """

    import h5py

    rf = h5py.File(filename, "r")

    stream_name = "Stream_" + str(stream_id)
    analog_stream_names = list(rf.require_group("/Data/Recording_0/AnalogStream").keys())
    assert stream_name in analog_stream_names, (
        f"Specified stream does not exist. " f"Available streams: {analog_stream_names}"
    )

    stream = rf.require_group("/Data/Recording_0/AnalogStream/" + stream_name)
    data = stream.get("ChannelData")
    timestamps = np.array(stream.get("ChannelDataTimeStamps"))
    info = np.array(stream.get("InfoChannel"))
    dtype = data.dtype

    Unit = info["Unit"][0]
    Tick = info["Tick"][0] / 1e6
    exponent = info["Exponent"][0]
    convFact = info["ConversionFactor"][0]
    gain_uV = 1e6 * (convFact.astype(float) * (10.0**exponent))
    offset_uV = -1e6 * (info["ADZero"].astype(float) * (10.0**exponent)) * gain_uV

    nRecCh, nFrames = data.shape
    channel_ids = [f"Ch{ch}" for ch in info["ChannelID"]]
    assert len(np.unique(channel_ids)) == len(channel_ids), "Duplicate MCS channel IDs found"
    electrodeLabels = [l.decode() for l in info["Label"]]

    assert timestamps[0][0] < timestamps[0][2], "Please check the validity of 'ChannelDataTimeStamps' in the stream."
    TimeVals = np.arange(timestamps[0][0], timestamps[0][2] + 1, 1) * Tick

    if Unit != b"V":
        print(f"Unexpected units found, expected volts, found {Unit.decode('UTF-8')}. Assuming Volts.")

    timestep_avg = np.mean(TimeVals[1:] - TimeVals[0:-1])
    timestep_min = np.min(TimeVals[1:] - TimeVals[0:-1])
    timestep_max = np.min(TimeVals[1:] - TimeVals[0:-1])
    assert all(
        np.abs(np.array((timestep_min, timestep_max)) - timestep_avg) / timestep_avg < 1e-6
    ), "Time steps vary by more than 1 ppm"
    samplingRate = 1.0 / timestep_avg

    mcs_info = {
        "filehandle": rf,
        "num_frames": nFrames,
        "sampling_frequency": samplingRate,
        "num_channels": nRecCh,
        "channel_ids": channel_ids,
        "electrode_labels": electrodeLabels,
        "gain": gain_uV,
        "dtype": dtype,
        "offset": offset_uV,
    }

    return mcs_info


read_mcsh5 = define_function_from_class(source_class=MCSH5RecordingExtractor, name="read_mcsh5")
