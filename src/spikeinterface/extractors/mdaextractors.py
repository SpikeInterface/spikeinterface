from __future__ import annotations

import os
import json
import struct
import tempfile
import traceback
from pathlib import Path
from typing import Union, List

import numpy as np

from spikeinterface.core import BaseRecording, BaseRecordingSegment, BaseSorting, BaseSortingSegment
from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.core import write_binary_recording
from spikeinterface.core.job_tools import fix_job_kwargs


class MdaRecordingExtractor(BaseRecording):
    """Load MDA format data as a recording extractor.

    Parameters
    ----------
    folder_path : str or Path
        Path to the MDA folder.
    raw_fname : str, default: "raw.mda"
        File name of raw file
    params_fname : str, default: "params.json"
        File name of params file
    geom_fname : str, default: "geom.csv"
        File name of geom file

    Returns
    -------
    extractor : MdaRecordingExtractor
        The loaded data.
    """

    def __init__(self, folder_path, raw_fname="raw.mda", params_fname="params.json", geom_fname="geom.csv"):
        folder_path = Path(folder_path)
        self._folder_path = folder_path
        self._dataset_params = read_dataset_params(self._folder_path, params_fname)
        self._timeseries_path = self._folder_path / raw_fname
        geom = np.loadtxt(self._folder_path / geom_fname, delimiter=",", ndmin=2)
        self._diskreadmda = DiskReadMda(str(self._timeseries_path))
        dtype = self._diskreadmda.dt()
        num_channels = self._diskreadmda.N1()
        assert geom.shape[0] == self._diskreadmda.N1(), (
            f"Incompatible dimensions between geom.csv and timeseries "
            f"file: {geom.shape[0]} <> {self._diskreadmda.N1()}"
        )
        sampling_frequency = float(self._dataset_params["samplerate"])
        BaseRecording.__init__(
            self, sampling_frequency=sampling_frequency, channel_ids=np.arange(num_channels), dtype=dtype
        )
        rec_segment = MdaRecordingSegment(self._diskreadmda, sampling_frequency)
        self.add_recording_segment(rec_segment)
        self.set_dummy_probe_from_locations(geom)
        self._kwargs = {
            "folder_path": str(Path(folder_path).absolute()),
            "raw_fname": raw_fname,
            "params_fname": params_fname,
            "geom_fname": geom_fname,
        }

    @staticmethod
    def write_recording(
        recording,
        save_path,
        params=dict(),
        raw_fname="raw.mda",
        params_fname="params.json",
        geom_fname="geom.csv",
        dtype=None,
        verbose=False,
        **job_kwargs,
    ):
        """Write a recording to file in MDA format.

        Parameters
        ----------
        recording : RecordingExtractor
            The recording extractor to be saved.
        save_path : str or Path
            The folder to save the Mda files.
        params : dictionary
            Dictionary with optional parameters to save metadata.
            Sampling frequency is appended to this dictionary.
        raw_fname : str, default: "raw.mda"
            File name of raw file
        params_fname : str, default: "params.json"
            File name of params file
        geom_fname : str, default: "geom.csv"
            File name of geom file
        dtype : dtype or None, default: None
            Data type to be used. If None dtype is same as recording traces.
        verbose : bool
            If True, shows progress bar when saving recording.
        **job_kwargs:
            Use by job_tools modules to set:

                * chunk_size or chunk_memory, or total_memory
                * n_jobs
                * progress_bar
        """
        job_kwargs = fix_job_kwargs(job_kwargs)
        assert recording.get_num_segments() == 1, (
            "MdaRecording.write_recording() can only write a single segment " "recording"
        )
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        save_file_path = save_path / raw_fname
        parent_dir = save_path
        num_channels = recording.get_num_channels()
        num_frames = recording.get_num_frames(0)

        geom = recording.get_channel_locations()

        if dtype is None:
            dtype = recording.get_dtype()

        if dtype == "float":
            dtype = "float32"
        if dtype == "int":
            dtype = "int16"

        header = MdaHeader(dt0=dtype, dims0=(num_channels, num_frames))
        header_size = header.header_size

        write_binary_recording(
            recording,
            file_paths=save_file_path,
            dtype=dtype,
            byte_offset=header_size,
            add_file_extension=False,
            verbose=verbose,
            **job_kwargs,
        )

        with save_file_path.open("rb+") as f:
            header.write(f)

        params["samplerate"] = float(recording.get_sampling_frequency())
        with (parent_dir / params_fname).open("w") as f:
            json.dump(params, f)
        np.savetxt(str(parent_dir / geom_fname), geom, delimiter=",")


class MdaRecordingSegment(BaseRecordingSegment):
    def __init__(self, diskreadmda, sampling_frequency):
        self._diskreadmda = diskreadmda
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self._num_samples = self._diskreadmda.N2()

    def get_num_samples(self):
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex : Number of samples in the signal block
        """
        return self._num_samples

    def get_traces(
        self,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        channel_indices: Union[List, None] = None,
    ) -> np.ndarray:
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()
        recordings = self._diskreadmda.readChunk(
            i1=0, i2=start_frame, N1=self._diskreadmda.N1(), N2=end_frame - start_frame
        )
        recordings = recordings[channel_indices, :].T
        return recordings


class MdaSortingExtractor(BaseSorting):
    """Load MDA format data as a sorting extractor.

    NOTE: As in the MDA format, the max_channel property indexes the channels that are given as input
    to the sorter.
    If sorting was run on a subset of channels of the recording, then the max_channel values are
    based on that subset, so care must be taken when associating these values with a recording.
    If additional sorting segments are added to this sorting extractor after initialization,
    then max_channel will not be updated. The max_channel indices begin at 1.

    Parameters
    ----------
    file_path : str or Path
        Path to the MDA file.
    sampling_frequency : int
        The sampling frequency.

    Returns
    -------
    extractor : MdaRecordingExtractor
        The loaded data.
    """

    def __init__(self, file_path, sampling_frequency):
        firings = readmda(str(Path(file_path).absolute()))
        labels = firings[2, :]
        unit_ids = np.unique(labels).astype(int)
        BaseSorting.__init__(self, unit_ids=unit_ids, sampling_frequency=sampling_frequency)

        sorting_segment = MdaSortingSegment(firings)
        self.add_sorting_segment(sorting_segment)

        # Store the max channel for each unit
        # Every spike assigned to a unit (label) has the same max channel
        # ref: https://github.com/SpikeInterface/spikeinterface/issues/3695#issuecomment-2663329006
        max_channels = []
        segment = self._sorting_segments[0]
        for unit_id in self.unit_ids:
            label_mask = segment._labels == unit_id
            # since all max channels are the same, we can just grab the first occurrence for the unit
            max_channel = segment._max_channels[label_mask][0]
            max_channels.append(max_channel)

        self.set_property(key="max_channel", values=max_channels)

        self._kwargs = {
            "file_path": str(Path(file_path).absolute()),
            "sampling_frequency": sampling_frequency,
        }

    @staticmethod
    def write_sorting(sorting, save_path, write_primary_channels=False):
        assert sorting.get_num_segments() == 1, "MdaSorting.write_sorting() can only write a single segment sorting"
        unit_ids = sorting.get_unit_ids()
        times_list = []
        labels_list = []
        primary_channels_list = []
        for unit_index, unit_id in enumerate(unit_ids):
            times = sorting.get_unit_spike_train(unit_id=unit_id)
            times_list.append(times)
            # unit id may not be numeric
            if unit_id.dtype.kind in "iu":
                labels_list.append(np.ones(times.shape, dtype=unit_id.dtype) * unit_id)
            else:
                labels_list.append(np.ones(times.shape, dtype=int) * unit_index)
            if write_primary_channels:
                if "max_channel" in sorting.get_property_keys():
                    primary_channels_list.append([sorting.get_unit_property(unit_id, "max_channel")] * times.shape[0])
                else:
                    raise ValueError(
                        "Unable to write primary channels because 'max_channel' spike feature not set in unit "
                        + str(unit_id)
                    )
            else:
                primary_channels_list.append(np.zeros(times.shape))
        all_times = _concatenate(times_list)
        all_labels = _concatenate(labels_list)
        all_primary_channels = _concatenate(primary_channels_list)
        sort_inds = np.argsort(all_times)
        all_times = all_times[sort_inds]
        all_labels = all_labels[sort_inds]
        all_primary_channels = all_primary_channels[sort_inds]
        L = len(all_times)
        firings = np.zeros((3, L))
        firings[0, :] = all_primary_channels
        firings[1, :] = all_times
        firings[2, :] = all_labels

        writemda64(firings, str(save_path))


class MdaSortingSegment(BaseSortingSegment):
    def __init__(self, firings):
        self._firings = firings
        self._max_channels = self._firings[0, :]
        self._spike_times = self._firings[1, :]
        self._labels = self._firings[2, :]
        BaseSortingSegment.__init__(self)

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
    ) -> np.ndarray:
        # must be implemented in subclass
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.inf
        inds = np.where(
            (self._labels == unit_id) & (start_frame <= self._spike_times) & (self._spike_times < end_frame)
        )
        return np.rint(self._spike_times[inds]).astype(int)


read_mda_recording = define_function_from_class(source_class=MdaRecordingExtractor, name="read_mda_recording")
read_mda_sorting = define_function_from_class(source_class=MdaSortingExtractor, name="read_mda_sorting")


def _concatenate(list):
    if len(list) == 0:
        return np.array([])
    return np.concatenate(list)


def read_dataset_params(dsdir, params_fname):
    fname1 = dsdir / params_fname
    if not fname1.is_file():
        raise Exception("Dataset parameter file does not exist: " + fname1)
    with open(fname1) as f:
        return json.load(f)


######### MDAIO ###########
class MdaHeader:
    def __init__(self, dt0, dims0):
        uses64bitdims = max(dims0) > 2e9
        self.uses64bitdims = uses64bitdims
        self.dt_code = _dt_code_from_dt(dt0)
        self.dt = dt0
        self.num_bytes_per_entry = get_num_bytes_per_entry_from_dt(dt0)
        self.num_dims = len(dims0)
        self.dimprod = np.prod(dims0)
        self.dims = dims0
        if uses64bitdims:
            self.header_size = 3 * 4 + self.num_dims * 8
        else:
            self.header_size = (3 + self.num_dims) * 4

    def write(self, f):
        H = self
        _write_int32(f, H.dt_code)
        _write_int32(f, H.num_bytes_per_entry)
        if H.uses64bitdims:
            _write_int32(f, -H.num_dims)
            for j in range(0, H.num_dims):
                _write_int64(f, H.dims[j])
        else:
            _write_int32(f, H.num_dims)
            for j in range(0, H.num_dims):
                _write_int32(f, H.dims[j])


def npy_dtype_to_string(dt):
    str = dt.str[1:]
    map = {
        "f2": "float16",
        "f4": "float32",
        "f8": "float64",
        "i1": "int8",
        "i2": "int16",
        "i4": "int32",
        "u2": "uint16",
        "u4": "uint32",
    }
    return map[str]


class DiskReadMda:
    def __init__(self, path, header=None):
        self._npy_mode = False
        self._path = path
        if file_extension(path) == ".npy":
            raise Exception("DiskReadMda implementation has not been tested for npy files")
            self._npy_mode = True
            if header:
                raise Exception("header not allowed in npy mode for DiskReadMda")
        if header:
            self._header = header
            self._header.header_size = 0
        else:
            self._header = _read_header(self._path)

    def dims(self):
        if self._npy_mode:
            A = np.load(self._path, mmap_mode="r")
            return A.shape
        return self._header.dims

    def N1(self):
        return self.dims()[0]

    def N2(self):
        return self.dims()[1]

    def N3(self):
        return self.dims()[2]

    def dt(self):
        if self._npy_mode:
            A = np.load(self._path, mmap_mode="r")
            return npy_dtype_to_string(A.dtype)
        return self._header.dt

    def numBytesPerEntry(self):
        if self._npy_mode:
            A = np.load(self._path, mmap_mode="r")
            return A.itemsize
        return self._header.num_bytes_per_entry

    def readChunk(self, i1=-1, i2=-1, i3=-1, N1=1, N2=1, N3=1):
        # print("Reading chunk {} {} {} {} {} {}".format(i1,i2,i3,N1,N2,N3))
        if i2 < 0:
            if self._npy_mode:
                A = np.load(self._path, mmap_mode="r")
                return A[:, :, i1 : i1 + N1]
            return self._read_chunk_1d(i1, N1)
        elif i3 < 0:
            if N1 != self.N1():
                print("Unable to support N1 {} != {}".format(N1, self.N1()))
                return None
            X = self._read_chunk_1d(i1 + N1 * i2, N1 * N2)

            if X is None:
                print("Problem reading chunk from file: " + self._path)
                return None
            if self._npy_mode:
                A = np.load(self._path, mmap_mode="r")
                return A[:, i2 : i2 + N2]
            return np.reshape(X, (N1, N2), order="F")
        else:
            if N1 != self.N1():
                print("Unable to support N1 {} != {}".format(N1, self.N1()))
                return None
            if N2 != self.N2():
                print("Unable to support N2 {} != {}".format(N2, self.N2()))
                return None
            if self._npy_mode:
                A = np.load(self._path, mmap_mode="r")
                return A[:, :, i3 : i3 + N3]
            X = self._read_chunk_1d(i1 + N1 * i2 + N1 * N2 * i3, N1 * N2 * N3)
            return np.reshape(X, (N1, N2, N3), order="F")

    def _read_chunk_1d(self, i, N):
        offset = self._header.header_size + self._header.num_bytes_per_entry * i
        if is_url(self._path):
            tmp_fname = _download_bytes_to_tmpfile(self._path, offset, offset + self._header.num_bytes_per_entry * N)
            try:
                ret = self._read_chunk_1d_helper(tmp_fname, N, offset=0)
            except:
                ret = None
            return ret
        return self._read_chunk_1d_helper(self._path, N, offset=offset)

    def _read_chunk_1d_helper(self, path0, N, *, offset):
        f = open(path0, "rb")
        try:
            f.seek(offset)
            ret = np.fromfile(f, dtype=self._header.dt, count=N)
            f.close()
            return ret
        except Exception as e:  # catch *all* exceptions
            print(e)
            f.close()
            return None


def is_url(path):
    return path.startswith("http://") or path.startswith("https://")


def _download_bytes_to_tmpfile(url, start, end):
    import requests

    headers = {"Range": f"bytes={start}-{end - 1}"}

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()  # Exposes HTTPError if one occurred

        with tempfile.NamedTemporaryFile(delete=False, mode="wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

            # Store the temp file name for return
            tmp_fname = f.name

    return tmp_fname


def _read_header(path):
    if is_url(path):
        tmp_fname = _download_bytes_to_tmpfile(path, 0, 200)
        if not tmp_fname:
            raise Exception("Problem downloading bytes from " + path)
        try:
            ret = _read_header(tmp_fname)
        except:
            ret = None
        Path(tmp_fname).unlink()
        return ret

    f = open(path, "rb")
    try:
        dt_code = _read_int32(f)
        num_bytes_per_entry = _read_int32(f)
        num_dims = _read_int32(f)
        uses64bitdims = False
        if num_dims < 0:
            uses64bitdims = True
            num_dims = -num_dims
        if num_dims < 1 or num_dims > 6:  # allow single dimension as of 12/6/17
            print("Invalid number of dimensions: {}".format(num_dims))
            f.close()
            return None
        dims = []
        dimprod = 1
        if uses64bitdims:
            for j in range(0, num_dims):
                tmp0 = _read_int64(f)
                dimprod = dimprod * tmp0
                dims.append(tmp0)
        else:
            for j in range(0, num_dims):
                tmp0 = _read_int32(f)
                dimprod = dimprod * tmp0
                dims.append(tmp0)
        dt = _dt_from_dt_code(dt_code)
        if dt is None:
            print("Invalid data type code: {}".format(dt_code))
            f.close()
            return None
        H = MdaHeader(dt, dims)
        if uses64bitdims:
            H.uses64bitdims = True
            H.header_size = 3 * 4 + H.num_dims * 8
        f.close()
        return H
    except Exception as e:  # catch *all* exceptions
        print(e)
        f.close()
        return None


def _dt_from_dt_code(dt_code):
    if dt_code == -2:
        dt = "uint8"
    elif dt_code == -3:
        dt = "float32"
    elif dt_code == -4:
        dt = "int16"
    elif dt_code == -5:
        dt = "int32"
    elif dt_code == -6:
        dt = "uint16"
    elif dt_code == -7:
        dt = "float64"
    elif dt_code == -8:
        dt = "uint32"
    else:
        dt = None
    return dt


def _dt_code_from_dt(dt):
    if dt == "uint8":
        return -2
    if dt == "float32":
        return -3
    if dt == "int16":
        return -4
    if dt == "int32":
        return -5
    if dt == "uint16":
        return -6
    if dt == "float64":
        return -7
    if dt == "uint32":
        return -8
    return None


def get_num_bytes_per_entry_from_dt(dt):
    if dt == "uint8":
        return 1
    if dt == "float32":
        return 4
    if dt == "int16":
        return 2
    if dt == "int32":
        return 4
    if dt == "uint16":
        return 2
    if dt == "float64":
        return 8
    if dt == "uint32":
        return 4
    return None


def readmda_header(path):
    if file_extension(path) == ".npy":
        raise Exception("Cannot read mda header for .npy file.")
    return _read_header(path)


def _write_header(path, H, rewrite=False):
    if rewrite:
        f = open(path, "r+b")
    else:
        f = open(path, "wb")
    try:
        _write_int32(f, H.dt_code)
        _write_int32(f, H.num_bytes_per_entry)
        if H.uses64bitdims:
            _write_int32(f, -H.num_dims)
            for j in range(0, H.num_dims):
                _write_int64(f, H.dims[j])
        else:
            _write_int32(f, H.num_dims)
            for j in range(0, H.num_dims):
                _write_int32(f, H.dims[j])
        f.close()
        return True
    except Exception as e:  # catch *all* exceptions
        print(e)
        f.close()
        return False


def readmda(path):
    if file_extension(path) == ".npy":
        return readnpy(path)
    H = _read_header(path)
    if H is None:
        print("Problem reading header of: {}".format(path))
        return None
    f = open(path, "rb")
    try:
        f.seek(H.header_size)
        # This is how I do the column-major order
        ret = np.fromfile(f, dtype=H.dt, count=H.dimprod)
        ret = np.reshape(ret, H.dims, order="F")
        f.close()
        return ret
    except Exception as e:  # catch *all* exceptions
        print(e)
        f.close()
        return None


def writemda32(X, fname):
    if file_extension(fname) == ".npy":
        return writenpy32(X, fname)
    return _writemda(X, fname, "float32")


def writemda64(X, fname):
    if file_extension(fname) == ".npy":
        return writenpy64(X, fname)
    return _writemda(X, fname, "float64")


def writemda8(X, fname):
    if file_extension(fname) == ".npy":
        return writenpy8(X, fname)
    return _writemda(X, fname, "uint8")


def writemda32i(X, fname):
    if file_extension(fname) == ".npy":
        return writenpy32i(X, fname)
    return _writemda(X, fname, "int32")


def writemda32ui(X, fname):
    if file_extension(fname) == ".npy":
        return writenpy32ui(X, fname)
    return _writemda(X, fname, "uint32")


def writemda16i(X, fname):
    if file_extension(fname) == ".npy":
        return writenpy16i(X, fname)
    return _writemda(X, fname, "int16")


def writemda16ui(X, fname):
    if file_extension(fname) == ".npy":
        return writenpy16ui(X, fname)
    return _writemda(X, fname, "uint16")


def writemda(X, fname, *, dtype):
    return _writemda(X, fname, dtype)


def _writemda(X, fname, dt):
    num_bytes_per_entry = get_num_bytes_per_entry_from_dt(dt)
    dt_code = _dt_code_from_dt(dt)
    if dt_code is None:
        print("Unexpected data type: {}".format(dt))
        return False

    if type(fname) == str:
        f = open(fname, "wb")
    else:
        f = fname
    try:
        _write_int32(f, dt_code)
        _write_int32(f, num_bytes_per_entry)
        _write_int32(f, X.ndim)
        for j in range(0, X.ndim):
            _write_int32(f, X.shape[j])
        # This is how I do column-major order
        # A=np.reshape(X,X.size,order='F').astype(dt)
        # A.tofile(f)

        bytes0 = X.astype(dt).tobytes(order="F")
        f.write(bytes0)

        if type(fname) == str:
            f.close()
        return True
    except Exception as e:  # catch *all* exceptions
        traceback.print_exc()
        print(e)
        if type(fname) == str:
            f.close()
        return False


def readnpy(path):
    return np.load(path)


def writenpy8(X, path):
    return _writenpy(X, path, dtype="int8")


def writenpy32(X, path):
    return _writenpy(X, path, dtype="float32")


def writenpy64(X, path):
    return _writenpy(X, path, dtype="float64")


def writenpy16i(X, path):
    return _writenpy(X, path, dtype="int16")


def writenpy16ui(X, path):
    return _writenpy(X, path, dtype="uint16")


def writenpy32i(X, path):
    return _writenpy(X, path, dtype="int32")


def writenpy32ui(X, path):
    return _writenpy(X, path, dtype="uint32")


def writenpy(X, path, *, dtype):
    return _writenpy(X, path, dtype=dtype)


def _writenpy(X, path, *, dtype):
    np.save(path, X.astype(dtype=dtype, copy=False))  # astype will always create copy if dtype does not match
    # apparently allowing pickling is a security issue. (according to the docs) ??
    # np.save(path,X.astype(dtype=dtype,copy=False),allow_pickle=False) # astype will always create copy if dtype does not match
    return True


def appendmda(X, path):
    if file_extension(path) == ".npy":
        raise Exception("appendmda not yet implemented for .npy files")
    H = _read_header(path)
    if H is None:
        print("Problem reading header of: {}".format(path))
        return None
    if len(H.dims) != len(X.shape):
        print("Incompatible number of dimensions in appendmda", H.dims, X.shape)
        return None
    num_entries_old = np.product(H.dims)
    num_dims = len(H.dims)
    for j in range(num_dims - 1):
        if X.shape[j] != X.shape[j]:
            print("Incompatible dimensions in appendmda", H.dims, X.shape)
            return None
    H.dims[num_dims - 1] = H.dims[num_dims - 1] + X.shape[num_dims - 1]
    try:
        _write_header(path, H, rewrite=True)
        f = open(path, "r+b")
        f.seek(H.header_size + H.num_bytes_per_entry * num_entries_old)
        A = np.reshape(X, X.size, order="F").astype(H.dt)
        A.tofile(f)
        f.close()
    except Exception as e:  # catch *all* exceptions
        print(e)
        f.close()
        return False


def file_extension(fname):
    if type(fname) == str:
        filename, ext = os.path.splitext(fname)
        return ext
    else:
        return None


def _read_int32(f):
    return struct.unpack("<i", f.read(4))[0]


def _read_int64(f):
    return struct.unpack("<q", f.read(8))[0]


def _write_int32(f, val):
    f.write(struct.pack("<i", val))


def _write_int64(f, val):
    f.write(struct.pack("<q", val))


def _header_from_file(f):
    try:
        dt_code = _read_int32(f)
        num_bytes_per_entry = _read_int32(f)
        num_dims = _read_int32(f)
        uses64bitdims = False
        if num_dims < 0:
            uses64bitdims = True
            num_dims = -num_dims
        if num_dims < 1 or num_dims > 6:  # allow single dimension as of 12/6/17
            print("Invalid number of dimensions: {}".format(num_dims))
            return None
        dims = []
        dimprod = 1
        if uses64bitdims:
            for j in range(0, num_dims):
                tmp0 = _read_int64(f)
                dimprod = dimprod * tmp0
                dims.append(tmp0)
        else:
            for j in range(0, num_dims):
                tmp0 = _read_int32(f)
                dimprod = dimprod * tmp0
                dims.append(tmp0)
        dt = _dt_from_dt_code(dt_code)
        if dt is None:
            print("Invalid data type code: {}".format(dt_code))
            return None
        H = MdaHeader(dt, dims)
        if uses64bitdims:
            H.uses64bitdims = True
            H.header_size = 3 * 4 + H.num_dims * 8
        return H
    except Exception as e:  # catch *all* exceptions
        print(e)
        return None
