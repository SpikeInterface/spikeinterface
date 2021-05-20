import numpy as np
import struct
import os
import tempfile
import traceback
from pathlib import Path


class MdaHeader:
    def __init__(self, dt0, dims0):
        uses64bitdims = (max(dims0) > 2e9)
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
        "f2": 'float16',
        "f4": 'float32',
        "f8": 'float64',
        "i1": 'int8',
        "i2": 'int16',
        "i4": 'int32',
        "u2": 'uint16',
        "u4": 'uint32'
    }
    return map[str]


class DiskReadMda:
    def __init__(self, path, header=None):
        self._npy_mode = False
        self._path = path
        if file_extension(path) == '.npy':
            raise Exception('DiskReadMda implementation has not been tested for npy files')
            self._npy_mode = True
            if header:
                raise Exception('header not allowed in npy mode for DiskReadMda')
        if header:
            self._header = header
            self._header.header_size = 0
        else:
            self._header = _read_header(self._path)

    def dims(self):
        if self._npy_mode:
            A = np.load(self._path, mmap_mode='r')
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
            A = np.load(self._path, mmap_mode='r')
            return npy_dtype_to_string(A.dtype)
        return self._header.dt

    def numBytesPerEntry(self):
        if self._npy_mode:
            A = np.load(self._path, mmap_mode='r')
            return A.itemsize
        return self._header.num_bytes_per_entry

    def readChunk(self, i1=-1, i2=-1, i3=-1, N1=1, N2=1, N3=1):
        # print("Reading chunk {} {} {} {} {} {}".format(i1,i2,i3,N1,N2,N3))
        if i2 < 0:
            if self._npy_mode:
                A = np.load(self._path, mmap_mode='r')
                return A[:, :, i1:i1 + N1]
            return self._read_chunk_1d(i1, N1)
        elif i3 < 0:
            if N1 != self.N1():
                print("Unable to support N1 {} != {}".format(N1, self.N1()))
                return None
            X = self._read_chunk_1d(i1 + N1 * i2, N1 * N2)
            if X is None:
                print('Problem reading chunk from file: ' + self._path)
                return None
            if self._npy_mode:
                A = np.load(self._path, mmap_mode='r')
                return A[:, i2:i2 + N2]
            return np.reshape(X, (N1, N2), order='F')
        else:
            if N1 != self.N1():
                print("Unable to support N1 {} != {}".format(N1, self.N1()))
                return None
            if N2 != self.N2():
                print("Unable to support N2 {} != {}".format(N2, self.N2()))
                return None
            if self._npy_mode:
                A = np.load(self._path, mmap_mode='r')
                return A[:, :, i3:i3 + N3]
            X = self._read_chunk_1d(i1 + N1 * i2 + N1 * N2 * i3, N1 * N2 * N3)
            return np.reshape(X, (N1, N2, N3), order='F')

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
    return path.startswith('http://') or path.startswith('https://')


def _download_bytes_to_tmpfile(url, start, end):
    try:
        import requests
    except:
        raise Exception('Unable to import module: requests')
    headers = {"Range": "bytes={}-{}".format(start, end - 1)}
    r = requests.get(url, headers=headers, stream=True)
    fd, tmp_fname = tempfile.mkstemp()
    with open(tmp_fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return tmp_fname


def _read_header(path):
    if is_url(path):
        tmp_fname = _download_bytes_to_tmpfile(path, 0, 200)
        if not tmp_fname:
            raise Exception('Problem downloading bytes from ' + path)
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
        dt = 'uint8'
    elif dt_code == -3:
        dt = 'float32'
    elif dt_code == -4:
        dt = 'int16'
    elif dt_code == -5:
        dt = 'int32'
    elif dt_code == -6:
        dt = 'uint16'
    elif dt_code == -7:
        dt = 'float64'
    elif dt_code == -8:
        dt = 'uint32'
    else:
        dt = None
    return dt


def _dt_code_from_dt(dt):
    if dt == 'uint8':
        return -2
    if dt == 'float32':
        return -3
    if dt == 'int16':
        return -4
    if dt == 'int32':
        return -5
    if dt == 'uint16':
        return -6
    if dt == 'float64':
        return -7
    if dt == 'uint32':
        return -8
    return None


def get_num_bytes_per_entry_from_dt(dt):
    if dt == 'uint8':
        return 1
    if dt == 'float32':
        return 4
    if dt == 'int16':
        return 2
    if dt == 'int32':
        return 4
    if dt == 'uint16':
        return 2
    if dt == 'float64':
        return 8
    if dt == 'uint32':
        return 4
    return None


def readmda_header(path):
    if file_extension(path) == '.npy':
        raise Exception('Cannot read mda header for .npy file.')
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
    if file_extension(path) == '.npy':
        return readnpy(path);
    H = _read_header(path)
    if H is None:
        print("Problem reading header of: {}".format(path))
        return None
    f = open(path, "rb")
    try:
        f.seek(H.header_size)
        # This is how I do the column-major order
        ret = np.fromfile(f, dtype=H.dt, count=H.dimprod)
        ret = np.reshape(ret, H.dims, order='F')
        f.close()
        return ret
    except Exception as e:  # catch *all* exceptions
        print(e)
        f.close()
        return None


def writemda32(X, fname):
    if file_extension(fname) == '.npy':
        return writenpy32(X, fname)
    return _writemda(X, fname, 'float32')


def writemda64(X, fname):
    if file_extension(fname) == '.npy':
        return writenpy64(X, fname)
    return _writemda(X, fname, 'float64')


def writemda8(X, fname):
    if file_extension(fname) == '.npy':
        return writenpy8(X, fname)
    return _writemda(X, fname, 'uint8')


def writemda32i(X, fname):
    if file_extension(fname) == '.npy':
        return writenpy32i(X, fname)
    return _writemda(X, fname, 'int32')


def writemda32ui(X, fname):
    if file_extension(fname) == '.npy':
        return writenpy32ui(X, fname)
    return _writemda(X, fname, 'uint32')


def writemda16i(X, fname):
    if file_extension(fname) == '.npy':
        return writenpy16i(X, fname)
    return _writemda(X, fname, 'int16')


def writemda16ui(X, fname):
    if file_extension(fname) == '.npy':
        return writenpy16ui(X, fname)
    return _writemda(X, fname, 'uint16')


def writemda(X, fname, *, dtype):
    return _writemda(X, fname, dtype)


def _writemda(X, fname, dt):
    num_bytes_per_entry = get_num_bytes_per_entry_from_dt(dt)
    dt_code = _dt_code_from_dt(dt)
    if dt_code is None:
        print("Unexpected data type: {}".format(dt))
        return False

    if type(fname) == str:
        f = open(fname, 'wb')
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

        bytes0 = X.astype(dt).tobytes(order='F')
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
    return _writenpy(X, path, dtype='int8')


def writenpy32(X, path):
    return _writenpy(X, path, dtype='float32')


def writenpy64(X, path):
    return _writenpy(X, path, dtype='float64')


def writenpy16i(X, path):
    return _writenpy(X, path, dtype='int16')


def writenpy16ui(X, path):
    return _writenpy(X, path, dtype='uint16')


def writenpy32i(X, path):
    return _writenpy(X, path, dtype='int32')


def writenpy32ui(X, path):
    return _writenpy(X, path, dtype='uint32')


def writenpy(X, path, *, dtype):
    return _writenpy(X, path, dtype=dtype)


def _writenpy(X, path, *, dtype):
    np.save(path, X.astype(dtype=dtype, copy=False))  # astype will always create copy if dtype does not match
    # apparently allowing pickling is a security issue. (according to the docs) ??
    # np.save(path,X.astype(dtype=dtype,copy=False),allow_pickle=False) # astype will always create copy if dtype does not match
    return True


def appendmda(X, path):
    if file_extension(path) == '.npy':
        raise Exception('appendmda not yet implemented for .npy files')
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
        A = np.reshape(X, X.size, order='F').astype(H.dt)
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
    return struct.unpack('<i', f.read(4))[0]


def _read_int64(f):
    return struct.unpack('<q', f.read(8))[0]


def _write_int32(f, val):
    f.write(struct.pack('<i', val))


def _write_int64(f, val):
    f.write(struct.pack('<q', val))


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
