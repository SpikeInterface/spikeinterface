"""
Low-level functions to work in frequency domain for n-dim arrays

Copied from https://github.com/int-brain-lab/ibl-neuropixel/ on 2/1/2024

"""

import numpy as np
from pathlib import Path
import re

def _dromedary(string) -> str:
    """
    Convert a string to camel case.  Acronyms/initialisms are preserved.

    Parameters
    ----------
    string : str
        To be converted to camel case

    Returns
    -------
    str
        The string in camel case

    Examples
    --------
    >>> _dromedary('Hello world') == 'helloWorld'
    >>> _dromedary('motion_energy') == 'motionEnergy'
    >>> _dromedary('passive_RFM') == 'passive RFM'
    >>> _dromedary('FooBarBaz') == 'fooBarBaz'

    See Also
    --------
    readableALF
    """

    def _capitalize(x):
        return x if x.isupper() else x.capitalize()

    if not string:  # short circuit on None and ''
        return string
    first, *other = re.split(r"[_\s]", string)
    if len(other) == 0:
        # Already camel/Pascal case, ensure first letter lower case
        return first[0].lower() + first[1:]
    # Convert to camel case, preserving all-uppercase elements
    first = first if first.isupper() else first.casefold()
    return "".join([first, *map(_capitalize, other)])


def to_alf(object, attribute, extension, namespace=None, timescale=None, extra=None):
    """
    Given a set of ALF file parts, return a valid ALF file name.  Essential periods and
    underscores are added by the function.

    Parameters
    ----------
    object : str
        The ALF object name
    attribute : str
        The ALF object attribute name
    extension : str
        The file extension
    namespace : str
        An optional namespace
    timescale : str, tuple
        An optional timescale
    extra : str, tuple
        One or more optional extra ALF attributes

    Returns
    -------
    str
        A file name string built from the ALF parts

    Examples
    --------
    >>> to_alf('spikes', 'times', 'ssv')
    'spikes.times.ssv'
    >>> to_alf('spikes', 'times', 'ssv', namespace='ibl')
    '_ibl_spikes.times.ssv'
    >>> to_alf('spikes', 'times', 'ssv', namespace='ibl', timescale='ephysClock')
    '_ibl_spikes.times_ephysClock.ssv'
    >>> to_alf('spikes', 'times', 'ssv', namespace='ibl', timescale=('ephys clock', 'minutes'))
    '_ibl_spikes.times_ephysClock_minutes.ssv'
    >>> to_alf('spikes', 'times', 'npy', namespace='ibl', timescale='ephysClock', extra='raw')
    '_ibl_spikes.times_ephysClock.raw.npy'
    >>> to_alf('wheel', 'timestamps', 'npy', 'ibl', 'bpod', ('raw', 'v12'))
    '_ibl_wheel.timestamps_bpod.raw.v12.npy'
    """
    # Validate inputs
    if not extension:
        raise TypeError("An extension must be provided")
    elif extension.startswith("."):
        extension = extension[1:]
    if any(pt is not None and "." in pt for pt in (object, attribute, namespace, extension, timescale)):
        raise ValueError("ALF parts must not contain a period (`.`)")
    if "_" in (namespace or ""):
        raise ValueError("Namespace must not contain extra underscores")
    if object[0] == "_":
        raise ValueError("Objects must not contain underscores; use namespace arg instead")
    # Ensure parts are camel case (converts whitespace and snake case)
    if timescale:
        timescale = filter(None, [timescale] if isinstance(timescale, str) else timescale)
        timescale = "_".join(map(_dromedary, timescale))
    # Convert attribute to camel case, leaving '_times', etc. in tact
    times_re = re.search("_(times|timestamps|intervals)$", attribute)
    idx = times_re.start() if times_re else len(attribute)
    attribute = _dromedary(attribute[:idx]) + attribute[idx:]
    object = _dromedary(object)

    # Optional extras may be provided as string or tuple of strings
    if not extra:
        extra = ()
    elif isinstance(extra, str):
        extra = extra.split(".")

    # Construct ALF file
    parts = (
        ("_%s_" % namespace if namespace else "") + object,
        attribute + ("_%s" % timescale if timescale else ""),
        *extra,
        extension,
    )
    return ".".join(parts)


def save_object_npy(alfpath, dico, object, parts=None, namespace=None, timescale=None) -> list:
    """
    Saves a dictionary in `ALF format`_ using object as object name and dictionary keys as
    attribute names. Dimensions have to be consistent.

    Simplified ALF example: _namespace_object.attribute.part1.part2.extension

    Parameters
    ----------
    alfpath : str, pathlib.Path
        Path of the folder to save data to
    dico : dict
        Dictionary to save to npy; keys correspond to ALF attributes
    object : str
        Name of the object to save
    parts : str, list, None
        Extra parts to the ALF name
    namespace : str, None
        The optional namespace of the object
    timescale : str, None
        The optional timescale of the object

    Returns
    -------
    list
        List of written files

    Examples
    --------
    >>> spikes = {'times': np.arange(50), 'depths': np.random.random(50)}
    >>> files = save_object_npy('/path/to/my/alffolder/', spikes, 'spikes')

    .. _ALF format:
        https://int-brain-lab.github.io/ONE/alf_intro.html
    """
    alfpath = Path(alfpath)
    status = check_dimensions(dico)
    if status != 0:
        raise ValueError(
            "Dimensions are not consistent to save all arrays in ALF format: "
            + str([(k, v.shape) for k, v in dico.items()])
        )
    out_files = []
    for k, v in dico.items():
        out_file = alfpath / to_alf(object, k, "npy", extra=parts, namespace=namespace, timescale=timescale)
        np.save(out_file, v)
        out_files.append(out_file)
    return out_files


def check_dimensions(dico):
    """
    Test for consistency of dimensions as per ALF specs in a dictionary.

    Alf broadcasting rules: only accepts consistent dimensions for a given axis
    a dimension is consistent with another if it's empty, 1, or equal to the other arrays
    dims [a, 1],  [1, b] and [a, b] are all consistent, [c, 1] is not

    Parameters
    ----------
    dico : ALFBunch, dict
        Dictionary containing data

    Returns
    -------
    int
        Status 0 for consistent dimensions, 1 for inconsistent dimensions
    """
    # supported = (np.ndarray, pd.DataFrame)  # idt any dataframes in this specific use case for SI
    supported = (np.ndarray,)  # Data types that have a shape attribute
    shapes = [dico[lab].shape for lab in dico if isinstance(dico[lab], supported) and not lab.startswith("timestamps")]
    first_shapes = [sh[0] for sh in shapes]
    # Continuous timeseries are permitted to be a (2, 2)
    timeseries = [k for k, v in dico.items() if k.startswith("timestamps") and isinstance(v, np.ndarray)]
    if any(timeseries):
        for key in timeseries:
            if dico[key].ndim == 1 or (dico[key].ndim == 2 and dico[key].shape[1] == 1):
                # Should be vector with same length as other attributes
                first_shapes.append(dico[key].shape[0])
            elif dico[key].ndim > 1 and dico[key].shape != (2, 2):
                return 1  # ts not a (2, 2) arr or a vector

    ok = len(first_shapes) == 0 or set(first_shapes).issubset({max(first_shapes), 1})
    return int(ok is False)


def rms(x, axis=-1):
    """
    Root mean square of array along axis

    :param x: array on which to compute RMS
    :param axis: (optional, -1)
    :return: numpy array
    """
    return np.sqrt(np.mean(x**2, axis=axis))


def _fcn_extrap(x, f, bounds):
    """
    Extrapolates a flat value before and after bounds
    x: array to be filtered
    f: function to be applied between bounds (cf. fcn_cosine below)
    bounds: 2 elements list or np.array
    """
    y = f(x)
    y[x < bounds[0]] = f(bounds[0])
    y[x > bounds[1]] = f(bounds[1])
    return y


def fcn_cosine(bounds, gpu=False):
    """
    Returns a soft thresholding function with a cosine taper:
    values <= bounds[0]: values
    values < bounds[0] < bounds[1] : cosine taper
    values < bounds[1]: bounds[1]
    :param bounds:
    :param gpu: bool
    :return: lambda function
    """
    if gpu:
        import cupy as gp
    else:
        gp = np

    def _cos(x):
        return (1 - gp.cos((x - bounds[0]) / (bounds[1] - bounds[0]) * gp.pi)) / 2

    func = lambda x: _fcn_extrap(x, _cos, bounds)  # noqa
    return func


def fscale(ns, si=1, one_sided=False):
    """
    numpy.fft.fftfreq returns Nyquist as a negative frequency so we propose this instead

    :param ns: number of samples
    :param si: sampling interval in seconds
    :param one_sided: if True, returns only positive frequencies
    :return: fscale: numpy vector containing frequencies in Hertz
    """
    fsc = np.arange(0, np.floor(ns / 2) + 1) / ns / si  # sample the frequency scale
    if one_sided:
        return fsc
    else:
        return np.concatenate((fsc, -fsc[slice(-2 + (ns % 2), 0, -1)]), axis=0)


def bp(ts, si, b, axis=None):
    """
    Band-pass filter in frequency domain

    :param ts: time serie
    :param si: sampling interval in seconds
    :param b: cutout frequencies: 4 elements vector or list
    :param axis: axis along which to perform reduction (last axis by default)
    :return: filtered time serie
    """
    return _freq_filter(ts, si, b, axis=axis, typ="bp")


def lp(ts, si, b, axis=None):
    """
    Low-pass filter in frequency domain

    :param ts: time serie
    :param si: sampling interval in seconds
    :param b: cutout frequencies: 2 elements vector or list
    :param axis: axis along which to perform reduction (last axis by default)
    :return: filtered time serie
    """
    return _freq_filter(ts, si, b, axis=axis, typ="lp")


def hp(ts, si, b, axis=None):
    """
    High-pass filter in frequency domain

    :param ts: time serie
    :param si: sampling interval in seconds
    :param b: cutout frequencies: 2 elements vector or list
    :param axis: axis along which to perform reduction (last axis by default)
    :return: filtered time serie
    """
    return _freq_filter(ts, si, b, axis=axis, typ="hp")


def _freq_filter(ts, si, b, axis=None, typ="lp"):
    """
    Wrapper for hp/lp/bp filters
    """
    if axis is None:
        axis = ts.ndim - 1
    ns = ts.shape[axis]
    f = fscale(ns, si=si, one_sided=True)
    if typ == "bp":
        filc = _freq_vector(f, b[0:2], typ="hp") * _freq_vector(f, b[2:4], typ="lp")
    else:
        filc = _freq_vector(f, b, typ=typ)
    if axis < (ts.ndim - 1):
        filc = filc[:, np.newaxis]
    return np.real(np.fft.ifft(np.fft.fft(ts, axis=axis) * fexpand(filc, ns, axis=0), axis=axis))


def _freq_vector(f, b, typ="lp"):
    """
    Returns a frequency modulated vector for filtering

    :param f: frequency vector, uniform and monotonic
    :param b: 2 bounds array
    :return: amplitude modulated frequency vector
    """
    filc = fcn_cosine(b)(f)
    if typ.lower() in ["hp", "highpass"]:
        return filc
    elif typ.lower() in ["lp", "lowpass"]:
        return 1 - filc


def fexpand(x, ns=1, axis=None):
    """
    Reconstructs full spectrum from positive frequencies
    Works on the last dimension (contiguous in c-stored array)

    :param x: numpy.ndarray
    :param axis: axis along which to perform reduction (last axis by default)
    :return: numpy.ndarray
    """
    if axis is None:
        axis = x.ndim - 1
    # dec = int(ns % 2) * 2 - 1
    # xcomp = np.conj(np.flip(x[..., 1:x.shape[-1] + dec], axis=axis))
    ilast = int((ns + (ns % 2)) / 2)
    xcomp = np.conj(np.flip(np.take(x, np.arange(1, ilast), axis=axis), axis=axis))
    return np.concatenate((x, xcomp), axis=axis)


class WindowGenerator(object):
    """
    `wg = WindowGenerator(ns, nswin, overlap)`

    Provide sliding windows indices generator for signal processing applications.
    For straightforward spectrogram / periodogram implementation, prefer scipy methods !

    Example of implementations in test_dsp.py.
    """

    def __init__(self, ns, nswin, overlap):
        """
        :param ns: number of sample of the signal along the direction to be windowed
        :param nswin: number of samples of the window
        :return: dsp.WindowGenerator object:
        """
        self.ns = int(ns)
        self.nswin = int(nswin)
        self.overlap = int(overlap)
        self.nwin = int(np.ceil(float(ns - nswin) / float(nswin - overlap))) + 1
        self.iw = None

    @property
    def firstlast(self):
        """
        Generator that yields first and last index of windows

        :return: tuple of [first_index, last_index] of the window
        """
        self.iw = 0
        first = 0
        while True:
            last = first + self.nswin
            last = min(last, self.ns)
            yield (first, last)
            if last == self.ns:
                break
            first += self.nswin - self.overlap
            self.iw += 1

    @property
    def slice(self):
        """
        Generator that yields slices of windows

        :return: a slice of the window
        """
        for first, last in self.firstlast:
            yield slice(first, last)

    def slice_array(self, sig, axis=-1):
        """
        Provided an array or sliceable object, generator that yields
        slices corresponding to windows. Especially useful when working on memmpaps

        :param sig: array
        :param axis: (optional, -1) dimension along which to provide the slice
        :return: array slice Generator
        """
        for first, last in self.firstlast:
            yield np.take(sig, np.arange(first, last), axis=axis)

    def tscale(self, fs):
        """
        Returns the time scale associated with Window slicing (middle of window)
        :param fs: sampling frequency (Hz)
        :return: time axis scale
        """
        return np.array([(first + (last - first - 1) / 2) / fs for first, last in self.firstlast])
