def kfilt(x, collection=None, ntr_pad=0, ntr_tap=None, lagc=300, butter_kwargs=None):
    """
    Applies a butterworth filter on the 0-axis with tapering / padding
    :param x: the input array to be filtered. dimension, the filtering is considering
    axis=0: spatial dimension, axis=1 temporal dimension. (ntraces, ns)
    :param collection:
    :param ntr_pad: traces added to each side (mirrored)
    :param ntr_tap: n traces for apodizatin on each side
    :param lagc: window size for time domain automatic gain control (no agc otherwise)
    :param butter_kwargs: filtering parameters: defaults: {'N': 3, 'Wn': 0.1, 'btype': 'highpass'}
    :return:
    """
    if butter_kwargs is None:
        butter_kwargs = {'N': 3, 'Wn': 0.1, 'btype': 'highpass'}
    if collection is not None:
        xout = np.zeros_like(x)
        for c in np.unique(collection):
            sel = collection == c
            xout[sel, :] = kfilt(x=x[sel, :], ntr_pad=0, ntr_tap=None, collection=None,
                                 butter_kwargs=butter_kwargs)
        return xout
    nx, nt = x.shape

    # lateral padding left and right
    ntr_pad = int(ntr_pad)
    ntr_tap = ntr_pad if ntr_tap is None else ntr_tap
    nxp = nx + ntr_pad * 2

    # apply agc and keep the gain in handy
    if not lagc:
        xf = np.copy(x)
        gain = 1
    else:
        xf, gain = agc(x, wl=lagc, si=1.0)
    if ntr_pad > 0:
        # pad the array with a mirrored version of itself and apply a cosine taper
        xf = np.r_[np.flipud(xf[:ntr_pad]), xf, np.flipud(xf[-ntr_pad:])]
    if ntr_tap > 0:
        taper = fourier.fcn_cosine([0, ntr_tap])(np.arange(nxp))  # taper up
        taper *= 1 - fourier.fcn_cosine([nxp - ntr_tap, nxp])(np.arange(nxp))   # taper down
        xf = xf * taper[:, np.newaxis]
    sos = scipy.signal.butter(**butter_kwargs, output='sos')

    xf = scipy.signal.sosfiltfilt(sos, xf, axis=0)

    if ntr_pad > 0:
        xf = xf[ntr_pad:-ntr_pad, :]
    return xf / gain

def agc(x, wl=.5, si=.002, epsilon=1e-8):
    """
    Automatic gain control
    w_agc, gain = agc(w, wl=.5, si=.002, epsilon=1e-8)
    such as w_agc / gain = w
    :param x: seismic array (sample last dimension)
    :param wl: window length (secs)
    :param si: sampling interval (secs)
    :param epsilon: whitening (useful mainly for synthetic data)
    :return: AGC data array, gain applied to data
    """
    ns_win = int(np.round(wl / si / 2) * 2 + 1)
    w = np.hanning(ns_win)
    w /= np.sum(w)
    gain = fourier.convolve(np.abs(x), w, mode='same')
    gain += (np.sum(gain, axis=1) * epsilon / x.shape[-1])[:, np.newaxis]
    gain = 1 / gain

    return x * gain, gain

def convolve(x, w, mode='full'):
    """
    Frequency domain convolution along the last dimension (2d arrays)
    Will broadcast if a matrix is convolved with a vector
    :param x:
    :param w:
    :param mode:
    :return: convolution
    """
    nsx = x.shape[-1]
    nsw = w.shape[-1]
    ns = ns_optim_fft(nsx + nsw)
    x_ = np.concatenate((x, np.zeros([*x.shape[:-1], ns - nsx], dtype=x.dtype)), axis=-1)
    w_ = np.concatenate((w, np.zeros([*w.shape[:-1], ns - nsw], dtype=w.dtype)), axis=-1)
    xw = np.real(np.fft.irfft(np.fft.rfft(x_, axis=-1) * np.fft.rfft(w_, axis=-1), axis=-1))
    xw = xw[..., :(nsx + nsw)]  # remove 0 padding
    if mode == 'full':
        return xw
    elif mode == 'same':
        first = int(np.floor(nsw / 2)) - ((nsw + 1) % 2)
        last = int(np.ceil(nsw / 2)) + ((nsw + 1) % 2)
        return xw[..., first:-last]


def ns_optim_fft(ns):
    """
    Gets the next higher combination of factors of 2 and 3 than ns to compute efficient ffts
    :param ns:
    :return: nsoptim
    """
    p2, p3 = np.meshgrid(2 ** np.arange(25), 3 ** np.arange(15))
    sz = np.unique((p2 * p3).flatten())
    return sz[np.searchsorted(sz, ns)]


