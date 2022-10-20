def interpolate_bad_channels(data, channel_labels=None, x=None, y=None, p=1.3, kriging_distance_um=20, gpu=False):
    """
    Interpolate the channel labeled as bad channels using linear interpolation.
    The weights applied to neighbouring channels come from an exponential decay function
    :param data: (nc, ns) np.ndarray
    :param channel_labels; (nc) np.ndarray: 0: channel is good, 1: dead, 2:noisy, 3: out of the brain
    :param x: channel x-coordinates, np.ndarray
    :param y: channel y-coordinates, np.ndarray
    :param p:
    :param kriging_distance_um:
    :param gpu: bool
    :return:
    """
    if gpu:
        import cupy as gp
    else:
        gp = np

    # from ibllib.plots.figures import ephys_bad_channels
    # ephys_bad_channels(x, 30000, channel_labels[0], channel_labels[1])

    # we interpolate only noisy channels or dead channels (0: good), out of the brain channels are left
    bad_channels = gp.where(np.logical_or(channel_labels == 1, channel_labels == 2))[0]
    for i in bad_channels:
        # compute the weights to apply to neighbouring traces
        offset = gp.abs(x - x[i] + 1j * (y - y[i]))
        weights = gp.exp(-(offset / kriging_distance_um) ** p)
        weights[bad_channels] = 0
        weights[weights < 0.005] = 0
        weights = weights / gp.sum(weights)
        imult = gp.where(weights > 0.005)[0]
        if imult.size == 0:
            data[i, :] = 0
            continue
        data[i, :] = gp.matmul(weights[imult], data[imult, :])
    # from viewephys.gui import viewephys
    # f = viewephys(data.T, fs=1/30, h=h, title='interp2')
    return