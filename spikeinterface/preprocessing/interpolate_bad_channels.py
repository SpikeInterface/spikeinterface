
# TODO: this assumes 20 um channel separation.
#       a) is channel separated always same in x, y across probes
#       b) is channel distance always stored in um
#       c) should have user specify or calculate directly from probe?

# removed GPU option (cupy install not assumed, add back in?)

def interpolate_bad_channels(data, channel_labels=None, x=None, y=None, p=1.3, kriging_distance_um=20, gpu=False):
    """
    Interpolate the channel labeled as bad channels using linear interpolation.
    This is based on the distance from the bad channel, as determined from x,y
    channel coordinates. The weights applied to neighbouring channels come
    from an exponential decay function.

    Details of the interpolation function (Olivier Winter) used in the IBL pipeline
    can be found at:

    International Brain Laboratory et al. (2022). Spike sorting pipeline for the
    International Brain Laboratory. https://www.internationalbrainlab.com/repro-ephys

    Parameters
    ----------

    data: (num_channels x num_samples) numpy array

    bad_channel_indexes: numpy array, indexes of the bad channels to interpolate.

    x: numpy array of channel x coordinates.

    y: numpy array of channel y coordinates.

    p: exponent of the Gaussian kernel. Determines rate of decay
       for distance weightings.

    kriging_distance_um: distance between sequential channels in um.
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