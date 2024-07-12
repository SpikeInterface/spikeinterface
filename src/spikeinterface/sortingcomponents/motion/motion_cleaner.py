import numpy as np

# TODO this need a full rewrite with motion object


def clean_motion_vector(motion, temporal_bins, bin_duration_s, speed_threshold=30, sigma_smooth_s=None):
    """
    Simple machinery to remove spurious fast bump in the motion vector.
    Also can apply a smoothing.


    Arguments
    ---------
    motion: numpy array 2d
        Motion estimate in um.
    temporal_bins: numpy.array 1d
        temporal bins (bin center)
    bin_duration_s: float
        bin duration in second
    speed_threshold: float (units um/s)
        Maximum speed treshold between 2 bins allowed.
        Expressed in um/s
    sigma_smooth_s: None or float
        Optional smooting gaussian kernel.

    Returns
    -------
    corr : tensor


    """
    motion_clean = motion.copy()

    # STEP 1 :
    #  * detect long plateau or small peak corssing the speed thresh
    #  * mask the period and interpolate
    for i in range(motion.shape[1]):
        one_motion = motion_clean[:, i]
        speed = np.diff(one_motion, axis=0) / bin_duration_s
        (inds,) = np.nonzero(np.abs(speed) > speed_threshold)
        inds += 1
        if inds.size % 2 == 1:
            # more compicated case: number of of inds is odd must remove first or last
            # take the smallest duration sum
            inds0 = inds[:-1]
            inds1 = inds[1:]
            d0 = np.sum(inds0[1::2] - inds0[::2])
            d1 = np.sum(inds1[1::2] - inds1[::2])
            if d0 < d1:
                inds = inds0
        mask = np.ones(motion_clean.shape[0], dtype="bool")
        for i in range(inds.size // 2):
            mask[inds[i * 2] : inds[i * 2 + 1]] = False
        import scipy.interpolate

        f = scipy.interpolate.interp1d(temporal_bins[mask], one_motion[mask])
        one_motion[~mask] = f(temporal_bins[~mask])

    # Step 2 : gaussian smooth
    if sigma_smooth_s is not None:
        half_size = motion_clean.shape[0] // 2
        if motion_clean.shape[0] % 2 == 0:
            # take care of the shift
            bins = (np.arange(motion_clean.shape[0]) - half_size + 1) * bin_duration_s
        else:
            bins = (np.arange(motion_clean.shape[0]) - half_size) * bin_duration_s
        smooth_kernel = np.exp(-(bins**2) / (2 * sigma_smooth_s**2))
        smooth_kernel /= np.sum(smooth_kernel)
        smooth_kernel = smooth_kernel[:, None]
        motion_clean = scipy.signal.fftconvolve(motion_clean, smooth_kernel, mode="same", axes=0)

    return motion_clean
