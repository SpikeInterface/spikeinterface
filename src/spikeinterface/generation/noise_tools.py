import numpy as np

from spikeinterface.core.generate import NoiseGeneratorRecording


def generate_noise(
    probe, sampling_frequency, durations, dtype="float32", noise_levels=15.0, spatial_decay=None, seed=None
):
    """

    Parameters
    ----------
    probe: Probe
        A probe object.
    sampling_frequency: float
        Sampling frequency
    durations: list of float
        Durations
    dtype: np.dtype
        Dtype
    noise_levels: float | np.array | tuple
        If scalar same noises on all channels.
        If array then per channels noise level.
        If tuple, then this represent the range.

    seed: None | int
        The seed for random generator.

    Returns
    -------
    noise: NoiseGeneratorRecording
        A lazy noise generator recording.
    """

    num_channels = probe.get_contact_count()
    locs = probe.contact_positions
    distances = np.linalg.norm(locs[:, np.newaxis] - locs[np.newaxis, :], axis=2)

    if spatial_decay is None:
        cov_matrix = None
    else:
        cov_matrix = np.exp(-distances / spatial_decay)

    if isinstance(noise_levels, tuple):
        rng = np.random.default_rng(seed=seed)
        lim0, lim1 = noise_levels
        noise_levels = rng.random(num_channels) * (lim1 - lim0) + lim0
    elif np.isscalar(noise_levels):
        noise_levels = np.full(shape=(num_channels), fill_value=noise_levels)
    elif isinstance(noise_levels, (list, np.ndarray)):
        noise_levels = np.asarray(noise_levels)
        assert noise_levels.size == num_channels
    else:
        raise ValueError("generate_noise: wrong noise_levels type")

    noise = NoiseGeneratorRecording(
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        durations=durations,
        dtype=dtype,
        strategy="on_the_fly",
        noise_levels=noise_levels,
        cov_matrix=cov_matrix,
        seed=seed,
    )

    return noise
