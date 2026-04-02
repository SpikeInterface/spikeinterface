import numpy as np
import probeinterface
import pytest

from spikeinterface.generation import (
    generate_noise,
    NoiseGeneratorRecording,
)


def test_generate_noise():
    probe = probeinterface.generate_multi_columns_probe()

    noise = generate_noise(
        probe,
        sampling_frequency=25000.0,
        durations=[10.0],
        dtype="float32",
        noise_levels=15.0,
        spatial_decay=None,
        seed=2205,
    )

    noise = generate_noise(
        probe,
        sampling_frequency=25000.0,
        durations=[10.0],
        dtype="float32",
        noise_levels=(12.0, 18.0),
        spatial_decay=20.0,
        seed=2205,
    )

    # print(noise)

    # from spikeinterface.widgets import plot_traces
    # plot_traces(noise, backend="ephyviewer")

    # import matplotlib.pyplot as plt
    # fig,  ax = plt.subplots()
    # im = ax.matshow(noise._kwargs["cov_matrix"])
    # fig.colorbar(im)
    # plt.show()


@pytest.mark.parametrize("duration", [1.0, 2.0, 2.2])
@pytest.mark.parametrize("strategy", ["tile_precomputed", "on_the_fly"])
def test_noise_generator_temporal(strategy, duration):
    psdlen = 25
    kdomain = np.linspace(0.0, 10.0, psdlen)
    fake_psd = (kdomain + 0.1) * np.exp(-kdomain)
    # this ensures std dev of output ~= 1
    fake_psd /= np.sqrt((fake_psd**2).mean())

    # Test that the recording has the correct size in shape
    sampling_frequency = 30000  # Hz
    durations = [duration]
    dtype = np.dtype("float32")
    num_channels = 2
    seed = 0

    rec = NoiseGeneratorRecording(
        num_channels=num_channels,
        sampling_frequency=sampling_frequency,
        durations=durations,
        dtype=dtype,
        seed=seed,
        spectral_density=fake_psd,
        strategy=strategy,
    )

    # check output matches at different chunks
    full_traces = rec.get_traces()
    end_frame = rec.get_num_frames()
    for t0 in [0, 100]:
        for t1 in [end_frame, end_frame - 100]:
            chk = rec.get_traces(0, t0, t1)
            chk0 = full_traces[t0:t1]
            np.testing.assert_array_equal(chk, chk0)

    np.testing.assert_allclose(full_traces.std(), 1.0, rtol=0.02)

    # re-estimate the psd from the result
    # it will not be perfect, okay!
    n = 2 * psdlen - 1
    snips = full_traces[: n * (full_traces.shape[0] // n)]
    snips = snips.reshape(-1, n, snips.shape[-1])
    psd = np.fft.rfft(snips, n=n, axis=1, norm="ortho")
    psd = np.sqrt(np.square(np.abs(psd)).mean(axis=(0, 2)))

    sample_size = snips.shape[0] * snips.shape[2]
    standard_error = 1.0 / np.sqrt(sample_size)

    # accuracy is good at low freqs
    np.testing.assert_allclose(
        psd[1 : psdlen // 3],
        fake_psd[1 : psdlen // 3],
        atol=3 * standard_error,
        rtol=0.1,
    )
    np.testing.assert_allclose(psd, fake_psd, atol=0.5)


if __name__ == "__main__":
    test_generate_noise()
