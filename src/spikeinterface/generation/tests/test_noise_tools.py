import probeinterface

from spikeinterface.generation import (
    generate_noise,
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


if __name__ == "__main__":
    test_generate_noise()
