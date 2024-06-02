import numpy as np
from spikeinterface import NumpyRecording

from spikeinterface.preprocessing import phase_shift


def create_shifted_channel():
    duration = 5.0
    sr_h = 10000.0
    times_h = np.arange(0, duration, 1 / sr_h)
    freq1 = 2.5
    freq2 = 8.5
    sig_h = np.sin(2 * np.pi * freq1 * times_h) + np.sin(2 * np.pi * freq2 * times_h)
    # ~ noise = np.random.randn(sig_h.size)
    # ~ sig_h += noise * 0.02

    ratio = 10
    sr = sr_h / ratio
    times = times_h[0::ratio]
    delay_sample = 4
    sig0 = sig_h[0::ratio]
    sig1 = sig_h[delay_sample::ratio]

    inter_sample_shift = [0.0, delay_sample / ratio]

    traces = np.stack([sig0, sig1], axis=1)
    traces *= 1000
    return traces, sr, inter_sample_shift


def test_phase_shift():
    traces, sampling_frequency, inter_sample_shift = create_shifted_channel()
    # traces = (traces * 1000).astype('int16')

    # ~ print(sampling_frequency)

    for dtype in ("float64", "float32", "int16"):
        rec = NumpyRecording([traces.astype(dtype)], sampling_frequency)
        rec.set_property("inter_sample_shift", inter_sample_shift)
        original_traces = rec.get_traces(end_frame=10)

        for margin_ms in (10.0, 30.0, 40.0):
            for chunk_size in (100, 500, 1000, 2000):
                rec2 = phase_shift(rec, margin_ms=margin_ms)
                assert rec2.dtype == rec.dtype

                # save by chunk rec3 is the cached version
                rec3 = rec2.save(format="memory", chunk_size=chunk_size, n_jobs=1, progress_bar=False)

                traces2 = rec2.get_traces()
                assert traces2.dtype == original_traces.dtype
                traces3 = rec3.get_traces()
                assert traces3.dtype == original_traces.dtype

                traces_slice = rec3.get_traces(channel_ids=[rec3.channel_ids[0]])
                assert traces_slice.shape[1] == 1

                # error between full and chunked
                error_mean = np.sqrt(np.mean((traces2 - traces3) ** 2))
                error_max = np.sqrt(np.max((traces2 - traces3) ** 2))
                rms = np.sqrt(np.mean(traces**2))

                # this will never be possible:
                #      assert np.allclose(traces2, traces3)
                # so we check that the diff between chunk processing and not chunked is small
                # ~ print()
                # ~ print(dtype, margin_ms, chunk_size)
                # ~ print(error_mean, rms, error_mean / rms)
                # ~ print(error_max, rms, error_max / rms)
                assert error_mean / rms < 0.001
                assert error_max / rms < 0.02

                # ~ import matplotlib.pyplot as plt
                # ~ fig, axs = plt.subplots(nrows=3, sharex=True)
                # ~ ax = axs[0]
                # ~ ax.set_title(f'margin_ms{margin_ms} chunk_size{chunk_size} {error_max/rms:.6f} {error_mean/rms:.6f}')
                # ~ ax.plot(traces[:, 0], color='r', label='no delay')
                # ~ ax.plot(traces[:, 1], color='b', label='delay')
                # ~ ax.plot(traces2[:, 1], color='c', ls='--', label='shift no chunk')
                # ~ ax.plot(traces3[:, 1], color='g', ls='--', label='shift no chunked')
                # ~ ax = axs[1]
                # ~ ax.plot(traces2[:, 1] - traces3[:, 1], color='k')
                # ~ ax = axs[2]
                # ~ ax.plot(traces2[:, 1] - traces[:, 0], color='c')
                # ~ ax.plot(traces3[:, 1] - traces[:, 0], color='g')
                # ~ plt.show()

    # ~ import matplotlib.pyplot as plt
    # ~ import spikeinterface.full as si
    # ~ si.plot_traces(rec, segment_index=0, time_range=[0, 10])
    # ~ si.plot_traces(rec2, segment_index=0, time_range=[0, 10])
    # ~ si.plot_traces(rec3, segment_index=0, time_range=[0, 10])
    # ~ plt.show()


if __name__ == "__main__":
    test_phase_shift()
