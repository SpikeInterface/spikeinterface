import pytest
from pathlib import Path

from spikeinterface import set_global_tmp_folder
from spikeinterface.core import generate_recording

from spikeinterface.preprocessing import whiten, scale

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "preprocessing"
else:
    cache_folder = Path("cache_folder") / "preprocessing"

set_global_tmp_folder(cache_folder)


def test_whiten():
    rec = generate_recording()

    rec2 = whiten(rec)
    rec2.save(verbose=False)

    # test dtype
    rec_int = scale(rec2, dtype="int16")
    rec3 = whiten(rec_int, dtype="float16")
    assert rec3.get_dtype() == "float16"

    # ~ import matplotlib.pyplot as plt
    # ~ from spikeinterface.widgets import plot_timeseries
    # ~ fig, ax = plt.subplots()
    # ~ ax.plot(rec.get_traces(segment_index=0)[:, 0], color='g')
    # ~ ax.plot(rec2.get_traces(segment_index=0)[:, 0], color='r')
    # ~ plt.show()


if __name__ == '__main__':
    test_whiten()
