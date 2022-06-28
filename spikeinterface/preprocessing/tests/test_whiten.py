import pytest
from pathlib import Path

from spikeinterface import set_global_tmp_folder
from spikeinterface.core.testing_tools import generate_recording

from spikeinterface.preprocessing import whiten

if hasattr(pytest, "global_test_folder"):
    cache_folder = pytest.global_test_folder / "toolkit"
else:
    cache_folder = Path("cache_folder") / "toolkit"

set_global_tmp_folder(cache_folder)

def test_normalize_by_quantile():
    rec = generate_recording()

    rec2 = whiten(rec)
    rec2.save(verbose=False)

    # ~ import matplotlib.pyplot as plt
    # ~ from spikeinterface.widgets import plot_timeseries
    # ~ fig, ax = plt.subplots()
    # ~ ax.plot(rec.get_traces(segment_index=0)[:, 0], color='g')
    # ~ ax.plot(rec2.get_traces(segment_index=0)[:, 0], color='r')
    # ~ plt.show()


if __name__ == '__main__':
    test_normalize_by_quantile()
