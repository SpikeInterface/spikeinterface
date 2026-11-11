import numpy as np
import pytest

from spikeinterface.core import NumpyRecording
from spikeinterface.widgets.inter_session_alignment import SessionAlignmentWidget


@pytest.mark.parametrize("num_sessions", [5, 8])
def test_uncorrected_sessions_wrap_to_second_row(num_sessions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recording = NumpyRecording([np.zeros((100, 2), dtype="float32")], sampling_frequency=1000)
    peaks = np.array(
        [(10, 0, -1.0), (20, 0, -2.0)],
        dtype=[("sample_index", "int64"), ("segment_index", "int64"), ("amplitude", "float64")],
    )
    locations = np.array([(10.0,), (20.0,)], dtype=[("y", "float64")])
    widget = SessionAlignmentWidget(
        [recording] * num_sessions,
        [peaks] * num_sessions,
        [locations] * num_sessions,
        session_histogram_list=[np.ones(4)] * num_sessions,
        spatial_bin_centers=np.arange(4),
    )
    try:
        assert len(widget.figure.axes) == num_sessions + 1
        for index, ax in enumerate(widget.figure.axes[:-1]):
            subplot = ax.get_subplotspec()
            assert (subplot.rowspan.start, subplot.colspan.start) == divmod(index, 4)
        assert widget.figure.axes[-1].get_subplotspec().rowspan.start == 2
    finally:
        plt.close(widget.figure)
