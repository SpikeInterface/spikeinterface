import numpy as np
from warnings import warn

from .base import BaseWidget, to_attr
from .utils import get_unit_colors


from ..core.template_tools import get_template_extremum_amplitude


class UnitDepthsWidget(BaseWidget):
    """
    Plot unit depths

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The input waveform extractor
    unit_colors :  dict or None, default: None
        If given, a dictionary with unit ids as keys and colors as values
    depth_axis : int, default: 1
        The dimension of unit_locations that is depth
    peak_sign: "neg" | "pos" | "both", default: "neg"
        Sign of peak for amplitudes
    """

    def __init__(
        self, waveform_extractor, unit_colors=None, depth_axis=1, peak_sign="neg", backend=None, **backend_kwargs
    ):
        we = waveform_extractor
        unit_ids = we.sorting.unit_ids

        if unit_colors is None:
            unit_colors = get_unit_colors(we.sorting)

        colors = [unit_colors[unit_id] for unit_id in unit_ids]

        self.check_extensions(waveform_extractor, "unit_locations")
        ulc = waveform_extractor.load_extension("unit_locations")
        unit_locations = ulc.get_data(outputs="numpy")
        unit_depths = unit_locations[:, depth_axis]

        unit_amplitudes = get_template_extremum_amplitude(we, peak_sign=peak_sign)
        unit_amplitudes = np.abs([unit_amplitudes[unit_id] for unit_id in unit_ids])

        num_spikes = np.array(list(we.sorting.count_num_spikes_per_unit().values()))

        plot_data = dict(
            unit_depths=unit_depths,
            unit_amplitudes=unit_amplitudes,
            num_spikes=num_spikes,
            unit_colors=unit_colors,
            colors=colors,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        ax = self.ax
        size = dp.num_spikes / max(dp.num_spikes) * 120
        ax.scatter(dp.unit_amplitudes, dp.unit_depths, color=dp.colors, s=size)

        ax.set_aspect(3)
        ax.set_xlabel("amplitude")
        ax.set_ylabel("depth [um]")
        ax.set_xlim(0, max(dp.unit_amplitudes) * 1.2)
