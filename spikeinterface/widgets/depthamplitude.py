import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget

from ..toolkit import get_template_extremum_channel, get_template_extremum_amplitude
from .utils import get_unit_colors


class UnitsDepthAmplitudeWidget(BaseWidget):
    def __init__(self, waveform_extractor, peak_sign='neg', depth_axis=1,
                 unit_colors=None, figure=None, ax=None):
        BaseWidget.__init__(self, figure, ax)

        self.we = waveform_extractor
        self.peak_sign = peak_sign
        self.depth_axis = depth_axis
        if unit_colors is None:
            unit_colors = get_unit_colors(self.we.sorting)
        self.unit_colors = unit_colors

    def plot(self):
        ax = self.ax
        we = self.we
        unit_ids = we.sorting.unit_ids

        channels_index = get_template_extremum_channel(we, peak_sign=self.peak_sign, outputs='index')
        probe = we.recording.get_probe()

        channel_depth = probe.contact_positions[:, self.depth_axis]
        unit_depth = [channel_depth[channels_index[unit_id]] for unit_id in unit_ids]

        unit_amplitude = get_template_extremum_amplitude(we, peak_sign=self.peak_sign)
        unit_amplitude = np.abs([unit_amplitude[unit_id] for unit_id in unit_ids])

        colors = [self.unit_colors[unit_id] for unit_id in unit_ids]

        num_spikes = np.zeros(len(unit_ids))
        for i, unit_id in enumerate(unit_ids):
            for segment_index in range(we.sorting.get_num_segments()):
                st = we.sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
                num_spikes[i] += st.size

        size = num_spikes / max(num_spikes) * 120
        ax.scatter(unit_amplitude, unit_depth, color=colors, s=size)

        ax.set_aspect(3)
        ax.set_xlabel('amplitude')
        ax.set_ylabel('depth [um]')
        ax.set_xlim(0, max(unit_amplitude) * 1.2)


def plot_units_depth_vs_amplitude(*args, **kwargs):
    W = UnitsDepthAmplitudeWidget(*args, **kwargs)
    W.plot()
    return W


plot_units_depth_vs_amplitude.__doc__ = UnitsDepthAmplitudeWidget.__doc__
