import numpy as np
from warnings import warn

from .base import BaseWidget
from .utils import get_unit_colors


from ..core.template_tools import get_template_extremum_amplitude



class UnitDepthsWidget(BaseWidget):
    """
    Plot unit depths

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The input waveform extractor
    unit_colors :  dict or None
        If given, a dictionary with unit ids as keys and colors as values
    depth_axis: int default 1
        Which dimension of unit_locations is depth. 1 by defaults
    peak_sign: str (neg/pos/both)
        Sign of peak for amplitudes.
    """
    possible_backends = {}

    
    def __init__(self, waveform_extractor, unit_colors=None, depth_axis=1,
                peak_sign='neg',
                 backend=None, **backend_kwargs):

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

        num_spikes = np.array(list(we.sorting.get_total_num_spikes().values()))

        plot_data = dict(
                unit_depths=unit_depths,
                unit_amplitudes=unit_amplitudes,
                num_spikes=num_spikes,
                unit_colors=unit_colors,
                colors=colors,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

