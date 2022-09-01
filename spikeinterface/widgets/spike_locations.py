import numpy as np
from typing import Union

from .base import BaseWidget
from .utils import get_unit_colors
from ..core.waveform_extractor import WaveformExtractor
from ..core.basesorting import BaseSorting
from ..postprocessing import compute_spike_locations


class SpikeLocationsWidget(BaseWidget):
    """
    Plots spike locations.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The object to compute/get spike locations from
    unit_ids: list
        List of unit ids.
    with_channel_ids: bool False default
        Add channel ids text on the probe
    unit_colors :  dict or None
        If given, a dictionary with unit ids as keys and colors as values
    hide_unit_selector : bool
        For sortingview backend, if True the unit selector is not displayed
    """

    possible_backends = {}

    def __init__(
        self,
        waveform_extractor: WaveformExtractor,
        unit_ids=None,
        segment_index=None,
        with_channel_ids=False,
        unit_colors=None,
        hide_unit_selector=False,
        plot_all_units=True,
        backend=None,
        **backend_kwargs
    ):
        self.check_extensions(waveform_extractor, "spike_locations")
        slc = waveform_extractor.load_extension("spike_locations")
        spike_locations = slc.get_data(outputs="by_unit")

        recording = waveform_extractor.recording
        sorting = waveform_extractor.sorting

        channel_ids = recording.channel_ids
        channel_locations = recording.get_channel_locations()
        probegroup = recording.get_probegroup()

        if sorting.get_num_segments() > 1:
            assert (
                segment_index is not None
            ), "Specify segment index for multi-segment object"
        else:
            segment_index = 0

        if unit_colors is None:
            unit_colors = get_unit_colors(sorting)

        if unit_ids is None:
            unit_ids = sorting.unit_ids

        plot_data = dict(
            sorting=sorting,
            all_unit_ids=sorting.unit_ids,
            spike_locations=spike_locations[segment_index],
            segment_index=segment_index,
            unit_ids=unit_ids,
            channel_ids=channel_ids,
            unit_colors=unit_colors,
            channel_locations=channel_locations,
            probegroup_dict=probegroup.to_dict(),
            with_channel_ids=with_channel_ids,
            hide_unit_selector=hide_unit_selector,
            plot_all_units=plot_all_units,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)


def estimate_axis_lims(spike_locations, quantile=0.02):
    # set proper axis limits
    all_locs = np.concatenate(list(spike_locations.values()))
    xlims = np.quantile(all_locs["x"], [quantile, 1 - quantile])
    ylims = np.quantile(all_locs["y"], [quantile, 1 - quantile])

    return xlims, ylims


