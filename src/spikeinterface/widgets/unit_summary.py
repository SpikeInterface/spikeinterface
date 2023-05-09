import numpy as np
from typing import Union

from .base import BaseWidget
from .utils import get_unit_colors


from .unit_locations import UnitLocationsWidget
from .unit_waveforms import UnitWaveformsWidget
from .unit_waveforms_density_map import UnitWaveformDensityMapWidget
from .autocorrelograms import AutoCorrelogramsWidget
from .amplitudes import AmplitudesWidget


class UnitSummaryWidget(BaseWidget):
    """
    Plot a unit summary.
    
    If amplitudes are alreday computed they are displayed.
    
    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor object
    unit_id: into or str
        The unit id to plot the summary of
    unit_colors :  dict or None
        If given, a dictionary with unit ids as keys and colors as values
    sparsity : ChannelSparsity or None
        Optional ChannelSparsity to apply.
        If WaveformExtractor is already sparse, the argument is ignored
    """
    possible_backends = {}

    def __init__(self, waveform_extractor, unit_id, unit_colors=None,
                 sparsity=None, radius_um=100, backend=None, **backend_kwargs):
        
        we = waveform_extractor

        if unit_colors is None:
            unit_colors = get_unit_colors(we.sorting)
                
        if we.is_extension('unit_locations'):
            plot_data_unit_locations = UnitLocationsWidget(we, unit_ids=[unit_id], 
                                                           unit_colors=unit_colors, plot_legend=False).plot_data
            unit_locations = waveform_extractor.load_extension("unit_locations").get_data(outputs="by_unit")
            unit_location = unit_locations[unit_id]
        else:
            plot_data_unit_locations = None
            unit_location = None

        plot_data_waveforms = UnitWaveformsWidget(we, unit_ids=[unit_id], unit_colors=unit_colors,
                                                  plot_templates=True, same_axis=True, plot_legend=False,
                                                  sparsity=sparsity).plot_data
        
        plot_data_waveform_density = UnitWaveformDensityMapWidget(we, unit_ids=[unit_id], unit_colors=unit_colors,
                                                                  use_max_channel=True, plot_templates=True,
                                                                  same_axis=False).plot_data
        
        if we.is_extension('correlograms'):
            plot_data_acc = AutoCorrelogramsWidget(we, unit_ids=[unit_id], unit_colors=unit_colors,).plot_data
        else:
            plot_data_acc = None

        # use other widget to plot data
        if we.is_extension('spike_amplitudes'):
            plot_data_amplitudes = AmplitudesWidget(we, unit_ids=[unit_id], unit_colors=unit_colors,
                                                    plot_legend=False, plot_histograms=True).plot_data
        else:
            plot_data_amplitudes = None

        
        plot_data = dict(
            unit_id=unit_id,
            unit_location=unit_location,
            plot_data_unit_locations=plot_data_unit_locations,
            plot_data_waveforms=plot_data_waveforms,
            plot_data_waveform_density=plot_data_waveform_density,
            plot_data_acc=plot_data_acc,
            plot_data_amplitudes=plot_data_amplitudes,

        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)



