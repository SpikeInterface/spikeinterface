import numpy as np

from .base import BaseWidget, define_widget_function_from_class

from .amplitudes import AmplitudesWidget
from .crosscorrelograms import CrossCorrelogramsWidget
from .template_similarity import TemplateSimilarityWidget
from .unit_locations import UnitLocationsWidget
from .unit_templates import UnitTemplatesWidget


from ..core import WaveformExtractor
from ..postprocessing import get_template_channel_sparsity, compute_template_similarity


class SortingSummaryWidget(BaseWidget):
    """
    Plots spike sorting summary
    
    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor object.
    sparsity: dict or None
        Optional dictionary with sparsity with unit ids as keys and 
        list of channel ids as values.
    sparsity_kwargs: dict or None
        Optional dictionary to compute sparsity 
        (e.g. dict("method": "radius", "radius_um": 100)).
        See: `si.postprocessing.get_template_channel_sparsity()`
    """
    possible_backends = {}

    
    def __init__(self, waveform_extractor: WaveformExtractor, unit_ids=None,
                 sparsity=None, sparsity_kwargs=None,
                 backend=None, **backend_kwargs):
        self.check_extensions(waveform_extractor, ['correlograms', 'spike_amplitudes',
                                                  'unit_locations', 'similarity'])
        we = waveform_extractor
        recording = we.recording
        sorting = we.sorting

        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()
        channel_ids = recording.channel_ids
            
        if sparsity is None:
            if sparsity_kwargs is not None:
                sparsity = get_template_channel_sparsity(we, **sparsity_kwargs)
            else:
                sparsity = {u: channel_ids for u in sorting.unit_ids}
        else:
            assert all(u in sparsity for u in sorting.unit_ids), "Sparsity needs to be defined for all units!"

        # use other widgets to generate data (except for similarity)
        template_plot_data = UnitTemplatesWidget(we, unit_ids=unit_ids, sparsity=sparsity).plot_data
        ccg_plot_data = CrossCorrelogramsWidget(we, unit_ids=unit_ids, hide_unit_selector=True).plot_data
        amps_plot_data = AmplitudesWidget(we, unit_ids=unit_ids, hide_unit_selector=True).plot_data
        locs_plot_data = UnitLocationsWidget(we, unit_ids=unit_ids, hide_unit_selector=True).plot_data
        sim_plot_data = TemplateSimilarityWidget(we, unit_ids=unit_ids).plot_data
        
        plot_data = dict(
            unit_ids=unit_ids,
            templates=template_plot_data,
            correlograms=ccg_plot_data,
            amplitudes=amps_plot_data,
            similarity=sim_plot_data,
            unit_locations=locs_plot_data
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

