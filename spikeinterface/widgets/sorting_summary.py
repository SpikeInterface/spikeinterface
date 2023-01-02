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
    waveform_extractor : WaveformExtractor
        The waveform extractor object.
    sparsity : dict or None
        Optional dictionary with sparsity with unit ids as keys and 
        list of channel ids as values.
    max_amplitudes_per_unit : int or None
        Maximum number of spikes per unit for plotting amplitudes,
        by default None (all spikes)
    curation : bool
        If True, manual curation is enabled, by default False
        (sortingview backend)
    unit_table_properties : list or None
        List of properties to be added to the unit table, by default None
        (sortingview backend)
    """
    possible_backends = {}

    
    def __init__(self, waveform_extractor: WaveformExtractor, unit_ids=None,
                 sparsity=None, max_amplitudes_per_unit=None, curation=False,
                 unit_table_properties=None, label_choices=None, backend=None, **backend_kwargs):
        self.check_extensions(waveform_extractor, ['correlograms', 'spike_amplitudes',
                                                   'unit_locations', 'similarity'])
        we = waveform_extractor
        sorting = we.sorting

        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()
        channel_ids = we.channel_ids
            
        if sparsity is None:
            sparsity = {u: channel_ids for u in sorting.unit_ids}
        else:
            assert all(u in sparsity for u in sorting.unit_ids), "Sparsity needs to be defined for all units!"

        # use other widgets to generate data (except for similarity)
        template_plot_data = UnitTemplatesWidget(we, unit_ids=unit_ids, sparsity=sparsity,
                                                 hide_unit_selector=True).plot_data
        ccg_plot_data = CrossCorrelogramsWidget(we, unit_ids=unit_ids, hide_unit_selector=True).plot_data
        amps_plot_data = AmplitudesWidget(we, unit_ids=unit_ids, max_spikes_per_unit=max_amplitudes_per_unit, 
                                          hide_unit_selector=True).plot_data
        locs_plot_data = UnitLocationsWidget(we, unit_ids=unit_ids, hide_unit_selector=True).plot_data
        sim_plot_data = TemplateSimilarityWidget(we, unit_ids=unit_ids).plot_data

        plot_data = dict(
            waveform_extractor=waveform_extractor,
            unit_ids=unit_ids,
            templates=template_plot_data,
            correlograms=ccg_plot_data,
            amplitudes=amps_plot_data,
            similarity=sim_plot_data,
            unit_locations=locs_plot_data,
            unit_table_properties=unit_table_properties,
            curation=curation,
            label_choices=label_choices
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

