import numpy as np

from .base import BaseWidget, define_widget_function_from_class
from .widget_list import AutoCorrelogramsWidget, CrossCorrelogramsWidget, UnitWaveformsWidget, AmplitudeTimeseriesWidget
from ..core import WaveformExtractor
from ..postprocessing import get_template_channel_sparsity, compute_template_similarity


class SortingSummaryWidget(BaseWidget):
    """
    Plots spike sorting summary
    
    Parameters
    ----------
    waveform_extractor: WaveformExtractor
    """
    possible_backends = {}

    
    def __init__(self, waveform_extractor: WaveformExtractor, unit_ids=None,
                 sparsity=None, sparsity_kwargs={}, 
                 correlograms_kwargs=None, amplitudes_kwargs=None, 
                 similarity_kwargs=None, job_kwargs=None,
                 # compute_pca_kwargs=None, localization_kwargs=None,
                 backend=None, **backend_kwargs):
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

        job_kwargs = job_kwargs if job_kwargs is not None else {}
        
        # use other widgets to generate data (except for similarity)
        waveforms_plot_data = UnitWaveformsWidget(we, unit_ids=unit_ids, sparsity=sparsity).plot_data
        
        correlograms_kwargs = correlograms_kwargs if correlograms_kwargs is not None else {}
        ccg_plot_data = CrossCorrelogramsWidget(we, unit_ids=unit_ids, hide_unit_selector=True,
                                                **correlograms_kwargs).plot_data
        
        amplitudes_kwargs = amplitudes_kwargs if amplitudes_kwargs is not None else {}
        amplitudes_kwargs = amplitudes_kwargs.update(job_kwargs)
        amps_plot_data = AmplitudeTimeseriesWidget(we, unit_ids=unit_ids, compute_kwargs=amplitudes_kwargs,
                                                   hide_unit_selector=True).plot_data
        
        if we.is_extension("similarity"):
            ccc = we.load_extension("similarity")
            similarity = ccc.get_data()
            
        else:
            similarity_kwargs = similarity_kwargs if similarity_kwargs is not None else {}
            similarity = compute_template_similarity(we, **similarity_kwargs)
        unit_indices = sorting.ids_to_indices(unit_ids)
        similarity = similarity[unit_indices][unit_indices]
        similarity_plot_data = dict(unit_ids=unit_ids, similarity=similarity)
        
        plot_data = dict(
            unit_ids=unit_ids,
            waveforms=waveforms_plot_data,
            correlograms=ccg_plot_data,
            amplitudes=amps_plot_data,
            similarity=similarity_plot_data
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)


plot_sorting_summary = define_widget_function_from_class(SortingSummaryWidget, "plot_sorting_summary")
