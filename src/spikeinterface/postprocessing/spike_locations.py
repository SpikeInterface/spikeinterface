from __future__ import annotations

import numpy as np

from spikeinterface.core.job_tools import _shared_job_kwargs_doc
from spikeinterface.core.sortinganalyzer import register_result_extension
from spikeinterface.core.template_tools import get_template_extremum_channel
from spikeinterface.core.node_pipeline import SpikeRetriever
from spikeinterface.core.analyzer_extension_core import BaseSpikeVectorExtension


class ComputeSpikeLocations(BaseSpikeVectorExtension):
    """
    Localize spikes in 2D or 3D with several methods given the template.

    Parameters
    ----------
    ms_before : float, default: 0.5
        The left window, before a peak, in milliseconds
    ms_after : float, default: 0.5
        The right window, after a peak, in milliseconds
    spike_retriver_kwargs : dict
        A dictionary to control the behavior for getting the maximum channel for each spike
        This dictionary contains:

          * channel_from_template: bool, default: True
              For each spike is the maximum channel computed from template or re estimated at every spikes
              channel_from_template = True is old behavior but less acurate
              channel_from_template = False is slower but more accurate
          * radius_um: float, default: 50
              In case channel_from_template=False, this is the radius to get the true peak
          * peak_sign, default: "neg"
              In case channel_from_template=False, this is the peak sign.
    method : "center_of_mass" | "monopolar_triangulation" | "grid_convolution", default: "center_of_mass"
        The localization method to use
    method_kwargs : dict, default: dict()
        Other kwargs depending on the method.

    Returns
    -------
    spike_locations: np.array
        All locations for all spikes
    """

    extension_name = "spike_locations"
    depend_on = ["templates"]
    nodepipeline_variables = ["spike_locations"]

    def _set_params(
        self,
        ms_before=0.5,
        ms_after=0.5,
        spike_retriver_kwargs=None,
        method="center_of_mass",
        method_kwargs={},
    ):
        spike_retriver_kwargs_ = dict(
            channel_from_template=True,
            radius_um=50,
            peak_sign="neg",
        )
        if spike_retriver_kwargs is not None:
            spike_retriver_kwargs_.update(spike_retriver_kwargs)
        return super()._set_params(
            ms_before=ms_before,
            ms_after=ms_after,
            spike_retriver_kwargs=spike_retriver_kwargs_,
            method=method,
            method_kwargs=method_kwargs,
        )

    def _get_pipeline_nodes(self):
        from spikeinterface.sortingcomponents.peak_localization import get_localization_pipeline_nodes

        recording = self.sorting_analyzer.recording
        sorting = self.sorting_analyzer.sorting
        peak_sign = self.params["spike_retriver_kwargs"]["peak_sign"]
        extremum_channels_indices = get_template_extremum_channel(
            self.sorting_analyzer, peak_sign=peak_sign, outputs="index"
        )

        retriever = SpikeRetriever(
            sorting,
            recording,
            channel_from_template=True,
            extremum_channel_inds=extremum_channels_indices,
        )
        nodes = get_localization_pipeline_nodes(
            recording,
            retriever,
            method=self.params["method"],
            method_kwargs=self.params["method_kwargs"],
            ms_before=self.params["ms_before"],
            ms_after=self.params["ms_after"],
        )
        return nodes


register_result_extension(ComputeSpikeLocations)
compute_spike_locations = ComputeSpikeLocations.function_factory()
