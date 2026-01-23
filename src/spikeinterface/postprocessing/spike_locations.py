from __future__ import annotations

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
    peak_sign : "neg" | "pos" | "both", default: "neg" 
        The peak sign to use when looking for the template extremum channel.
    spike_retriever_kwargs : dict
        Arguments to control the spike retriever behavior. See 
        `spikeinterface.sortingcomponents.peak_localization.SpikeRetriever`.
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
    need_backward_compatibility_on_load = True

    def _handle_backward_compatibility_on_load(self):
        # For backwards compatibility - this renames spike_retriver_kwargs to spike_retriever_kwargs
        if "spike_retriver_kwargs" in self.params:
            self.params['peak_sign'] = self.params['spike_retriver_kwargs'].get('peak_sign', 'neg')
            self.params["spike_retriever_kwargs"] = self.params.pop("spike_retriver_kwargs")

    def _set_params(
        self,
        ms_before=0.5,
        ms_after=0.5,
        peak_sign="neg",
        spike_retriever_kwargs=None,
        method="center_of_mass",
        method_kwargs={},
    ):
        if spike_retriever_kwargs is None:
            spike_retriever_kwargs = {}
        return super()._set_params(
            ms_before=ms_before,
            ms_after=ms_after,
            peak_sign=peak_sign,
            spike_retriever_kwargs=spike_retriever_kwargs,
            method=method,
            method_kwargs=method_kwargs,
        )

    def _get_pipeline_nodes(self):
        from spikeinterface.sortingcomponents.peak_localization import get_localization_pipeline_nodes

        recording = self.sorting_analyzer.recording
        sorting = self.sorting_analyzer.sorting
        peak_sign = self.params["peak_sign"]
        extremum_channels_indices = get_template_extremum_channel(
            self.sorting_analyzer, peak_sign=peak_sign, outputs="index"
        )

        retriever_kwargs = {"channel_from_template": True,
                            "extremum_channel_inds": extremum_channels_indices,
                            **self.params["spike_retriever_kwargs"]}
        retriever = SpikeRetriever(sorting, recording, **retriever_kwargs)
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
