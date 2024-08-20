from __future__ import annotations

import numpy as np

from spikeinterface.core.job_tools import _shared_job_kwargs_doc, fix_job_kwargs
from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension
from spikeinterface.core.template_tools import get_template_extremum_channel

from spikeinterface.core.sorting_tools import spike_vector_to_indices

from spikeinterface.core.node_pipeline import SpikeRetriever, run_node_pipeline


class ComputeSpikeLocations(AnalyzerExtension):
    """
    Localize spikes in 2D or 3D with several methods given the template.

    Parameters
    ----------
    sorting_analyzer: SortingAnalyzer
        A SortingAnalyzer object
    ms_before : float, default: 0.5
        The left window, before a peak, in milliseconds
    ms_after : float, default: 0.5
        The right window, after a peak, in milliseconds
    spike_retriver_kwargs: dict
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
    outputs : "concatenated" | "by_unit", default: "concatenated"
        The output format
    {}

    Returns
    -------
    spike_locations: np.array
        All locations for all spikes
    """

    extension_name = "spike_locations"
    depend_on = ["templates"]
    need_recording = True
    use_nodepipeline = True
    nodepipeline_variables = ["spike_locations"]
    need_job_kwargs = True

    def __init__(self, sorting_analyzer):
        AnalyzerExtension.__init__(self, sorting_analyzer)

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
        params = dict(
            ms_before=ms_before,
            ms_after=ms_after,
            spike_retriver_kwargs=spike_retriver_kwargs_,
            method=method,
            method_kwargs=method_kwargs,
        )
        return params

    def _select_extension_data(self, unit_ids):
        old_unit_ids = self.sorting_analyzer.unit_ids
        unit_inds = np.flatnonzero(np.isin(old_unit_ids, unit_ids))
        spikes = self.sorting_analyzer.sorting.to_spike_vector()

        spike_mask = np.isin(spikes["unit_index"], unit_inds)
        new_spike_locations = self.data["spike_locations"][spike_mask]
        return dict(spike_locations=new_spike_locations)

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):

        if keep_mask is None:
            new_spike_locations = self.data["spike_locations"].copy()
        else:
            new_spike_locations = self.data["spike_locations"][keep_mask]

        ### In theory here, we should recompute the locations since the peak positions
        ### in a merged could be different. Should be discussed
        return dict(spike_locations=new_spike_locations)

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
            ms_before=self.params["ms_before"],
            ms_after=self.params["ms_after"],
            **self.params["method_kwargs"],
        )
        return nodes

    def _run(self, verbose=False, **job_kwargs):
        job_kwargs = fix_job_kwargs(job_kwargs)
        nodes = self.get_pipeline_nodes()
        spike_locations = run_node_pipeline(
            self.sorting_analyzer.recording,
            nodes,
            job_kwargs=job_kwargs,
            job_name="spike_locations",
            gather_mode="memory",
            verbose=verbose,
        )
        self.data["spike_locations"] = spike_locations

    def _get_data(self, outputs="numpy"):
        all_spike_locations = self.data["spike_locations"]
        if outputs == "numpy":
            return all_spike_locations
        elif outputs == "by_unit":
            unit_ids = self.sorting_analyzer.unit_ids
            spike_vector = self.sorting_analyzer.sorting.to_spike_vector(concatenated=False)
            spike_indices = spike_vector_to_indices(spike_vector, unit_ids, absolute_index=True)
            spike_locations_by_units = {}
            for segment_index in range(self.sorting_analyzer.sorting.get_num_segments()):
                spike_locations_by_units[segment_index] = {}
                for unit_id in unit_ids:
                    inds = spike_indices[segment_index][unit_id]
                    spike_locations_by_units[segment_index][unit_id] = all_spike_locations[inds]
            return spike_locations_by_units
        else:
            raise ValueError(f"Wrong .get_data(outputs={outputs})")


ComputeSpikeLocations.__doc__.format(_shared_job_kwargs_doc)

register_result_extension(ComputeSpikeLocations)
compute_spike_locations = ComputeSpikeLocations.function_factory()
