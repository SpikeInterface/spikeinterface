from __future__ import annotations

import numpy as np
import warnings

from spikeinterface.core.job_tools import fix_job_kwargs

from spikeinterface.core.template_tools import get_template_extremum_channel, get_template_extremum_channel_peak_shift

from spikeinterface.core.sortingresult import register_result_extension, ResultExtension
from spikeinterface.core.node_pipeline import SpikeRetriever, PipelineNode, run_node_pipeline, find_parent_of_type

class ComputeSpikeAmplitudes(ResultExtension):
    """
    ResultExtension
    Computes the spike amplitudes.

    Need "templates" or "fast_templates" to be computed first.
    Localize spikes in 2D or 3D with several methods given the template.

    Parameters
    ----------
    sorting_result: SortingResult
        A SortingResult object
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

    Returns
    -------
    spike_locations: np.array
        All locations for all spikes and all units are concatenated

    """
    extension_name = "spike_amplitudes"
    depend_on = ["fast_templates|templates", ]
    need_recording = True
    use_nodepipeline = True
    nodepipeline_variables = ["amplitudes"]
    need_job_kwargs = True

    def __init__(self, sorting_result):
        ResultExtension.__init__(self, sorting_result)

        self._all_spikes = None

    def _set_params(self, peak_sign="neg", return_scaled=True):
        params = dict(peak_sign=str(peak_sign), return_scaled=bool(return_scaled))
        return params

    def _select_extension_data(self, unit_ids):
        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_result.unit_ids, unit_ids))

        spikes = self.sorting_result.sorting.to_spike_vector()
        keep_spike_mask = np.isin(spikes["unit_index"], keep_unit_indices)

        new_data = dict()
        new_data["amplitudes"] = self.data["amplitudes"][keep_spike_mask]

        return new_data


    def _get_pipeline_nodes(self):

        recording = self.sorting_result.recording
        sorting = self.sorting_result.sorting

        peak_sign = self.params["peak_sign"]
        return_scaled = self.params["return_scaled"]

        extremum_channels_indices = get_template_extremum_channel(self.sorting_result, peak_sign=peak_sign, outputs="index")
        peak_shifts = get_template_extremum_channel_peak_shift(self.sorting_result, peak_sign=peak_sign)

        if return_scaled:
            # check if has scaled values:
            if not recording.has_scaled_traces():
                warnings.warn("Recording doesn't have scaled traces! Setting 'return_scaled' to False")
                return_scaled = False

        spike_retriever_node = SpikeRetriever(
            recording, sorting, channel_from_template=True, extremum_channel_inds=extremum_channels_indices
        )
        spike_amplitudes_node = SpikeAmplitudeNode(
            recording,
            parents=[spike_retriever_node],
            return_output=True,
            peak_shifts=peak_shifts,
            return_scaled=return_scaled,
        )
        nodes = [spike_retriever_node, spike_amplitudes_node]
        return nodes

    def _run(self, **job_kwargs):
        # TODO later gather to disk when format="binary_folder"
        job_kwargs = fix_job_kwargs(job_kwargs)
        nodes = self.get_pipeline_nodes()
        amps = run_node_pipeline(
            self.sorting_result.recording, nodes, job_kwargs=job_kwargs, job_name="spike_amplitudes", gather_mode="memory"
        )
        self.data["amplitudes"] = amps

    def _get_data(self):
        return self.data["amplitudes"]

register_result_extension(ComputeSpikeAmplitudes)

compute_spike_amplitudes = ComputeSpikeAmplitudes.function_factory()




class SpikeAmplitudeNode(PipelineNode):
    def __init__(
        self,
        recording,
        parents=None,
        return_output=True,
        peak_shifts=None,
        return_scaled=True,
    ):
        PipelineNode.__init__(self, recording, parents=parents, return_output=return_output)
        self.return_scaled = return_scaled
        if return_scaled and recording.has_scaled():
            self._dtype = np.float32
            self._gains = recording.get_channel_gains()
            self._offsets = recording.get_channel_gains()
        else:
            self._dtype = recording.get_dtype()
            self._gains = None
            self._offsets = None
        spike_retriever = find_parent_of_type(parents, SpikeRetriever)
        assert isinstance(
            spike_retriever, SpikeRetriever
        ), "SpikeAmplitudeNode needs a single SpikeRetriever as a parent"
        # put peak_shifts in vector way
        self._peak_shifts = np.array(list(peak_shifts.values()), dtype="int64")
        self._margin = np.max(np.abs(self._peak_shifts))
        self._kwargs.update(
            peak_shifts=peak_shifts,
            return_scaled=return_scaled,
        )

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, peaks):
        sample_indices = peaks["sample_index"].copy()
        unit_index = peaks["unit_index"]
        chan_inds = peaks["channel_index"]

        # apply shifts per spike
        sample_indices += self._peak_shifts[unit_index]

        # and get amplitudes
        amplitudes = traces[sample_indices, chan_inds]

        # and scale
        if self._gains is not None:
            traces = traces.astype("float32") * self._gains + self._offsets
            amplitudes = amplitudes.astype('float32', copy=True)
            amplitudes *= self._gains[chan_inds]
            amplitudes += self._offsets[chan_inds]

        return amplitudes

    def get_trace_margin(self):
        return self._margin
