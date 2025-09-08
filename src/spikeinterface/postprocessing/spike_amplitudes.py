from __future__ import annotations

import numpy as np

from spikeinterface.core.job_tools import fix_job_kwargs

from spikeinterface.core.template_tools import get_template_extremum_channel, get_template_extremum_channel_peak_shift

from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension
from spikeinterface.core.node_pipeline import SpikeRetriever, PipelineNode, run_node_pipeline, find_parent_of_type
from spikeinterface.core.sorting_tools import spike_vector_to_indices


class ComputeSpikeAmplitudes(AnalyzerExtension):
    """
    AnalyzerExtension
    Computes the spike amplitudes.

    Needs "templates" to be computed first.
    Computes spike amplitudes from the template's peak channel for every spike.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object
    peak_sign : "neg" | "pos" | "both", default: "neg"
        Sign of the template to compute extremum channel used to retrieve spike amplitudes.

    Returns
    -------
    spike_amplitudes: np.array
        All amplitudes for all spikes and all units are concatenated (along time, like in spike vector)

    """

    extension_name = "spike_amplitudes"
    depend_on = ["templates"]
    need_recording = True
    use_nodepipeline = True
    nodepipeline_variables = ["amplitudes"]
    need_job_kwargs = True

    def __init__(self, sorting_analyzer):
        AnalyzerExtension.__init__(self, sorting_analyzer)

        self._all_spikes = None

    def _set_params(self, peak_sign="neg"):
        params = dict(peak_sign=peak_sign)
        return params

    def _select_extension_data(self, unit_ids):
        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_analyzer.unit_ids, unit_ids))

        spikes = self.sorting_analyzer.sorting.to_spike_vector()
        keep_spike_mask = np.isin(spikes["unit_index"], keep_unit_indices)

        new_data = dict()
        new_data["amplitudes"] = self.data["amplitudes"][keep_spike_mask]

        return new_data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):
        new_data = dict()

        if keep_mask is None:
            new_data["amplitudes"] = self.data["amplitudes"].copy()
        else:
            new_data["amplitudes"] = self.data["amplitudes"][keep_mask]

        return new_data

    def _split_extension_data(self, split_units, new_unit_ids, new_sorting_analyzer, verbose=False, **job_kwargs):
        # splitting only changes random spikes assignments
        return self.data.copy()

    def _get_pipeline_nodes(self):

        recording = self.sorting_analyzer.recording
        sorting = self.sorting_analyzer.sorting

        peak_sign = self.params["peak_sign"]
        return_in_uV = self.sorting_analyzer.return_in_uV

        extremum_channels_indices = get_template_extremum_channel(
            self.sorting_analyzer, peak_sign=peak_sign, outputs="index"
        )
        peak_shifts = get_template_extremum_channel_peak_shift(self.sorting_analyzer, peak_sign=peak_sign)

        spike_retriever_node = SpikeRetriever(
            sorting, recording, channel_from_template=True, extremum_channel_inds=extremum_channels_indices
        )
        spike_amplitudes_node = SpikeAmplitudeNode(
            recording,
            parents=[spike_retriever_node],
            return_output=True,
            peak_shifts=peak_shifts,
            return_in_uV=return_in_uV,
        )
        nodes = [spike_retriever_node, spike_amplitudes_node]
        return nodes

    def _run(self, verbose=False, **job_kwargs):
        job_kwargs = fix_job_kwargs(job_kwargs)
        nodes = self.get_pipeline_nodes()
        amps = run_node_pipeline(
            self.sorting_analyzer.recording,
            nodes,
            job_kwargs=job_kwargs,
            job_name="spike_amplitudes",
            gather_mode="memory",
            verbose=False,
        )
        self.data["amplitudes"] = amps

    def _get_data(self, outputs="numpy", concatenated=False):
        all_amplitudes = self.data["amplitudes"]
        if outputs == "numpy":
            return all_amplitudes
        elif outputs == "by_unit":
            unit_ids = self.sorting_analyzer.unit_ids
            spike_vector = self.sorting_analyzer.sorting.to_spike_vector(concatenated=False)
            spike_indices = spike_vector_to_indices(spike_vector, unit_ids, absolute_index=True)
            amplitudes_by_units = {}
            for segment_index in range(self.sorting_analyzer.sorting.get_num_segments()):
                amplitudes_by_units[segment_index] = {}
                for unit_id in unit_ids:
                    inds = spike_indices[segment_index][unit_id]
                    amplitudes_by_units[segment_index][unit_id] = all_amplitudes[inds]

            if concatenated:
                amplitudes_by_units_concatenated = {
                    unit_id: np.concatenate(
                        [amps_in_segment[unit_id] for amps_in_segment in amplitudes_by_units.values()]
                    )
                    for unit_id in unit_ids
                }
                return amplitudes_by_units_concatenated

            return amplitudes_by_units
        else:
            raise ValueError(f"Wrong .get_data(outputs={outputs}); possibilities are `numpy` or `by_unit`")


register_result_extension(ComputeSpikeAmplitudes)

compute_spike_amplitudes = ComputeSpikeAmplitudes.function_factory()


class SpikeAmplitudeNode(PipelineNode):
    def __init__(
        self,
        recording,
        parents=None,
        return_output=True,
        peak_shifts=None,
        return_in_uV=True,
    ):
        PipelineNode.__init__(self, recording, parents=parents, return_output=return_output)
        self.return_in_uV = return_in_uV
        if return_in_uV and recording.has_scaleable_traces():
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
            return_in_uV=return_in_uV,
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
            amplitudes = amplitudes.astype("float32", copy=True)
            amplitudes *= self._gains[chan_inds]
            amplitudes += self._offsets[chan_inds]

        return amplitudes

    def get_trace_margin(self):
        return self._margin
