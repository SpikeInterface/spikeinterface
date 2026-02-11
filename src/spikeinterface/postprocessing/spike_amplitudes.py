from __future__ import annotations

import numpy as np

from spikeinterface.core.sortinganalyzer import register_result_extension
from spikeinterface.core.analyzer_extension_core import BaseSpikeVectorExtension
from spikeinterface.core.template_tools import get_template_extremum_channel, get_template_extremum_channel_peak_shift
from spikeinterface.core.node_pipeline import SpikeRetriever, PipelineNode, find_parent_of_type


class ComputeSpikeAmplitudes(BaseSpikeVectorExtension):
    """
    Computes the spike amplitudes.

    Needs "templates" to be computed first.
    Computes spike amplitudes from the template's peak channel for every spike.

    Parameters
    ----------
    peak_sign : "neg" | "pos" | "both", default: "neg"
        Sign of the template to compute extremum channel used to retrieve spike amplitudes.
    """

    extension_name = "spike_amplitudes"
    depend_on = ["templates"]
    nodepipeline_variables = ["amplitudes"]

    def _set_params(self, peak_sign="neg"):
        return super()._set_params(peak_sign=peak_sign)

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
