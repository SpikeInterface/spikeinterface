import numpy as np
import warnings

from spikeinterface.core.job_tools import ChunkRecordingExecutor, _shared_job_kwargs_doc, ensure_n_jobs, fix_job_kwargs
from spikeinterface.core.template_tools import get_template_extremum_channel, get_template_extremum_channel_peak_shift
from spikeinterface.core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension
from spikeinterface.core.node_pipeline import SpikeRetriever, PipelineNode, run_node_pipeline, find_parent_of_type


class SpikeAmplitudesCalculator(BaseWaveformExtractorExtension):
    """
    Computes spike amplitudes from WaveformExtractor.
    """

    extension_name = "spike_amplitudes"
    pipeline_compatible = True

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

        self._all_spikes = None

    def _set_params(self, peak_sign="neg", return_scaled=True):
        params = dict(peak_sign=str(peak_sign), return_scaled=bool(return_scaled))
        return params

    def _select_extension_data(self, unit_ids):
        # load filter and save amplitude files
        sorting = self.waveform_extractor.sorting
        spikes = sorting.to_spike_vector(concatenated=True)
        (keep_unit_indices,) = np.nonzero(np.isin(sorting.unit_ids, unit_ids))

        new_extension_data = dict()
        amps = self._extension_data["spike_amplitudes"]
        filtered_idxs = np.isin(spikes["unit_index"], keep_unit_indices)
        new_extension_data["spike_amplitudes"] = amps[filtered_idxs]
        return new_extension_data

    def _run(self, **job_kwargs):
        job_kwargs = fix_job_kwargs(job_kwargs)
        # build pipeline nodes
        nodes = self.get_pipeline_nodes()
        # and run
        recording = self.waveform_extractor.recording
        amps = run_node_pipeline(
            recording, nodes, job_kwargs=job_kwargs, job_name="spike amplitudes", gather_mode="memory"
        )
        self._extension_data[f"spike_amplitudes"] = amps

    def get_data(self, outputs="concatenated"):
        """
        Get computed spike amplitudes.

        Parameters
        ----------
        outputs : str, optional
            'concatenated' or 'by_unit', by default 'concatenated'

        Returns
        -------
        spike_amplitudes : np.array or dict
            The spike amplitudes as an array (outputs='concatenated') or
            as a dict with units as key and spike amplitudes as values.
        """
        we = self.waveform_extractor
        sorting = we.sorting

        if "spike_amplitudes" in self._extension_data:
            legacy_format = False
        else:
            legacy_format = True

        if not legacy_format:
            amplitudes = self._extension_data["spike_amplitudes"]
            if outputs == "concatenated":
                return amplitudes
            elif outputs == "by_unit":
                all_spikes = sorting.to_spike_vector(concatenated=True)
                amplitudes_by_unit = []
                for segment_index in range(we.get_num_segments()):
                    amplitudes_by_unit.append({})
                    segment_mask = all_spikes["segment_index"] == segment_index
                    spikes_in_segment = all_spikes[segment_mask]
                    amps_in_segment = amplitudes[segment_mask]
                    for unit_index, unit_id in enumerate(sorting.unit_ids):
                        spike_labels = spikes_in_segment["unit_index"]
                        unit_mask = spike_labels == unit_index
                        amps = amps_in_segment[unit_mask]
                        amplitudes_by_unit[segment_index][unit_id] = amps
                return amplitudes_by_unit
        else:
            if outputs == "concatenated":
                amplitudes = []
                for segment_index in range(we.get_num_segments()):
                    amplitudes.append(self._extension_data[f"amplitude_segment_{segment_index}"])
                return np.concatenate(amplitudes)
            elif outputs == "by_unit":
                all_spikes = sorting.to_spike_vector(concatenated=False)
                amplitudes_by_unit = []
                for segment_index in range(we.get_num_segments()):
                    amplitudes_by_unit.append({})
                    for unit_index, unit_id in enumerate(sorting.unit_ids):
                        spike_labels = all_spikes[segment_index]["unit_index"]
                        mask = spike_labels == unit_index
                        amps = self._extension_data[f"amplitude_segment_{segment_index}"][mask]
                        amplitudes_by_unit[segment_index][unit_id] = amps
                return amplitudes_by_unit

    @staticmethod
    def get_extension_function():
        return compute_spike_amplitudes

    def get_pipeline_nodes(self):
        we = self.waveform_extractor
        recording = we.recording
        sorting = we.sorting

        peak_sign = self._params["peak_sign"]
        return_scaled = self._params["return_scaled"]

        extremum_channels_indices = get_template_extremum_channel(we, peak_sign=peak_sign, outputs="index")
        peak_shifts = get_template_extremum_channel_peak_shift(we, peak_sign=peak_sign)

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
            peak_shifts=peak_shifts,
            extremum_channels_indices=extremum_channels_indices,
            return_scaled=return_scaled,
        )
        nodes = [spike_retriever_node, spike_amplitudes_node]
        return nodes


WaveformExtractor.register_extension(SpikeAmplitudesCalculator)


def compute_spike_amplitudes(
    waveform_extractor, load_if_exists=False, peak_sign="neg", return_scaled=True, outputs="concatenated", **job_kwargs
):
    """
    Computes the spike amplitudes from a WaveformExtractor.

    1. The waveform extractor is used to determine the max channel per unit.
    2. Then a "peak_shift" is estimated because for some sorters the spike index is not always at the
       peak.
    3. Amplitudes are extracted in chunks (parallel or not)

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor object
    load_if_exists : bool, default: False
        Whether to load precomputed spike amplitudes, if they already exist.
    peak_sign: str
        The sign to compute maximum channel:
            - 'neg'
            - 'pos'
            - 'both'
    return_scaled: bool
        If True and recording has gain_to_uV/offset_to_uV properties, amplitudes are converted to uV.
    outputs: str
        How the output should be returned:
            - 'concatenated'
            - 'by_unit'
    {}

    Returns
    -------
    amplitudes: np.array or list of dict
        The spike amplitudes.
            - If 'concatenated' all amplitudes for all spikes and all units are concatenated
            - If 'by_unit', amplitudes are returned as a list (for segments) of dictionaries (for units)
    """
    if load_if_exists and waveform_extractor.is_extension(SpikeAmplitudesCalculator.extension_name):
        sac = waveform_extractor.load_extension(SpikeAmplitudesCalculator.extension_name)
    else:
        sac = SpikeAmplitudesCalculator(waveform_extractor)
        sac.set_params(peak_sign=peak_sign, return_scaled=return_scaled)
        sac.run(**job_kwargs)

    amps = sac.get_data(outputs=outputs)
    return amps


compute_spike_amplitudes.__doc__.format(_shared_job_kwargs_doc)


class SpikeAmplitudeNode(PipelineNode):
    def __init__(
        self,
        recording,
        parents=None,
        return_output=True,
        peak_shifts=None,
        extremum_channels_indices=None,
        return_scaled=None,
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
        # put extremum_channels_index and peak_shifts in vector way
        self._extremum_channels_indices = np.array(list(extremum_channels_indices.values()), dtype="int64")
        self._peak_shifts = np.array(list(peak_shifts.values()), dtype="int64")
        self._margin = np.max(np.abs(self._peak_shifts))
        self._kwargs.update(
            peak_shifts=peak_shifts,
            extremum_channels_indices_array=extremum_channels_indices,
            return_scaled=return_scaled,
        )

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, peaks):
        sample_indices = peaks["sample_index"].copy()
        labels = peaks["unit_index"]

        # apply shifts per spike
        sample_indices += self._peak_shifts[labels]

        # get channels per spike
        chan_inds = self._extremum_channels_indices[labels]

        # # prevent border accident due to shift
        # sample_indices[sample_indices < 0] = 0
        # sample_indices[sample_indices >= traces.shape[0]] = traces.shape[0] - 1

        if self._gains is not None:
            traces = traces.astype("float32") * self._gains + self._offsets

        # and get amplitudes
        amplitudes = traces[sample_indices, chan_inds]

        return amplitudes

    def get_trace_margin(self):
        return self._margin
