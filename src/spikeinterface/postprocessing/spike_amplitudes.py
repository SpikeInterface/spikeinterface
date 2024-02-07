from __future__ import annotations

import numpy as np
import shutil

from spikeinterface.core.job_tools import ChunkRecordingExecutor, _shared_job_kwargs_doc, ensure_n_jobs, fix_job_kwargs

from spikeinterface.core.template_tools import get_template_extremum_channel, get_template_extremum_channel_peak_shift

from spikeinterface.core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension


class SpikeAmplitudesCalculator(BaseWaveformExtractorExtension):
    """
    Computes spike amplitudes from WaveformExtractor.
    """

    extension_name = "spike_amplitudes"

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

        self._all_spikes = None

    def _set_params(self, peak_sign="neg", return_scaled=True):
        params = dict(peak_sign=str(peak_sign), return_scaled=bool(return_scaled))
        return params

    def _select_extension_data(self, unit_ids):
        # load filter and save amplitude files
        sorting = self.waveform_extractor.sorting
        spikes = sorting.to_spike_vector(concatenated=False)
        (keep_unit_indices,) = np.nonzero(np.isin(sorting.unit_ids, unit_ids))

        new_extension_data = dict()
        for seg_index in range(sorting.get_num_segments()):
            amp_data_name = f"amplitude_segment_{seg_index}"
            amps = self._extension_data[amp_data_name]
            filtered_idxs = np.isin(spikes[seg_index]["unit_index"], keep_unit_indices)
            new_extension_data[amp_data_name] = amps[filtered_idxs]
        return new_extension_data

    def _run(self, **job_kwargs):
        if not self.waveform_extractor.has_recording():
            self.waveform_extractor.delete_extension(SpikeAmplitudesCalculator.extension_name)
            raise ValueError("compute_spike_amplitudes() cannot run with a WaveformExtractor in recordless mode.")

        job_kwargs = fix_job_kwargs(job_kwargs)
        we = self.waveform_extractor
        recording = we.recording
        sorting = we.sorting

        all_spikes = sorting.to_spike_vector()
        self._all_spikes = all_spikes

        peak_sign = self._params["peak_sign"]
        return_scaled = self._params["return_scaled"]

        extremum_channels_index = get_template_extremum_channel(we, peak_sign=peak_sign, outputs="index")
        peak_shifts = get_template_extremum_channel_peak_shift(we, peak_sign=peak_sign)

        # put extremum_channels_index and peak_shifts in vector way
        extremum_channels_index = np.array(
            [extremum_channels_index[unit_id] for unit_id in sorting.unit_ids], dtype="int64"
        )
        peak_shifts = np.array([peak_shifts[unit_id] for unit_id in sorting.unit_ids], dtype="int64")

        if return_scaled:
            # check if has scaled values:
            if not recording.has_scaled_traces():
                print("Setting 'return_scaled' to False")
                return_scaled = False

        # and run
        func = _spike_amplitudes_chunk
        init_func = _init_worker_spike_amplitudes
        n_jobs = ensure_n_jobs(recording, job_kwargs.get("n_jobs", None))
        init_args = (recording, sorting.to_multiprocessing(n_jobs), extremum_channels_index, peak_shifts, return_scaled)
        processor = ChunkRecordingExecutor(
            recording, func, init_func, init_args, handle_returns=True, job_name="extract amplitudes", **job_kwargs
        )
        out = processor.run()
        amps, segments = zip(*out)
        amps = np.concatenate(amps)
        segments = np.concatenate(segments)

        for segment_index in range(recording.get_num_segments()):
            mask = segments == segment_index
            amps_seg = amps[mask]
            self._extension_data[f"amplitude_segment_{segment_index}"] = amps_seg

    def get_data(self, outputs="concatenated"):
        """
        Get computed spike amplitudes.

        Parameters
        ----------
        outputs : "concatenated" | "by_unit", default: "concatenated"
            The output format

        Returns
        -------
        spike_amplitudes : np.array or dict
            The spike amplitudes as an array (outputs="concatenated") or
            as a dict with units as key and spike amplitudes as values.
        """
        we = self.waveform_extractor
        sorting = we.sorting

        if outputs == "concatenated":
            amplitudes = []
            for segment_index in range(we.get_num_segments()):
                amplitudes.append(self._extension_data[f"amplitude_segment_{segment_index}"])
            return amplitudes
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
    peak_sign: "neg" | "pos" | "both", default: "neg
        The sign to compute maximum channel
    return_scaled: bool, deafult: True
        If True and recording has gain_to_uV/offset_to_uV properties, amplitudes are converted to uV.
    outputs: "concatenated" | "by_unit", default: "concatenated"
        How the output should be returned
    {}

    Returns
    -------
    amplitudes: np.array or list of dict
        The spike amplitudes.
            - If "concatenated" all amplitudes for all spikes and all units are concatenated
            - If "by_unit", amplitudes are returned as a list (for segments) of dictionaries (for units)
    """
    if load_if_exists and waveform_extractor.has_extension(SpikeAmplitudesCalculator.extension_name):
        sac = waveform_extractor.load_extension(SpikeAmplitudesCalculator.extension_name)
    else:
        sac = SpikeAmplitudesCalculator(waveform_extractor)
        sac.set_params(peak_sign=peak_sign, return_scaled=return_scaled)
        sac.run(**job_kwargs)

    amps = sac.get_data(outputs=outputs)
    return amps


compute_spike_amplitudes.__doc__.format(_shared_job_kwargs_doc)


def _init_worker_spike_amplitudes(recording, sorting, extremum_channels_index, peak_shifts, return_scaled):
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["sorting"] = sorting
    worker_ctx["return_scaled"] = return_scaled
    worker_ctx["peak_shifts"] = peak_shifts
    worker_ctx["min_shift"] = np.min(peak_shifts)
    worker_ctx["max_shifts"] = np.max(peak_shifts)

    worker_ctx["all_spikes"] = sorting.to_spike_vector(concatenated=False)
    worker_ctx["extremum_channels_index"] = extremum_channels_index

    return worker_ctx


def _spike_amplitudes_chunk(segment_index, start_frame, end_frame, worker_ctx):
    # recover variables of the worker
    all_spikes = worker_ctx["all_spikes"]
    recording = worker_ctx["recording"]
    return_scaled = worker_ctx["return_scaled"]
    peak_shifts = worker_ctx["peak_shifts"]

    seg_size = recording.get_num_samples(segment_index=segment_index)

    spike_times = all_spikes[segment_index]["sample_index"]
    spike_labels = all_spikes[segment_index]["unit_index"]

    d = np.diff(spike_times)
    assert np.all(d >= 0)

    i0, i1 = np.searchsorted(spike_times, [start_frame, end_frame])
    n_spikes = i1 - i0
    amplitudes = np.zeros(n_spikes, dtype=recording.get_dtype())

    if i0 != i1:
        # some spike in the chunk

        extremum_channels_index = worker_ctx["extremum_channels_index"]

        sample_inds = spike_times[i0:i1].copy()
        labels = spike_labels[i0:i1]

        # apply shifts per spike
        sample_inds += peak_shifts[labels]

        # get channels per spike
        chan_inds = extremum_channels_index[labels]

        # prevent border accident due to shift
        sample_inds[sample_inds < 0] = 0
        sample_inds[sample_inds >= seg_size] = seg_size - 1

        first = np.min(sample_inds)
        last = np.max(sample_inds)
        sample_inds -= first

        # load trace in memory
        traces = recording.get_traces(
            start_frame=first, end_frame=last + 1, segment_index=segment_index, return_scaled=return_scaled
        )

        # and get amplitudes
        amplitudes = traces[sample_inds, chan_inds]

    segments = np.zeros(amplitudes.size, dtype="int64") + segment_index

    return amplitudes, segments
