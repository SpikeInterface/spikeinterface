import numpy as np

from spikeinterface.core import ChannelSparsity, get_chunk_with_margin
from spikeinterface.core.job_tools import ChunkRecordingExecutor, _shared_job_kwargs_doc, ensure_n_jobs, fix_job_kwargs

from spikeinterface.core.template_tools import get_template_extremum_channel, get_template_extremum_channel_peak_shift
from spikeinterface.core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension


class AmplitudeScalingsCalculator(BaseWaveformExtractorExtension):
    """
    Computes amplitude scalings from WaveformExtractor.
    """

    extension_name = "amplitude_scalings"

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

        extremum_channel_inds = get_template_extremum_channel(self.waveform_extractor, outputs="index")
        self.spikes = self.waveform_extractor.sorting.to_spike_vector(
            extremum_channel_inds=extremum_channel_inds, use_cache=False
        )

    def _set_params(self, sparsity, max_dense_channels, ms_before, ms_after):
        params = dict(sparsity=sparsity, max_dense_channels=max_dense_channels, ms_before=ms_before, ms_after=ms_after)
        return params

    def _select_extension_data(self, unit_ids):
        old_unit_ids = self.waveform_extractor.sorting.unit_ids
        unit_inds = np.flatnonzero(np.in1d(old_unit_ids, unit_ids))

        spike_mask = np.in1d(self.spikes["unit_index"], unit_inds)
        new_amplitude_scalings = self._extension_data["amplitude_scalings"][spike_mask]
        return dict(amplitude_scalings=new_amplitude_scalings)

    def _run(self, **job_kwargs):
        job_kwargs = fix_job_kwargs(job_kwargs)
        we = self.waveform_extractor
        recording = we.recording
        nbefore = we.nbefore
        nafter = we.nafter
        ms_before = self._params["ms_before"]
        ms_after = self._params["ms_after"]

        return_scaled = we._params["return_scaled"]
        unit_ids = we.unit_ids

        if ms_before is not None:
            assert (
                ms_before <= we._params["ms_before"]
            ), f"`ms_before` must be smaller than `ms_before` used in WaveformExractor: {we._params['ms_before']}"
        if ms_after is not None:
            assert (
                ms_after <= we._params["ms_after"]
            ), f"`ms_after` must be smaller than `ms_after` used in WaveformExractor: {we._params['ms_after']}"

        cut_out_before = int(ms_before / 1000 * we.sampling_frequency) if ms_before is not None else nbefore
        cut_out_after = int(ms_after / 1000 * we.sampling_frequency) if ms_after is not None else nafter

        if we.is_sparse():
            sparsity = we.sparsity
        elif self._params["sparsity"] is not None:
            sparsity = self._params["sparsity"]
        else:
            if self._params["max_dense_channels"] is not None:
                assert recording.get_num_channels() <= self._params["max_dense_channels"], ""
            sparsity = ChannelSparsity.create_dense(we)
        sparsity_inds = sparsity.unit_id_to_channel_indices
        unit_inds_to_channel_indices = {unit_ind: sparsity_inds[unit_id] for unit_ind, unit_id in enumerate(unit_ids)}
        all_templates = we.get_all_templates()

        # precompute segment slice
        segment_slices = []
        for segment_index in range(we.get_num_segments()):
            i0 = np.searchsorted(self.spikes["segment_index"], segment_index)
            i1 = np.searchsorted(self.spikes["segment_index"], segment_index + 1)
            segment_slices.append(slice(i0, i1))

        # and run
        func = _amplitude_scalings_chunk
        init_func = _init_worker_amplitude_scalings
        n_jobs = ensure_n_jobs(recording, job_kwargs.get("n_jobs", None))
        job_kwargs["n_jobs"] = n_jobs
        init_args = (
            recording,
            self.spikes,
            all_templates,
            segment_slices,
            unit_inds_to_channel_indices,
            nbefore,
            nafter,
            cut_out_before,
            cut_out_after,
            return_scaled,
        )
        processor = ChunkRecordingExecutor(
            recording,
            func,
            init_func,
            init_args,
            handle_returns=True,
            job_name="extract amplitude scalings",
            **job_kwargs,
        )
        out = processor.run()
        (amp_scalings,) = zip(*out)
        amp_scalings = np.concatenate(amp_scalings)

        self._extension_data[f"amplitude_scalings"] = amp_scalings

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

        if outputs == "concatenated":
            return self._extension_data[f"amplitude_scalings"]
        elif outputs == "by_unit":
            amplitudes_by_unit = []
            for segment_index in range(we.get_num_segments()):
                amplitudes_by_unit.append({})
                segment_mask = self.spikes["segment_index"] == segment_index
                spikes_segment = self.spikes[segment_mask]
                amp_scalings_segment = self._extension_data[f"amplitude_scalings"][segment_mask]
                for unit_index, unit_id in enumerate(sorting.unit_ids):
                    unit_mask = spikes_segment["unit_index"] == unit_index
                    amp_scalings = amp_scalings_segment[unit_mask]
                    amplitudes_by_unit[segment_index][unit_id] = amp_scalings
            return amplitudes_by_unit

    @staticmethod
    def get_extension_function():
        return compute_amplitude_scalings


WaveformExtractor.register_extension(AmplitudeScalingsCalculator)


def compute_amplitude_scalings(
    waveform_extractor,
    sparsity=None,
    max_dense_channels=16,
    ms_before=None,
    ms_after=None,
    load_if_exists=False,
    outputs="concatenated",
    **job_kwargs,
):
    """
    Computes the amplitude scalings from a WaveformExtractor.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor object
    sparsity: ChannelSparsity
        If waveforms are not sparse, sparsity is required if the number of channels is greater than
        `max_dense_channels`. If the waveform extractor is sparse, its sparsity is automatically used.
        By default None
    max_dense_channels: int, default: 16
        Maximum number of channels to allow running without sparsity. To compute amplitude scaling using
        dense waveforms, set this to None, sparsity to None, and pass dense waveforms as input.
    ms_before : float, optional
        The cut out to apply before the spike peak to extract local waveforms.
        If None, the WaveformExtractor ms_before is used, by default None
    ms_after : float, optional
        The cut out to apply after the spike peak to extract local waveforms.
        If None, the WaveformExtractor ms_after is used, by default None
    load_if_exists : bool, default: False
        Whether to load precomputed spike amplitudes, if they already exist.
    outputs: str
        How the output should be returned:
            - 'concatenated'
            - 'by_unit'
    {}

    Returns
    -------
    amplitude_scalings: np.array or list of dict
        The amplitude scalings.
            - If 'concatenated' all amplitudes for all spikes and all units are concatenated
            - If 'by_unit', amplitudes are returned as a list (for segments) of dictionaries (for units)
    """
    if load_if_exists and waveform_extractor.is_extension(AmplitudeScalingsCalculator.extension_name):
        sac = waveform_extractor.load_extension(AmplitudeScalingsCalculator.extension_name)
    else:
        sac = AmplitudeScalingsCalculator(waveform_extractor)
        sac.set_params(sparsity=sparsity, max_dense_channels=max_dense_channels, ms_before=ms_before, ms_after=ms_after)
        sac.run(**job_kwargs)

    amps = sac.get_data(outputs=outputs)
    return amps


compute_amplitude_scalings.__doc__.format(_shared_job_kwargs_doc)


def _init_worker_amplitude_scalings(
    recording,
    spikes,
    all_templates,
    segment_slices,
    unit_inds_to_channel_indices,
    nbefore,
    nafter,
    cut_out_before,
    cut_out_after,
    return_scaled,
):
    # create a local dict per worker
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["spikes"] = spikes
    worker_ctx["all_templates"] = all_templates
    worker_ctx["segment_slices"] = segment_slices
    worker_ctx["nbefore"] = nbefore
    worker_ctx["nafter"] = nafter
    worker_ctx["cut_out_before"] = cut_out_before
    worker_ctx["cut_out_after"] = cut_out_after
    worker_ctx["margin"] = max(nbefore, nafter)
    worker_ctx["return_scaled"] = return_scaled
    worker_ctx["unit_inds_to_channel_indices"] = unit_inds_to_channel_indices

    return worker_ctx


def _amplitude_scalings_chunk(segment_index, start_frame, end_frame, worker_ctx):
    from scipy.stats import linregress

    # recover variables of the worker
    spikes = worker_ctx["spikes"]
    recording = worker_ctx["recording"]
    all_templates = worker_ctx["all_templates"]
    segment_slices = worker_ctx["segment_slices"]
    unit_inds_to_channel_indices = worker_ctx["unit_inds_to_channel_indices"]
    nbefore = worker_ctx["nbefore"]
    cut_out_before = worker_ctx["cut_out_before"]
    cut_out_after = worker_ctx["cut_out_after"]
    margin = worker_ctx["margin"]
    return_scaled = worker_ctx["return_scaled"]

    spikes_in_segment = spikes[segment_slices[segment_index]]

    i0 = np.searchsorted(spikes_in_segment["sample_index"], start_frame)
    i1 = np.searchsorted(spikes_in_segment["sample_index"], end_frame)

    local_waveforms = []
    templates = []
    scalings = []

    if i0 != i1:
        local_spikes = spikes_in_segment[i0:i1]
        traces_with_margin, left, right = get_chunk_with_margin(
            recording._recording_segments[segment_index], start_frame, end_frame, channel_indices=None, margin=margin
        )

        # scale traces with margin to match scaling of templates
        if return_scaled and recording.has_scaled():
            gains = recording.get_property("gain_to_uV")
            offsets = recording.get_property("offset_to_uV")
            traces_with_margin = traces_with_margin.astype("float32") * gains + offsets

        # get all waveforms
        for spike in local_spikes:
            unit_index = spike["unit_index"]
            sample_index = spike["sample_index"]
            sparse_indices = unit_inds_to_channel_indices[unit_index]
            template = all_templates[unit_index][:, sparse_indices]
            template = template[nbefore - cut_out_before : nbefore + cut_out_after]
            sample_centered = sample_index - start_frame
            cut_out_start = left + sample_centered - cut_out_before
            cut_out_end = left + sample_centered + cut_out_after
            if sample_index - cut_out_before < 0:
                local_waveform = traces_with_margin[:cut_out_end, sparse_indices]
                template = template[cut_out_before - sample_index :]
            elif sample_index + cut_out_after > end_frame + right:
                local_waveform = traces_with_margin[cut_out_start:, sparse_indices]
                template = template[: -(sample_index + cut_out_after - end_frame)]
            else:
                local_waveform = traces_with_margin[cut_out_start:cut_out_end, sparse_indices]
            assert template.shape == local_waveform.shape
            local_waveforms.append(local_waveform)
            templates.append(template)
            linregress_res = linregress(template.flatten(), local_waveform.flatten())
            scalings.append(linregress_res[0])
    scalings = np.array(scalings)

    return (scalings,)
