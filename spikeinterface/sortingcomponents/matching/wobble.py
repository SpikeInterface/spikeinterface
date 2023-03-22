import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

from .main import BaseTemplateMatchingEngine

@dataclass
class WobbleParameters:
    """Parameters for the WobbleMatch algorithm.

    Parameters
    ----------
    amplitude_variance : float
        Variance of the spike amplitudes for each template: amplitude scaling factor ~ N(1, amplitude_variance).
    max_iter : int
        Maximum # of iterations the algorithm will run before aborting.
    jitter_factor : int
        Number of upsampled jittered templates for each distinct provided template.
    threshold : float
        Minimum amplitude to classify a peak (relative maximum in the objective) as a spike (EAP). Units depend on the
        underlying units of voltage trace and templates.
    conv_approx_rank : int
        Rank of the compressed template matrices.
    visibility_thresold : float
        Minimum peak amplitude to determine channel sparsity for a given unit. Units depend on the underlying units of
        voltage trace and templates.
    verbose : bool
        If True, print out informative messages.
    template_ids2unit_ids : numpy.ndarray
        Maps from the index of provided templates to their corresponding units.
    refractory_period_frames : int
        Duration of refractory period in frames/samples.
    scale_min : float
        Minimum value for amplitude scaling of templates.
    scale_max : float
        Maximum value for ampltiude scaling of templates.
    scale_amplitudes : bool
        If True, scale amplitudes of templates to match spikes.

    Notes
    -----
    For consistency, 'peaks' refer to relative maxima in the convolution of the templates with the voltage trace
    (or residual) and 'spikes' refer to putative extracellular action potentials (EAPs). Peaks are considered spikes
     if their amplitude clears the threshold parameter.

    """
    amplitude_variance: float = 0
    max_iter: int = 1_000
    jitter_factor: int = 8
    threshold: float = 30 # TODO : Add units to threshold? (ex. thresold_uV) --> benchmark
    conv_approx_rank: int = 5
    visibility_threshold: float = 1
    verbose: bool = False
    template_ids2unit_ids: Optional[np.ndarray] = None
    refractory_period_frames: int = 10 # TODO : convert to refractory_period_ms --> benchmark
    scale_min : float = 0
    scale_max : float = np.inf
    scale_amplitudes: bool = False


@dataclass
class TemplateMetadata:
    """Handy metadata to describe size/shape/etc. of templates.

    Parameters
    ----------
    n_time : int
        Template duration in samples/frames.
    n_chan : int
        Number of channels.
    n_units : int
        Number of units.
    n_templates : int
        Number of templates.
    n_jittered : int
        Number of jittered templates.
    unit_ids : ndarray (n_units,)
        Indexes corresponding to each unit.
    template_ids : ndarray (n_templates,)
        Indexes corresponding to each template.
    jittered_ids : ndarray (n_jittered,)
        Indexes corresponding to each jittered template.
    template_ids2unit_ids : ndarray (n_templates,)
        Maps from the index of provided templates to their corresponding units.
    unit_ids2template_ids : list[set]
        Maps each unit index to the set of its corresponding templates.
    overlapping_spike_buffer : int
        Buffer to prevent adjacent spikes from being subtracted from the objective at the same time.
    peak_window : ndarray
        Window of indexes around the peak in each spike waveform used to perform upsampling.
    peak_window_len : int
        Length of peak_window.
    jitter_window : ndarray
        Window of indexes around around the maximum of the upsampled peak waveform.
    jitter2template_shift : ndarray
        Maps the optimal jitter from the jitter_window to the optimal template shift.
    jitter2spike_time_shift : ndarray
        Maps the optimal jitter from teh jitter_window to the optimal shift in spike time index.

    Notes
    -----
    For consistency, a 'unit' refers to a putative neuron which may have one or more 'templates' of its spike waveform.
    Each 'template' may have many upsampled 'jittered_templates' depending on the 'jitter_factor'.
    """
    n_time : int
    n_chan : int
    n_units : int
    n_templates : int
    n_jittered : int
    unit_ids : np.ndarray
    template_ids : np.ndarray
    jittered_ids : np.ndarray
    template_ids2unit_ids : np.ndarray
    unit_ids2template_ids : List[set]
    overlapping_spike_buffer : int
    peak_window : np.ndarray
    peak_window_len : int
    jitter_window : np.ndarray
    jitter2template_shift : np.ndarray
    jitter2spike_time_shift : np.ndarray


@dataclass
class Sparsity:
    """Variables that describe channel sparsity.

    Parameters
    ----------
    vis_chan : ndarray (n_units, n_chan)
        vis_chan[unit, channel] is True if the unit's template has sufficient amplitude on that channel.
    unit_overlap : ndarray (n_jittered, n_jittered)
        unit_overlap[i, j] is True if there exists at least one channel on which both template i and template j are
        visible.
    """
    vis_chan : np.ndarray
    unit_overlap : np.ndarray


@dataclass
class Objective:
    """Variables needed to compute the objective and the objective itself (obj)

    Parameters
    ----------
    compressed_templates : (ndarray, ndarray, ndarray, ndarray)
        Templates compressed by singular value decomposition into temporal, singular, spatial, and upsampled_temporal
        components.
    pairwise_convolution : list[ndarray]
        For each jittered template, pairwise_convolution of that template with each other overlapping template.
    norm : ndarray (n_templates,)
        Magnitude of each template for normalization.
    obj_len : int
        Length of the convolution of templates with the voltage trace.
    obj : ndarray (n_templates, obj_len)
        Template matching objective for each template.
    obj_normalized ndarray (n_templates, obj_len)
        Template matching objective normalized by the magnitude of each template.
    """
    compressed_templates : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    pairwise_convolution : Optional[List[np.ndarray]] = None
    norm : Optional[np.ndarray] = None
    obj_len : Optional[int] = None
    obj: Optional[np.ndarray] = None
    obj_normalized: Optional[np.ndarray] = None


class WobbleMatch(BaseTemplateMatchingEngine):
    """Template matching method from the Paninski lab.

    Templates are jittered or 'wobbled' in time and amplitude to capture variability in spike amplitude and
    super-resolution jitter in spike timing.

    Notes
    -----
    For consistency,
    - a 'unit' refers to a putative neuron which may have one or more 'templates' of its spike waveform
    - Each 'template' may have many upsampled 'jittered_templates' depending on the 'jitter_factor'
    - 'peaks' refer to relative maxima in the convolution of the templates with the voltage trace
    - 'spikes' refer to putative extracellular action potentials (EAPs)
    - 'peaks' are considered spikes if their amplitude clears the threshold parameter
    """
    default_params = {
        'waveform_extractor' : None,
    }
    spike_dtype = [('sample_ind', 'int64'), ('channel_ind', 'int64'), ('cluster_ind', 'int64'),
                   ('amplitude', 'float64'), ('segment_ind', 'int64')]

    @classmethod
    def initialize_and_check_kwargs(cls, recording, kwargs):
        """Initialize the objective and precompute various useful objects.

        Parameters
        ----------
        recording : RecordingExtractor
            The recording extractor object.
        kwargs : dict
            Keyword arguments for matching method.

        Returns
        -------
        d : dict
            Updated Keyword arguments.
        """
        d = cls.default_params.copy()
        objective_kwargs = kwargs['objective_kwargs']
        templates = objective_kwargs.pop('templates')
        templates = templates.astype(np.float32, casting='safe')

        # Aggregate useful parameters/variables for handy access in downstream functions
        params = cls.aggregate_parameters(objective_kwargs)
        template_meta = cls.aggregate_template_metadata(params, templates)
        sparsity = cls.aggregate_sparsity(params, templates) # TODO: replace with spikeinterface sparsity

        # Perform initial computations necessary for computing the objective
        compressed_templates = cls.compress_templates(templates, params, template_meta)
        pairwise_convolution = cls.conv_filter(compressed_templates, params, template_meta, sparsity)
        norm = cls.compute_template_norm(sparsity, template_meta, templates)
        objective = Objective(compressed_templates=compressed_templates,
                              pairwise_convolution=pairwise_convolution,
                              norm=norm)

        cls.pack_kwargs(kwargs, params, template_meta, sparsity, objective)
        d.update(kwargs)
        return d


    @staticmethod
    def aggregate_parameters(objective_kwargs):
        """Aggregate parameters from objective_kwargs into dataclass object.

        Parameters
        ----------
        objective_kwargs : dict
            Keyword arguments for WobbleMatch algorithm.

        Returns
        -------
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.
        """
        params = WobbleParameters(**objective_kwargs)
        assert(params.amplitude_variance >= 0, "amplitude_variance must be a non-negative scalar")
        params.scale_amplitudes = params.amplitude_variance > 0
        return params


    @staticmethod
    def aggregate_template_metadata(params, templates):
        """Aggregate template metadata into dataclass object.

        Parameters
        ----------
        params : WobbleParameters
            Parameters for WobbleMatch algorithm.
        templates : ndarray (n_templates, n_time, n_chan)
            Spike template waveforms.

        Returns
        -------
        template_meta : TemplateMetadata
            Dataclass object for aggregating template metadata together.
        """
        n_templates, n_time, n_chan = templates.shape
        # handle units with many templates, as in the super-resolution case
        if params.template_ids2unit_ids is None: # Trivial grouping of templates = units
            template_ids2unit_ids = np.arange(n_templates)
        else:
            assert params.template_ids2unit_ids.shape == (templates.shape[0],), \
                "template_ids2unit_ids must have shape (n_templates,)"
            template_ids2unit_ids = params.template_ids2unit_ids
        unit_ids = np.unique(template_ids2unit_ids)
        n_units = len(unit_ids)
        template_ids = np.arange(n_templates)
        unit_ids2template_ids = []
        for unit_id in unit_ids:
            template_ids_of_unit = set(template_ids[template_ids2unit_ids == unit_id])
            unit_ids2template_ids.append(template_ids_of_unit)
        n_jittered = n_templates * params.jitter_factor
        jittered_ids = np.arange(n_jittered)
        overlapping_spike_buffer = n_time - 1  # makes sure two overlapping spikes aren't subtracted at the same time

        # TODO: Benchmark peak_radius with alternative expressions
        peak_radius = (params.jitter_factor // 2) + (params.jitter_factor % 2) # Empirical --> needs to be benchmarked
        peak_window = np.arange(-peak_radius, peak_radius + 1)
        peak_window_len = len(peak_window)
        jitter_window = peak_radius * params.jitter_factor + peak_window
        jitter2template_shift = np.concatenate(
            (np.arange(peak_radius, -1, -1), (params.jitter_factor - 1) - np.arange(peak_radius))
        )
        jitter2spike_time_shift = np.concatenate( ([0], np.array([0, 1]).repeat(peak_radius)) )
        template_meta = TemplateMetadata(
            n_time=n_time, n_chan=n_chan, n_units=n_units, n_templates=n_templates, n_jittered=n_jittered,
            unit_ids=unit_ids, template_ids=template_ids, jittered_ids=jittered_ids,
            template_ids2unit_ids=template_ids2unit_ids, unit_ids2template_ids=unit_ids2template_ids,
            overlapping_spike_buffer=overlapping_spike_buffer, peak_window=peak_window, peak_window_len=peak_window_len,
            jitter_window=jitter_window, jitter2template_shift=jitter2template_shift,
            jitter2spike_time_shift=jitter2spike_time_shift
        )
        return template_meta


    @classmethod
    def aggregate_sparsity(cls, params, templates):
        """Aggregate variables relevant to sparse representation of templates.

        Parameters
        ----------
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.
        templates : ndarray (n_templates, n_time, n_chan)
            Spike template waveforms.

        Returns
        -------
        sparsity : Sparsity
            Dataclass object for aggregating channel sparsity variables together.
        """
        vis_chan = cls.spatially_mask_templates(templates, params.visibility_threshold)
        unit_overlap = cls.template_overlaps(vis_chan, params.jitter_factor)
        sparsity = Sparsity(vis_chan=vis_chan, unit_overlap=unit_overlap)
        return sparsity


    @staticmethod
    def compute_template_norm(sparsity, template_meta, templates):
        """Computes norm of each template.

        Parameters
        ----------
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.
        template_meta : TemplateMetadata
            Dataclass object for aggregating template metadata together.
        templates : ndarray (n_templates, n_time, n_chan)
            Spike template waveforms.

        Returns
        -------
        norm : ndarray (n_templates,)
        Magnitude of each template for normalization.
        """
        norm = np.zeros(template_meta.n_templates, dtype=np.float32)
        for i in range(template_meta.n_templates):
            norm[i] = np.sum(
                np.square(templates[i, :, sparsity.vis_chan[i, :]])
            )
        return norm


    @staticmethod
    def pack_kwargs(kwargs, params, template_meta, sparsity, objective):
        """Packs initial data into kwargs. Operates in-place on kwargs.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments for matching method.
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.
        template_meta : TemplateMetadata
            Dataclass object for aggregating template metadata together.
        sparsity : Sparsity
            Dataclass object for aggregating channel sparsity variables together.
        objective : Objective
            Dataclass object for aggregating variables related to the objective together.
        """
        kwargs['params'] = params
        kwargs['template_meta'] = template_meta
        kwargs['sparsity'] = sparsity
        kwargs['objective'] = objective


    @staticmethod
    def spatially_mask_templates(templates, visibility_threshold):
        """Determine which channels are 'visible' for each template, based on peak-to-peak amplitude, and
        set the template to 0 for 'invisible' channels.

        Parameters
        ----------
        templates : ndarray (n_templates, n_time, n_chan)
            Spike template waveforms.
        visibility_thresold : float
            Minimum peak amplitude to determine channel sparsity for a given unit. Units depend on the underlying units
            of voltage trace and templates.

        Returns
        -------
        vis_chan : ndarray (n_units, n_chan)
            vis_chan[unit, channel] is True if the unit's template has sufficient amplitude on that channel.
        """
        vis_chan = np.ptp(templates, axis=1) > visibility_threshold
        invisible_channels = np.logical_not(vis_chan)
        for i in range(templates.shape[0]):
            templates[i, :, invisible_channels[i, :]] = 0.0
        return vis_chan

    # TODO: Replace vis_chan, template_overlaps & spatially_mask_templates with spikeinterface sparsity representation
    @staticmethod
    def template_overlaps(vis_chan, jitter_factor):
        """Finds overlapping templates based on peak-to-peak amplitude at different channels.

        Parameters
        ----------
        vis_chan : ndarray (n_units, n_chan)
            vis_chan[unit, channel] is True if the unit's template has sufficient amplitude on that channel.
        jitter_factor : int
            Number of upsampled jittered templates for each distinct provided template.

        Returns
        -------
        unit_overlap : ndarray (n_jittered, n_jittered)
            unit_overlap[i, j] is True if there exists at least one channel on which both template i and template j are
            visible.
        """
        unit_overlap = np.sum(np.logical_and(vis_chan[:, np.newaxis, :], vis_chan[np.newaxis, :, :]), axis=2)
        unit_overlap = unit_overlap > 0
        unit_overlap = np.repeat(unit_overlap, jitter_factor, axis=0)
        return unit_overlap



    @staticmethod
    def compress_templates(templates, params, template_meta):
        """Compress templates using singular value decomposition.

        Parameters
        ----------
        templates : ndarray (n_templates, n_time, n_chan)
            Spike template waveforms.
        params : WobbleParameters
            Parameters for WobbleMatch algorithm.
        template_meta : TemplateMetadata
            Dataclass object for aggregating template metadata together.

        Returns
        -------
        compressed_templates : (ndarray, ndarray, ndarray, ndarray)
            Templates compressed by singular value decomposition into temporal, singular, spatial, and
            upsampled_temporal components.
        """
        temporal, singular, spatial = np.linalg.svd(templates)

        # Keep only the strongest components
        temporal = temporal[:, :, :params.conv_approx_rank]
        singular = singular[:, :params.conv_approx_rank]
        spatial = spatial[:, :params.conv_approx_rank, :]

        # Upsample the temporal components of the SVD -- i.e. upsample the reconstruction
        if params.jitter_factor == 1:  # Trivial Case
            temporal = np.flip(temporal, axis=1)
            temporal_jittered = temporal.copy()
            return temporal, singular, spatial, temporal_jittered

        num_samples = template_meta.n_time * params.jitter_factor
        temporal_jittered = signal.resample(temporal, num_samples, axis=1)

        original_idx = np.arange(0, num_samples, params.jitter_factor)  # indices of original data
        shift_idx = np.arange(params.jitter_factor)[:, np.newaxis]  # shift for each super-res template
        shifted_idx = original_idx + shift_idx  # array of all shifted template indices

        shape_temporal_jittered = [-1, template_meta.n_time, params.conv_approx_rank]
        temporal_jittered = np.reshape(temporal_jittered[:, shifted_idx, :], shape_temporal_jittered)

        temporal = np.flip(temporal, axis=1)
        temporal_jittered = np.flip(temporal_jittered, axis=1)
        compressed_templates = temporal, singular, spatial, temporal_jittered
        return compressed_templates


    @staticmethod
    def conv_filter(compressed_templates, params, template_meta, sparsity):
        """Perform pairwise convolution on the compressed templates.

        Parameters
        ----------
        compressed_templates : (ndarray, ndarray, ndarray, ndarray)
            Templates compressed by singular value decomposition into temporal, singular, spatial, and
            upsampled_temporal components.
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.
        template_meta : TemplateMetadata
            Dataclass object for aggregating template metadata together.
        sparsity : Sparsity
            Dataclass object for aggregating channel sparsity variables together.

        Returns
        -------
        pairwise_convolution : list[ndarray]
            For each jittered template, pairwise_convolution of that template with each other overlapping template.
        """
        temporal, singular, spatial, temporal_jittered = compressed_templates
        conv_res_len = get_convolution_len(template_meta.n_time, template_meta.n_time)
        pairwise_convolution = []
        for jittered_id in template_meta.jittered_ids:
            n_overlap = np.sum(sparsity.unit_overlap[jittered_id, :])
            template_id = jittered_id // params.jitter_factor
            pconv = np.zeros([n_overlap, conv_res_len], dtype=np.float32)

            # Reconstruct unit template from SVD Matrices
            temporal_jittered_scaled = temporal_jittered[jittered_id] * singular[template_id][np.newaxis, :]
            template_reconstructed = np.matmul(temporal_jittered_scaled, spatial[template_id, :, :])
            template_reconstructed = np.flipud(template_reconstructed)

            units_are_overlapping = sparsity.unit_overlap[jittered_id, :]
            overlapping_units = np.where(units_are_overlapping)[0]
            for j, jittered_id2 in enumerate(overlapping_units):
                temporal_overlapped = temporal[jittered_id2]
                singular_overlapped = singular[jittered_id2]
                spatial_overlapped = spatial[jittered_id2]
                visible_overlapped_channels = sparsity.vis_chan[jittered_id2, :]
                visible_template = template_reconstructed[:, visible_overlapped_channels]
                spatial_filters = spatial_overlapped[:params.conv_approx_rank, visible_overlapped_channels].T
                spatially_filtered_template = np.matmul(visible_template, spatial_filters)
                scaled_filtered_template = spatially_filtered_template * singular_overlapped
                for i in range(params.conv_approx_rank):
                    pconv[j, :] += np.convolve(scaled_filtered_template[:, i], temporal_overlapped[:, i],
                                                       'full')
            pairwise_convolution.append(pconv)
        return pairwise_convolution

    @classmethod
    def serialize_method_kwargs(cls, kwargs):
        # This function does nothing without a waveform extractor -- candidate for refactor
        kwargs = dict(kwargs)
        return kwargs

    @classmethod
    def unserialize_in_worker(cls, kwargs):
        # This function does nothing without a waveform extractor -- candidate for refactor
        return kwargs

    @classmethod
    def get_margin(cls, recording, kwargs):
        """Get margin for chunking recording.

        Parameters
        ----------
        recording : RecordingExtractor
            The recording extractor object.
        kwargs : dict
            Keyword arguments for matching method.

        Returns
        -------
        margin : int
            Buffer in samples on each side of a chunk.
        """
        buffer_ms = 10
        # margin = int(buffer_ms*1e-3 * recording.sampling_frequency)
        margin = 300 # To ensure equivalence with spike-psvae version of the algorithm
        return margin

    @classmethod
    def main_function(cls, traces, method_kwargs):
        """Main function that performs template matching.

        Includes spike time correction to ensure spike times are at trough, and unit id correction to account for
        upsampling.

        Parameters
        ----------
        traces : ndarray (chunk_len + 2*margin, n_chan)
            Voltage traces for a chunk of the recording.
        method_kwargs : dict
            Keyword arguments for matching method.

        Returns
        -------
        spikes : ndarray (n_spikes,)
            Resulting spike train.
        """
        # Unpack method_kwargs
        nbefore, nafter = method_kwargs['nbefore'], method_kwargs['nafter']
        template_meta = method_kwargs['template_meta']
        params = method_kwargs['params']
        sparsity = method_kwargs['sparsity']
        objective = method_kwargs['objective']

        # Check traces
        traces = traces.astype(np.float32, casting='safe') # ensure traces are specified as np.float32
        objective.obj_len = get_convolution_len(traces.shape[0], template_meta.n_time)

        # run using run_array
        spike_train, scalings, distance_metric = cls.run_array(traces, template_meta, params, sparsity, objective)

        # extract spiketrain and perform adjustments
        spike_train[:, 0] += nbefore
        spike_train[:, 1] //= params.jitter_factor

        # TODO : Benchmark spike amplitudes
        # Find spike amplitudes / channels
        amplitudes, channel_inds = [], []
        for i, spike_idx in enumerate(spike_train[:, 0]):
            best_ch = np.argmax(np.abs(traces[spike_idx, :]))
            amp = np.abs(traces[spike_idx, best_ch])
            amplitudes.append(amp)
            channel_inds.append(best_ch)

        # assign result to spikes array
        spikes = np.zeros(spike_train.shape[0], dtype=cls.spike_dtype)
        spikes['sample_ind'] = spike_train[:, 0]
        spikes['cluster_ind'] = spike_train[:, 1]
        spikes['channel_ind'] = channel_inds
        spikes['amplitude'] = amplitudes

        return spikes


    @classmethod
    def run_array(cls, traces, template_meta, params, sparsity, objective):
        """Performs template matching using the WobbleMatch algorithm.

        Spike times are indexed at the beginning of the template rather than the trough of the spike.
        Spike ids refer to the jittered ids rather than the unit ids.

        Parameters
        ----------
        traces : ndarray (chunk_len + 2*margin, n_chan)
            Voltage traces for a chunk of the recording.
        template_meta : TemplateMetadata
            Dataclass object for aggregating template metadata together.
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.
        sparsity : Sparsity
            Dataclass object for aggregating channel sparsity variables together.
        objective : Objective
            Dataclass object for aggregating variables related to the objective together.

        Returns
        -------
        spike_train : ndarray (n_spikes, 2)
            Spike train from template matching.
        scalings : ndarray (n_spikes,)
            Amplitude scaling used for each spike.
        distance_metric : ndarray (n_spikes)
            A metric that describes how good of a 'fit' each spike is to its corresponding template
        """
        # Compute objective
        cls.compute_objective(traces, objective, params, template_meta)

        # Compute spike train
        spike_trains, scalings, distance_metrics = [], [], []
        for i in range(params.max_iter):
            # find peaks
            spike_train, scaling, distance_metric = cls.find_peaks(objective, params, template_meta)
            if len(spike_train) == 0:
                break

            # update spike_train, scaling, distance metrics with new values
            spike_trains.extend(list(spike_train))
            scalings.extend(list(scaling))
            distance_metrics.extend(list(distance_metric))

            # subtract newly detected spike train from traces (via the objective)
            cls.subtract_spike_train(spike_train, scaling, objective, params, template_meta, sparsity)

        spike_train = np.array(spike_trains)
        scalings = np.array(scalings)
        distance_metric = np.array(distance_metrics)
        if len(spike_train) == 0:
            spike_train = np.zeros((0, 2), dtype=np.int32)

        # order spike times
        idx = np.argsort(spike_train[:, 0])
        spike_train = spike_train[idx]
        scalings = scalings[idx]
        distance_metric = distance_metric[idx]

        return spike_train, scalings, distance_metric


    @classmethod
    def compute_objective(cls, traces, objective, params, template_meta):
        """Compute objective by convolving templates with voltage traces.

        Operates on objective in-place.

        Parameters
        ----------
        traces : ndarray (chunk_len + 2*margin, n_chan)
            Voltage traces for a chunk of the recording.
        objective : Objective
            Dataclass object for aggregating variables related to the objective together.
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.
        template_meta : TemplateMetadata
            Dataclass object for aggregating template metadata together.
        """
        temporal, singular, spatial, temporal_jittered = objective.compressed_templates
        conv_shape = (template_meta.n_templates, objective.obj_len)
        obj = np.zeros(conv_shape, dtype=np.float32)
        # TODO: vectorize this loop
        for rank in range(params.conv_approx_rank):
            spatial_filters = spatial[:, rank, :]
            temporal_filters = temporal[:, :, rank]
            spatially_filtered_data = np.matmul(spatial_filters, traces.T)
            scaled_filtered_data = spatially_filtered_data * singular[:, [rank]]
            # TODO: vectorize this loop
            for template_id in range(template_meta.n_templates):
                template_data = scaled_filtered_data[template_id, :]
                template_temporal_filter = temporal_filters[template_id]
                obj[template_id, :] += np.convolve(template_data,
                                                   template_temporal_filter,
                                                   mode='full')
        obj_normalized = 2 * obj - objective.norm[:, np.newaxis]
        objective.obj = obj
        objective.obj_normalized = obj_normalized


    # TODO: Replace this method with equivalent from spikeinterface
    @classmethod
    def find_peaks(cls, objective, params, template_meta):
        """Find new peaks in the objective and update spike train accordingly.

        Parameters
        ----------
        objective : Objective
            Dataclass object for aggregating variables related to the objective together.
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.
        template_meta : TemplateMetadata
            Dataclass object for aggregating template metadata together.

        Returns
        -------
        new_spike_train : ndarray (n_spikes, 2)
            Spike train from template matching with newly detected spikes added.
        scalings : ndarray (n_spikes,)
            Amplitude scaling used for each spike.
        distance_metric : ndarray (n_spikes)
            A metric that describes how good of a 'fit' each spike is to its corresponding template
        """
        # Get spike times (indices) using peaks in the objective
        obj_template_max = np.max(objective.obj_normalized, axis=0)
        spike_window = (template_meta.n_time - 1, objective.obj_normalized.shape[1] - template_meta.n_time)
        obj_windowed = obj_template_max[spike_window[0]:spike_window[1]]
        spike_time_indices = signal.argrelmax(obj_windowed, order=template_meta.n_time-1)[0]
        spike_time_indices += template_meta.n_time - 1
        obj_spikes = obj_template_max[spike_time_indices]
        spike_time_indices = spike_time_indices[obj_spikes > params.threshold]

        # Extract metrics using spike times (indices)
        distance_metric = obj_template_max[spike_time_indices]
        scalings = np.ones(len(spike_time_indices), dtype=objective.obj_normalized.dtype)

        # Find the best upsampled template
        spike_template_ids = np.argmax(objective.obj_normalized[:, spike_time_indices], axis=0)
        high_res_peaks = cls.high_res_peak(spike_time_indices, spike_template_ids, objective, params, template_meta)
        template_shift, time_shift, non_refractory_indices, scaling = high_res_peaks

        # Update unit_ids, spike_times, and scalings
        spike_jittered_ids = spike_template_ids * params.jitter_factor
        at_least_one_spike = bool(len(non_refractory_indices))
        if at_least_one_spike:
            spike_jittered_ids[non_refractory_indices] += template_shift
            spike_time_indices[non_refractory_indices] += time_shift
            scalings[non_refractory_indices] = scaling

        # Generate new spike train from spike times (indices)
        convolution_correction = -1*(template_meta.n_time - 1) # convolution indices --> raw_indices
        spike_time_indices += convolution_correction
        new_spike_train = np.array([spike_time_indices, spike_jittered_ids]).T

        return new_spike_train, scalings, distance_metric


    @classmethod
    def subtract_spike_train(cls, spike_train, scalings, objective, params, template_meta, sparsity):
        """Subtract spike train of templates from the objective directly.

        Operates in-place on the objective.

        Parameters
        ----------
        spike_train : ndarray (n_spikes, 2)
            Spike train from template matching.
        scalings : ndarray (n_spikes,)
            Amplitude scaling used for each spike.
        objective : Objective
            Dataclass object for aggregating variables related to the objective together.
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.
        template_meta : TemplateMetadata
            Dataclass object for aggregating template metadata together.
        sparsity : Sparsity
            Dataclass object for aggregating channel sparsity variables together.
        """
        present_jittered_ids = np.unique(spike_train[:, 1])
        convolution_resolution_len = get_convolution_len(template_meta.n_time, template_meta.n_time)
        for jittered_id in present_jittered_ids:
            id_mask = spike_train[:, 1] == jittered_id
            id_spiketrain = spike_train[id_mask, 0]
            id_scaling = scalings[id_mask]
            overlapping_templates = sparsity.unit_overlap[jittered_id]
            # Note: pairwise_conv only has overlapping template convolutions already
            pconv = objective.pairwise_convolution[jittered_id]
            # TODO: If optimizing for speed -- check this loop
            for spike_start_idx, spike_scaling in zip(id_spiketrain, id_scaling):
                spike_stop_idx = spike_start_idx + convolution_resolution_len
                objective.obj_normalized[overlapping_templates, spike_start_idx:spike_stop_idx] -= 2*pconv
                if params.scale_amplitudes:
                    pconv_scaled = pconv * spike_scaling
                    objective.obj[overlapping_templates, spike_start_idx:spike_stop_idx] -= pconv_scaled

            cls.enforce_refractory(spike_train, objective, params, template_meta)


    @classmethod
    def high_res_peak(cls, spike_time_indices, spike_unit_ids, objective, params, template_meta):
        """Determines optimal shifts when super-resolution, scaled templates are used.

        Parameters
        ----------
        spike_time_indices : ndarray (n_spikes,)
            Indices in the voltage traces corresponding to the time of each spike.
        spike_unit_ids : ndarray (n_spikes)
            Units corresponding to each spike.
        objective : Objective
            Dataclass object for aggregating variables related to the objective together.
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.
        template_meta : TemplateMetadata
            Dataclass object for aggregating template metadata together.

        Returns
        -------
        template_shift : ndarray (n_spikes,)
            Indices to shift each spike template_id to the correct jittered_id.
        time_shift : ndarray (n_spikes,)
            Indices to shift each spike time index to the adjusted time index.
        non_refractory_indices : ndarray
            Indices of the spike train that correspond to non-refractory spikes.
        scalings : ndarray (n_spikes,)
            Amplitude scaling used for each spike.
        """
        # Return identities if no high-resolution templates are necessary
        not_high_res = params.jitter_factor == 1 and not params.scale_amplitudes
        at_least_one_spike = bool(len(spike_time_indices))
        if not_high_res or not at_least_one_spike:
            upsampled_template_idx = np.zeros_like(spike_time_indices)
            time_shift = np.zeros_like(spike_time_indices)
            non_refractory_indices = range(len(spike_time_indices))
            scalings = np.ones_like(spike_time_indices)
            return upsampled_template_idx, time_shift, non_refractory_indices, scalings

        peak_indices = spike_time_indices + template_meta.peak_window[:, np.newaxis]
        obj_peaks = objective.obj_normalized[spike_unit_ids, peak_indices]

        # Omit refractory spikes
        peak_is_refractory = np.logical_or(np.isinf(obj_peaks[0, :]), np.isinf(obj_peaks[-1, :]))
        refractory_before_spike = np.arange(-template_meta.overlapping_spike_buffer, 1)[:, np.newaxis]
        refractory_indices = spike_time_indices[peak_is_refractory] + refractory_before_spike
        objective.obj_normalized[spike_unit_ids[peak_is_refractory], refractory_indices] = -1 * np.inf
        non_refractory_indices = np.flatnonzero(np.logical_not(peak_is_refractory))
        obj_peaks = obj_peaks[:, non_refractory_indices]
        if obj_peaks.shape[1] == 0: # no non-refractory peaks --> exit function
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Upsample and compute optimal template shift
        window_len_upsampled = template_meta.peak_window_len * params.jitter_factor
        if not params.scale_amplitudes:
            # Perform simple upsampling using scipy.signal.resample
            high_resolution_peaks = signal.resample(obj_peaks, window_len_upsampled, axis=0)
            jitter = np.argmax(high_resolution_peaks[template_meta.jitter_window, :], axis=0)
            scalings = np.ones(len(non_refractory_indices))
        else:
            # upsampled the convolution for the detected peaks only
            obj_peaks_high_res = objective.obj[spike_unit_ids, peak_indices]
            obj_peaks_high_res = obj_peaks_high_res[:, non_refractory_indices]
            high_resolution_conv = signal.resample(obj_peaks_high_res, window_len_upsampled, axis=0)

            # Find template norms for detected peaks only
            norm_peaks = objective.norm[spike_unit_ids[non_refractory_indices]]

            high_res_obj, scalings = cls.compute_scale_amplitudes(high_resolution_conv, norm_peaks, params)
            jitter = np.argmax(high_res_obj[template_meta.jitter_window, :], axis=0)
            scalings = scalings[jitter, np.arange(len(non_refractory_indices))]

        # Extract outputs from jitter
        template_shift = template_meta.jitter2template_shift[jitter]
        time_shift = template_meta.jitter2spike_time_shift[jitter]
        return template_shift, time_shift, non_refractory_indices, scalings

    @classmethod
    def compute_scale_amplitudes(cls, high_resolution_conv, norm_peaks, params):
        """Compute optimal amplitude scaling and the high-resolution objective resulting from scaled spikes.

        Without hard clipping, the objective can be obtained via
            obj = (conv + 1/amplitude_variance)^2 / (norm + 1/amplitude_variance) - 1/amplitude_variance
        But, squaring the variables can lead to overflow, so we clip the possible amplitude scaling:
            scaling = clip(b / a, scale_min, scale_max);
                where b = conv + 1/amplitude_variance,
                      a = norm + 1/amplitude_variance
        Then, we have the resulting modified formula for the objective:
            obj = 2 * b * scaling - (scaling^2 * a) - 1 / amplitude_variance

        Parameters
        ----------
        high_resolution_conv : ndarray
            Super-resolution upsampled convolution of the spike templates with the traces, but only for a small window
            in time around the peak of the spike.
        norm_peaks : ndarray (n_spikes,)
            Magnitude of the template corresponding to each spike in the spike train.
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.

        Returns
        -------
        high_res_obj : ndarray
            Super-resolution upsampled objective, but only for a small window in time around the peak of each spike.
        scalings : ndarray (n_spikes,)
            Amplitude scaling used for each spike.
        """
        b = high_resolution_conv + 1 / params.amplitude_variance
        a = norm_peaks[np.newaxis, :] + 1 / params.amplitude_variance
        scalings = np.clip(b / a, params.scale_min, params.scale_max)
        high_res_obj = (2 * scalings * b) - (np.square(scalings) * a) - (1 / params.amplitude_variance)
        return high_res_obj, scalings

    @classmethod
    def enforce_refractory(cls, spike_train, objective, params, template_meta):
        """Enforcing the refractory period for each unit by setting the objective to -infinity.

        Operates in-place on the objective.

        Parameters
        ----------
        spike_train : ndarray (n_spikes, 2)
            Spike train from template matching.
        objective : Objective
            Dataclass object for aggregating variables related to the objective together.
        params : WobbleParameters
            Dataclass object for aggregating the parameters together.
        template_meta : TemplateMetadata
            Dataclass object for aggregating template metadata together.
        """
        window = np.arange(-params.refractory_period_frames, params.refractory_period_frames+1)

        # Adjust cluster IDs so that they match original templates
        spike_times = spike_train[:, 0]
        spike_template_ids = spike_train[:, 1] // params.jitter_factor

        # We want to enforce refractory conditions on unit_ids rather than template_ids for units with many templates
        spike_unit_ids = spike_template_ids.copy()
        for template_id in set(spike_template_ids):
            unit_id = template_meta.template_ids2unit_ids[template_id] # unit_id corresponding to this template
            spike_unit_ids[spike_template_ids==template_id] = unit_id

        # Get the samples (time indices) that correspond to the waveform for each spike
        waveform_samples = get_convolution_len(spike_times[:, np.newaxis], template_meta.n_time) + window

        # Enforce refractory by setting objective to negative infinity in invalid regions
        objective.obj_normalized[spike_unit_ids[:, np.newaxis], waveform_samples[:, 1:-1]] = -1 * np.inf
        if params.scale_amplitudes: # template_convolution is only used with amplitude scaling
            objective.obj[spike_unit_ids[:, np.newaxis], waveform_samples[:, 1:-1]] = -1 * np.inf


def get_convolution_len(x, y):
    """Returns the length of the convolution of vectors with lengths x and y."""
    return x + y - 1
