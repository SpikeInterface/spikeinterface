import numpy as np
from scipy import signal
from dataclasses import dataclass

@dataclass
class ObjectiveParameters:
    lambd: float = 0
    allowed_scale: float = np.inf
    save_residual: bool = False
    t_start: int = 0
    t_end: int = None
    n_sec_chunk: int = 1
    sampling_rate: int = 30_000
    max_iter: int = 1_000
    upsample: int = 8
    threshold: float = 30
    conv_approx_rank: int = 5
    n_processors: int = 1
    multi_processing: bool = False
    vis_su: float = 1
    verbose: bool = False
    template_ids2unit_ids: np.ndarray = None
    refractory_period_frames: int = 10


class MyObjective:
    """Refactored Version of MatchPursuitObjectiveUpsample from Spike-PSVAE"""

    def __init__(self, templates, **params):
        # Unpack params
        self.params = ObjectiveParameters(**params)
        self.verbose = self.params.verbose
        self.lambd = self.params.lambd
        self.max_iter = self.params.max_iter
        self.n_processors = self.params.n_processors
        self.multi_processing = self.params.multi_processing
        self.threshold = self.params.threshold
        self.approx_rank = self.params.conv_approx_rank
        self.vis_su_threshold = self.params.vis_su

        # templates --> self.temps
        assert templates.dtype == np.float32, "templates must have dtype np.float32"
        self.templates = templates
        self.n_templates, self.n_time, self.n_chan = self.templates.shape

        # handle grouped templates, as in the superresolution case
        self.grouped_temps = False
        self.n_units = self.n_templates
        if self.params.template_ids2unit_ids is not None:
            self.grouped_temps = True
            self.unit_ids = np.unique(self.params.template_ids2unit_ids)
            self.n_units = len(self.unit_ids)
            self.template_ids = np.arange(self.n_templates)
            self.template_ids2unit_ids = self.params.template_ids2unit_ids
            assert self.params.template_ids2unit_ids.shape == (self.n_templates,)
            self.unit_ids2template_ids = []
            for unit_id in self.unit_ids:
                template_ids_of_unit = set(self.template_ids[self.template_ids2unit_ids == unit_id])
                self.unit_ids2template_ids.append(template_ids_of_unit)

        # variance parameter for the amplitude scaling prior
        assert self.lambd is None or self.lambd >= 0, "lambd must be a non-negative scalar"
        self.no_amplitude_scaling = self.lambd is None or self.lambd == 0
        self.scale_min = 1 / (1 + self.params.allowed_scale)
        self.scale_max = 1 + self.params.allowed_scale

        if self.verbose:
            print("expected shape of templates loaded (n_templates, n_time, n_chan):", self.templates.shape)
            print(f"Instantiating MatchPursuitObjectiveUpsample on {self.params.t_end - self.params.t_start}",
                  f"seconds long recording with threshold {self.threshold}")

        self.start_sample = self.params.t_start * self.params.sampling_rate
        self.end_sample = self.params.t_end * self.params.sampling_rate

        # Upsample and downsample time shifted versions
        self.up_factor = self.params.upsample
        self.n_jittered = self.n_templates * self.up_factor
        self.jittered_ids = np.arange(self.n_jittered)

        self.vis_chan = self.spatially_mask_templates(self.templates, self.vis_su_threshold)
        self.unit_overlap = self.template_overlaps(self.vis_chan, self.up_factor)

        # Computing SVD for each template.
        svd_matrices = self.compress_templates(self.templates, self.approx_rank, self.up_factor, self.n_time)
        self.temporal, self.singular, self.spatial, self.temporal_jittered = svd_matrices

        # Compute pairwise convolution of filters
        self.pairwise_conv = self.pairwise_filter_conv(self.multi_processing, self.jittered_ids, self.temporal,
                                                       self.temporal_jittered, self.singular, self.spatial, self.n_time,
                                                       self.unit_overlap, self.up_factor, self.vis_chan,
                                                       self.approx_rank)

        # compute squared norm of templates
        self.norm = np.zeros(self.n_templates, dtype=np.float32)
        for i in range(self.n_templates):
            self.norm[i] = np.sum(
                np.square(self.templates[i, :, self.vis_chan[i, :]])
            )

        # Initialize outputs
        self.spike_train = np.zeros((0, 2), dtype=np.int32)
        self.scalings = np.zeros((0,), dtype=np.float32)
        self.distance_metric = np.array([])

        # Single time preperation for high resolution matches
        # matching indeces of peaks to indices of upsampled templates
        radius = (self.up_factor // 2) + (self.up_factor % 2)
        self.up_window = np.arange(-radius, radius + 1)[:, np.newaxis]
        self.up_window_len = len(self.up_window)

        # Indices of single time window the window around peak after upsampling
        self.zoom_index = radius * self.up_factor + np.arange(-radius, radius + 1)
        self.peak_to_template_idx = np.concatenate(
            (np.arange(radius, -1, -1), (self.up_factor - 1) - np.arange(radius))
        )
        self.peak_time_jitter = np.concatenate(
            ([0], np.array([0, 1]).repeat(radius))
        )

        # Refractory Period Setup.
        # DO NOT MAKE IT SMALLER THAN self.n_time - 1 !!!
        # (This is not actually the refractory condition we enforce.
        #  It is a radius to make sure the algorithm does not subtract
        #  two overlapping spikes at the same time, which would confuse it.)
        self.refrac_radius = self.n_time - 1

        # Account for upsampling window so that np.inf does not fall into the
        # window around peak for valid spikes.
        # (This is the actual refractory condition we enforce.)
        self.adjusted_refrac_radius = self.params.refractory_period_frames

    # TODO: Replace vis_chan, template_overlaps & spatially_mask_templates with spikeinterface sparsity representation
    @classmethod
    def template_overlaps(cls, vis_chan, up_factor):
        unit_overlap = np.sum(np.logical_and(vis_chan[:, np.newaxis, :], vis_chan[np.newaxis, :, :]), axis=2)
        unit_overlap = unit_overlap > 0
        unit_overlap = np.repeat(unit_overlap, up_factor, axis=0)
        return unit_overlap


    @classmethod
    def spatially_mask_templates(cls, templates, visibility_threshold):
        visible_channels = np.ptp(templates, axis=1) > visibility_threshold
        invisible_channels = np.logical_not(visible_channels)
        for i in range(templates.shape[0]):
            templates[i, :, invisible_channels[i, :]] = 0.0
        return visible_channels


    @classmethod
    def compress_templates(cls, templates, approx_rank, up_factor, n_time):
        temporal, singular, spatial = np.linalg.svd(templates)

        # Keep only the strongest components
        temporal = temporal[:, :, :approx_rank]
        singular = singular[:, :approx_rank]
        spatial = spatial[:, :approx_rank, :]

        # Upsample the temporal components of the SVD -- i.e. upsample the reconstruction
        if up_factor == 1: # Trivial Case
            temporal = np.flip(temporal, axis=1)
            temporal_jittered = temporal.copy()
            return temporal, singular, spatial, temporal_jittered

        num_samples = n_time * up_factor
        temporal_jittered = signal.resample(temporal, num_samples, axis=1)

        original_idx = np.arange(0, num_samples, up_factor) # indices of original data
        shift_idx = np.arange(up_factor)[:, np.newaxis] # shift for each super-res template
        shifted_idx = original_idx + shift_idx # array of all shifted template indices

        temporal_jittered = np.reshape(temporal_jittered[:, shifted_idx, :], [-1, n_time, approx_rank])
        temporal_jittered = temporal_jittered.astype(np.float32, casting='safe') # TODO: Redundant?

        temporal = np.flip(temporal, axis=1)
        temporal_jittered = np.flip(temporal_jittered, axis=1)
        return temporal, singular, spatial, temporal_jittered


    @classmethod
    def pairwise_filter_conv(cls, multi_processing, jittered_ids, temporal, temporal_jittered, singular, spatial, n_time,
                             unit_overlap, up_factor, vis_chan, approx_rank):
        if multi_processing:
            raise NotImplementedError # TODO: Fold in spikeinterface multi-processing if necessary
        pairwise_conv = cls.conv_filter(jittered_ids, temporal, temporal_jittered, singular, spatial, n_time,
                                        unit_overlap, up_factor, vis_chan, approx_rank)
        return pairwise_conv


    @classmethod
    def conv_filter(cls, jittered_ids, temporal, temporal_jittered, singular, spatial,
                    n_time, unit_overlap, up_factor, vis_chan, approx_rank):
        conv_res_len = (n_time * 2) - 1  # TODO: convolution length as a function
        pairwise_conv_array = []
        for jittered_id in jittered_ids:
            n_overlap = np.sum(unit_overlap[jittered_id, :])
            template_id = jittered_id // up_factor
            pairwise_conv = np.zeros([n_overlap, conv_res_len], dtype=np.float32)

            # Reconstruct unit template from SVD Matrices
            temporal_jittered_scaled = temporal_jittered[jittered_id] * singular[template_id][np.newaxis, :]
            template_reconstructed = np.matmul(temporal_jittered_scaled, spatial[template_id, :, :])
            template_reconstructed = np.flipud(template_reconstructed)

            # TODO : Make order consistent with compute_objective (units and then rank or rank and then units?)
            units_are_overlapping = unit_overlap[jittered_id, :]
            overlapping_units = np.where(units_are_overlapping)[0]
            for j, jittered_id2 in enumerate(overlapping_units):
                temporal_overlapped = temporal[jittered_id2]
                singular_overlapped = singular[jittered_id2]
                spatial_overlapped = spatial[jittered_id2]
                visible_overlapped_channels = vis_chan[jittered_id2, :]
                visible_template = template_reconstructed[:, visible_overlapped_channels]
                spatial_filters = spatial_overlapped[:approx_rank, visible_overlapped_channels].T
                spatially_filtered_template = np.matmul(visible_template, spatial_filters)
                scaled_filtered_template = spatially_filtered_template * singular_overlapped
                for i in range(approx_rank):
                    pairwise_conv[j, :] += np.convolve(scaled_filtered_template[:, i], temporal_overlapped[:, i], 'full')
            pairwise_conv_array.append(pairwise_conv)
        return pairwise_conv_array


    # ------------------------------------------------------------------------------------------------------------------
    # DEPRECATED
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def upsample_templates_mp(cls, temps, max_upsample, n_unit):
        """This method is deprecated in favor of a static upsampling factor"""
        assert max_upsample >= 1, "upsample must be a positive integer"
        if max_upsample == 1: # Trivial Case
            up_factor = max_upsample
            unit_up_factor = np.ones(n_unit, dtype=int)
            jittered_ids = np.arange(n_unit * up_factor)
            return up_factor, unit_up_factor, jittered_ids

        # Compute appropriate upsample factor for each template
        template_range = np.max( np.ptp(temps, axis=0), axis=0 )
        unit_up_factor = np.power(4, np.floor( np.log2(template_range) )) # Why this computation?
        up_factor_unitmax = int(np.max(unit_up_factor))
        up_factor = np.clip(up_factor_unitmax, 1, max_upsample)
        unit_up_factor = np.clip(unit_up_factor.astype(np.int32), 1, max_upsample)
        jittered_ids = np.zeros(n_unit*up_factor, dtype=np.int32)

        for i in range(n_unit):
            start_idx, stop_idx = i * up_factor, (i+1) * up_factor
            skip = up_factor // unit_up_factor[i]
            map_idx = start_idx + np.arange(0, up_factor, skip).repeat(skip)
            jittered_ids[start_idx:stop_idx] = map_idx
        print(f"{jittered_ids = }")

        return up_factor, unit_up_factor, jittered_ids














