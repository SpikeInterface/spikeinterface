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
    template_index_to_unit_id: np.ndarray = None
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
        self.temps = templates.transpose(1, 2, 0)
        self.n_time, self.n_chan, self.n_unit = self.temps.shape

        # handle grouped templates, as in the superresolution case
        # TODO: refactor grouped_index mapping
        self.grouped_temps = False
        if self.params.template_index_to_unit_id is not None:
            self.grouped_temps = True
            assert self.params.template_index_to_unit_id.shape == (self.n_unit,)
            group_index = [
                np.flatnonzero(self.params.template_index_to_unit_id == u)
                for u in self.params.template_index_to_unit_id
            ]
            self.max_group_size = max(map(len, group_index))

            # like a channel index, sort of
            # this is a n_templates x group_size array that maps each
            # template index to the set of other template indices that
            # are part of its group. so that the array is not ragged,
            # we pad rows with -1s when their group is smaller than the
            # largest group.
            self.group_index = np.full((self.n_unit, self.max_group_size), -1)
            for j, row in enumerate(group_index):
                self.group_index[j, : len(row)] = row

                # variance parameter for the amplitude scaling prior
        assert self.lambd is None or self.lambd >= 0, "lambd must be a non-negative scalar"
        self.no_amplitude_scaling = self.lambd is None or self.lambd == 0
        self.scale_min = 1 / (1 + self.params.allowed_scale)
        self.scale_max = 1 + self.params.allowed_scale

        if self.verbose:
            print("expected shape of templates loaded (n_times, n_chan, n_units):", self.temps.shape)
            print(f"Instantiating MatchPursuitObjectiveUpsample on {self.params.t_end - self.params.t_start}",
                  f"seconds long recording with threshold {self.threshold}")

        self.start_sample = self.params.t_start * self.params.sampling_rate
        self.end_sample = self.params.t_end * self.params.sampling_rate

        # Upsample and downsample time shifted versions
        # Dynamic Upsampling Setup; function for upsampling based on PTP
        # Cat: TODO find better ptp-> upsample function
        upsampling_factors = self.upsample_templates_mp(self.temps, self.params.upsample, self.n_unit)
        self.up_factor, self.unit_up_factor, self.up_up_map = upsampling_factors

        self.vis_chan = self.spatially_mask_templates(self.temps, self.vis_su_threshold)
        self.unit_overlap = self.template_overlaps(self.vis_chan, self.up_factor)

        # Index of the original templates prior to
        # upsampling them.
        self.orig_n_unit = self.n_unit
        self.n_unit = self.orig_n_unit * self.up_factor

        # Computing SVD for each template.
        svd_matrices = self.compress_templates(self.temps, self.approx_rank, self.up_factor, self.n_time)
        self.temporal, self.singular, self.spatial, self.temporal_up = svd_matrices

        # Compute pairwise convolution of filters
        self.pairwise_conv = self.pairwise_filter_conv(self.multi_processing, self.up_up_map, self.temporal,
                                                       self.temporal_up, self.singular, self.spatial, self.n_time,
                                                       self.unit_overlap, self.up_factor, self.vis_chan,
                                                       self.approx_rank)

        # compute squared norm of templates
        self.norm = np.zeros(self.orig_n_unit, dtype=np.float32)
        for i in range(self.orig_n_unit):
            self.norm[i] = np.sum(
                np.square(self.temps[:, self.vis_chan[:, i], i])
            )

        # Initialize outputs
        self.dec_spike_train = np.zeros((0, 2), dtype=np.int32)
        self.dec_scalings = np.zeros((0,), dtype=np.float32)
        self.dist_metric = np.array([])

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


    @classmethod
    def upsample_templates_mp(cls, temps, max_upsample, n_unit):
        assert max_upsample >= 1, "upsample must be a positive integer"
        if max_upsample == 1: # Trivial Case
            up_factor = max_upsample
            unit_up_factor = np.ones(n_unit, dtype=int)
            up_up_map = np.arange(n_unit * up_factor)
            return up_factor, unit_up_factor, up_up_map

        # Compute appropriate upsample factor for each template
        template_range = np.max( np.ptp(temps, axis=0), axis=0 )
        unit_up_factor = np.power(4, np.floor( np.log2(template_range) )) # Why this computation?
        up_factor_unitmax = int(np.max(unit_up_factor))
        up_factor = np.clip(up_factor_unitmax, 1, max_upsample)
        unit_up_factor = np.clip(unit_up_factor.astype(np.int32), 1, max_upsample)
        up_up_map = np.zeros(n_unit*up_factor, dtype=np.int32)

        for i in range(n_unit):
            start_idx, stop_idx = i * up_factor, (i+1) * up_factor
            skip = up_factor // unit_up_factor[i]
            map_idx = start_idx + np.arange(0, up_factor, skip).repeat(skip)
            up_up_map[start_idx:stop_idx] = map_idx

        return up_factor, unit_up_factor, up_up_map


    @classmethod
    def template_overlaps(cls, vis_chan, up_factor):
        vis = vis_chan.T
        unit_overlap = np.sum(np.logical_and(vis[:, np.newaxis, :], vis[np.newaxis, :, :]), axis=2)
        unit_overlap = unit_overlap > 0
        unit_overlap = np.repeat(unit_overlap, up_factor, axis=0)
        return unit_overlap


    @classmethod
    def spatially_mask_templates(cls, templates, visibility_threshold):
        visible_channels = np.ptp(templates, axis=0) > visibility_threshold
        invisible_channels = np.logical_not(visible_channels)
        templates[:, invisible_channels] = 0.0
        return visible_channels


    @classmethod
    def compress_templates(cls, templates, approx_rank, up_factor, n_time):
        templates_unit_time_channel = np.transpose(templates, [2, 0, 1]) # TODO: Define template order
        temporal, singular, spatial = np.linalg.svd(templates_unit_time_channel)

        # Keep only the strongest components
        temporal = temporal[:, :, :approx_rank]
        singular = singular[:, :approx_rank]
        spatial = spatial[:, :approx_rank, :]

        # Upsample the temporal components of the SVD -- i.e. upsample the reconstruction
        if up_factor == 1: # Trivial Case
            temporal = np.flip(temporal, axis=1)
            temporal_up = temporal.copy()
            return temporal, singular, spatial, temporal_up

        num_samples = n_time * up_factor
        temporal_up = signal.resample(temporal, num_samples, axis=1)

        original_idx = np.arange(0, num_samples, up_factor) # indices of original data
        shift_idx = np.arange(up_factor)[:, np.newaxis] # shift for each super-res template
        shifted_idx = original_idx + shift_idx # array of all shifted template indices

        temporal_up = np.reshape(temporal_up[:, shifted_idx, :], [-1, n_time, approx_rank])
        temporal_up = temporal_up.astype(np.float32, casting='safe') # TODO: Redundant?

        temporal = np.flip(temporal, axis=1)
        temporal_up = np.flip(temporal_up, axis=1)
        return temporal, singular, spatial, temporal_up


    @classmethod
    def pairwise_filter_conv(cls, multi_processing, up_up_map, temporal, temporal_up, singular, spatial, n_time,
                             unit_overlap, up_factor, vis_chan, approx_rank):
        if multi_processing:
            raise NotImplementedError # TODO: Fold in spikeinterface multi-processing if necessary
        units = np.unique(up_up_map)
        pairwise_conv = cls.conv_filter(units, temporal, temporal_up, singular, spatial, n_time,
                                        unit_overlap, up_factor, vis_chan, approx_rank)
        return pairwise_conv


    @classmethod
    def conv_filter(cls, unit_array, temporal, temporal_up, singular, spatial,
                    n_time, unit_overlap, up_factor, vis_chan, approx_rank):
        conv_res_len = (n_time * 2) - 1  # TODO: convolution length as a function
        pairwise_conv_array = []
        for unit in unit_array:
            n_overlap = np.sum(unit_overlap[unit, :])
            orig_unit = unit // up_factor
            pairwise_conv = np.zeros([n_overlap, conv_res_len], dtype=np.float32)

            # Reconstruct unit template from SVD Matrices
            temporal_up_scaled = temporal_up[unit] * singular[orig_unit][np.newaxis, :]
            template_reconstructed = np.matmul(temporal_up_scaled, spatial[orig_unit, :, :])
            template_reconstructed = np.flipud(template_reconstructed)

            # TODO : Make order consistent with compute_objective (units and then rank or rank and then units?)
            units_are_overlapping = unit_overlap[unit, :]
            overlapping_units = np.where(units_are_overlapping)[0]
            for j, unit2 in enumerate(overlapping_units):
                temporal_overlapped = temporal[unit2]
                singular_overlapped = singular[unit2]
                spatial_overlapped = spatial[unit2]
                visible_overlapped_channels = vis_chan[:, unit2]
                visible_template = template_reconstructed[:, visible_overlapped_channels]
                spatial_filters = spatial_overlapped[:approx_rank, visible_overlapped_channels].T
                spatially_filtered_template = np.matmul(visible_template, spatial_filters)
                scaled_filtered_template = spatially_filtered_template * singular_overlapped
                for i in range(approx_rank):
                    pairwise_conv[j, :] += np.convolve(scaled_filtered_template[:, i], temporal_overlapped[:, i], 'full')
            pairwise_conv_array.append(pairwise_conv)
        return pairwise_conv_array















