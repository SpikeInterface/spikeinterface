import numpy as np
from spike_psvae.deconvolve import MatchPursuitObjectiveUpsample as Objective
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


class MyObjective(Objective):
    """Wrapper for MatchPursuitObjectiveUpsample"""

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
        (self.up_factor,
         self.unit_up_factor,
         self.up_up_map) = self.upsample_templates_mp(self.temps, self.params.upsample, self.n_unit)

        self.vis_chan = self.spatially_mask_templates(self.temps, self.vis_su_threshold)
        self.unit_overlap = self.template_overlaps(self.vis_chan, self.up_factor)

        # Index of the original templates prior to
        # upsampling them.
        self.orig_n_unit = self.n_unit
        self.n_unit = self.orig_n_unit * self.up_factor

        # Computing SVD for each template.
        self.compress_templates()

        # Compute pairwise convolution of filters
        self.pairwise_filter_conv()

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















