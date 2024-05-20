import numpy as np



class Motion:
    """
    Motion of the tissue relative the probe.

    Parameters
    ----------

    displacement: numpy array 2d or list of
        Motion estimate in um.
        List is the number of segment.
        For each semgent : 
            * shape (temporal bins, spatial bins)
            * motion.shape[0] = temporal_bins.shape[0]
            * motion.shape[1] = 1 (rigid) or spatial_bins.shape[1] (non rigid)
    temporal_bins_s: numpy.array 1d or list of
        temporal bins (bin center)
    spatial_bins_um: numpy.array 1d
        Windows center.
        spatial_bins_um.shape[0] == displacement.shape[1]
        If rigid then spatial_bins_um.shape[0] == 1

    """
    def __init__(self, displacement, temporal_bins_s, spatial_bins_um, direction="y"):
        if isinstance(displacement, np.ndarray):
            self.displacement = [displacement]
            assert isinstance(temporal_bins_s, np.ndarray)
            self.temporal_bins_s = [temporal_bins_s]
        else:
            assert isinstance(displacement, (list, tuple))
            self.displacement = displacement
            self.temporal_bins_s = temporal_bins_s

        assert isinstance(spatial_bins_um, np.ndarray)
        self.spatial_bins_um = spatial_bins_um

        self.num_segments = len(self.displacement)
        self.interpolator = None
        
        self.direction = direction
        self.dim = ["x", "y", "z"].index(direction)

    def make_interpolators(self):
        from scipy.interpolate import RegularGridInterpolator
        self.interpolator = [
            RegularGridInterpolator((self.temporal_bins_s[j], self.spatial_bins_um), self.displacement[j])
            for j in range(self.num_segments)
        ]
        self.temporal_bounds = [(t[0], t[-1]) for t in self.temporal_bins_s]
        self.spatial_bounds = (self.spatial_bins_um.min(), self.spatial_bins_um.max())
    
    def get_displacement_at_time_and_depth(self, times_s, locations_um, segment_index=None):
        """


        Parameters
        ----------
        times_s: np.array


        locations_um: np.array
        
        segment_index: 
        
        """
        if self.interpolator is None:
            self.make_interpolators()

        if segment_index is None:
            if self.num_segments == 1:
                segment_index = 0
            else:
                raise ValueError("Several segment need segment_index=")
        
        times_s = np.asarray(times_s)
        locations_um = np.asarray(times_s)

        if locations_um.ndim == 1:
            locations_um = locations_um
        else:
            locations_um = locations_um[:, self.dim]
        times_s = np.clip(times_s, *self.temporal_bounds[segment_index])
        locations_um = np.clip(locations_um, *self.spatial_bounds)
        points = np.stack([times_s, locations_um,], axis=1)

        return self.interpolator[segment_index](points)

    def to_dict(self):
        return dict(
            displacement=self.displacement,
            temporal_bins_s=self.temporal_bins_s,
            spatial_bins_um=self.spatial_bins_um,
        )
    
    def save(self):
        # TODO
        pass

    @classmethod
    def load(cls):
        # TODO
        pass

    def __eq__(self, other):

        for segment_index in range(self.num_segments):
            if not np.allclose(self.displacement[segment_index], other.displacement[segment_index]):
                return False
            if not np.allclose(self.temporal_bins_s[segment_index], other.temporal_bins_s[segment_index]):
                return False
        
        if not np.allclose(self.spatial_bins_um, other.spatial_bins_um):
            return False
        
        return True
