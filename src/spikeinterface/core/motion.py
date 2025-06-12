import json
from pathlib import Path

import numpy as np
import spikeinterface
from spikeinterface.core.core_tools import check_json


class Motion:
    """
    Motion of the tissue relative the probe.

    Parameters
    ----------
    displacement : numpy array 2d or list of
        Motion estimate in um.
        List is the number of segment.
        For each semgent :

            * shape (temporal bins, spatial bins)
            * motion.shape[0] = temporal_bins.shape[0]
            * motion.shape[1] = 1 (rigid) or spatial_bins.shape[1] (non rigid)
    temporal_bins_s : numpy.array 1d or list of
        temporal bins (bin center)
    spatial_bins_um : numpy.array 1d
        Windows center.
        spatial_bins_um.shape[0] == displacement.shape[1]
        If rigid then spatial_bins_um.shape[0] == 1
    direction : str, default: 'y'
        Direction of the motion.
    interpolation_method : str
        How to determine the displacement between bin centers? See the docs
        for scipy.interpolate.RegularGridInterpolator for options.
    """

    def __init__(self, displacement, temporal_bins_s, spatial_bins_um, direction="y", interpolation_method="linear"):
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
        self.interpolators = None
        self.interpolation_method = interpolation_method

        self.direction = direction
        self.dim = ["x", "y", "z"].index(direction)
        self.check_properties()
        self.temporal_bin_edges_s = [ensure_time_bin_edges(tbins) for tbins in self.temporal_bins_s]

    def check_properties(self):
        assert all(d.ndim == 2 for d in self.displacement)
        assert all(t.ndim == 1 for t in self.temporal_bins_s)
        assert all(self.spatial_bins_um.shape == (d.shape[1],) for d in self.displacement)

    def __repr__(self):
        nbins = self.spatial_bins_um.shape[0]
        if nbins == 1:
            rigid_txt = "rigid"
        else:
            rigid_txt = f"non-rigid - {nbins} spatial bins"

        interval_s = self.temporal_bins_s[0][1] - self.temporal_bins_s[0][0]
        txt = f"Motion {rigid_txt} - interval {interval_s}s - {self.num_segments} segments"
        return txt

    def make_interpolators(self):
        from scipy.interpolate import RegularGridInterpolator

        self.interpolators = [
            RegularGridInterpolator(
                (self.temporal_bins_s[j], self.spatial_bins_um), self.displacement[j], method=self.interpolation_method
            )
            for j in range(self.num_segments)
        ]
        self.temporal_bounds = [(t[0], t[-1]) for t in self.temporal_bins_s]
        self.spatial_bounds = (self.spatial_bins_um.min(), self.spatial_bins_um.max())

    def get_displacement_at_time_and_depth(self, times_s, locations_um, segment_index=None, grid=False):
        """Evaluate the motion estimate at times and positions

        Evaluate the motion estimate, returning the (linearly interpolated) estimated displacement
        at the given times and locations.

        Parameters
        ----------
        times_s: np.array
        locations_um: np.array
            Either this is a one-dimensional array (a vector of positions along self.dimension), or
            else a 2d array with the 2 or 3 spatial dimensions indexed along axis=1.
        segment_index: int, default: None
            The index of the segment to evaluate. If None, and there is only one segment, then that segment is used.
        grid : bool, default: False
            If grid=False, the default, then times_s and locations_um should have the same one-dimensional
            shape, and the returned displacement[i] is the displacement at time times_s[i] and location
            locations_um[i].
            If grid=True, times_s and locations_um determine a grid of positions to evaluate the displacement.
            Then the returned displacement[i,j] is the displacement at depth locations_um[i] and time times_s[j].

        Returns
        -------
        displacement : np.array
            A displacement per input location, of shape times_s.shape if grid=False and (locations_um.size, times_s.size)
            if grid=True.
        """
        if self.interpolators is None:
            self.make_interpolators()

        if segment_index is None:
            if self.num_segments == 1:
                segment_index = 0
            else:
                raise ValueError("Several segment need segment_index=")

        times_s = np.asarray(times_s)
        locations_um = np.asarray(locations_um)

        if locations_um.ndim == 1:
            locations_um = locations_um
        elif locations_um.ndim == 2:
            locations_um = locations_um[:, self.dim]
        else:
            assert False

        times_s = times_s.clip(*self.temporal_bounds[segment_index])
        locations_um = locations_um.clip(*self.spatial_bounds)

        if grid:
            # construct a grid over which to evaluate the displacement
            locations_um, times_s = np.meshgrid(locations_um, times_s, indexing="ij")
            out_shape = times_s.shape
            locations_um = locations_um.ravel()
            times_s = times_s.ravel()
        else:
            # usual case: input is a point cloud
            assert locations_um.shape == times_s.shape
            assert times_s.ndim == 1
            out_shape = times_s.shape

        points = np.column_stack((times_s, locations_um))
        displacement = self.interpolators[segment_index](points)
        # reshape to grid domain shape if necessary
        displacement = displacement.reshape(out_shape)

        return displacement

    def to_dict(self):
        return dict(
            object="Motion",
            displacement=self.displacement,
            temporal_bins_s=self.temporal_bins_s,
            spatial_bins_um=self.spatial_bins_um,
            interpolation_method=self.interpolation_method,
            direction=self.direction,
        )

    @staticmethod
    def from_dict(d):
        return Motion(
            d["displacement"],
            d["temporal_bins_s"],
            d["spatial_bins_um"],
            direction=d["direction"],
            interpolation_method=d["interpolation_method"],
        )

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(exist_ok=False, parents=True)

        info_file = folder / f"spikeinterface_info.json"
        info = dict(
            version=spikeinterface.__version__,
            dev_mode=spikeinterface.DEV_MODE,
            object="Motion",
            num_segments=self.num_segments,
            direction=self.direction,
            interpolation_method=self.interpolation_method,
        )
        with open(info_file, mode="w") as f:
            json.dump(check_json(info), f, indent=4)

        np.save(folder / "spatial_bins_um.npy", self.spatial_bins_um)

        for segment_index in range(self.num_segments):
            np.save(folder / f"displacement_seg{segment_index}.npy", self.displacement[segment_index])
            np.save(folder / f"temporal_bins_s_seg{segment_index}.npy", self.temporal_bins_s[segment_index])

    @classmethod
    def load(cls, folder):
        folder = Path(folder)

        info_file = folder / f"spikeinterface_info.json"
        err_msg = f"Motion.load(folder): the folder {folder} does not contain a Motion object."
        if not info_file.exists():
            raise IOError(err_msg)

        with open(info_file, "r") as f:
            info = json.load(f)
        if "object" not in info or info["object"] != "Motion":
            raise IOError(err_msg)

        direction = info["direction"]
        interpolation_method = info["interpolation_method"]
        spatial_bins_um = np.load(folder / "spatial_bins_um.npy")
        displacement = []
        temporal_bins_s = []
        for segment_index in range(info["num_segments"]):
            displacement.append(np.load(folder / f"displacement_seg{segment_index}.npy"))
            temporal_bins_s.append(np.load(folder / f"temporal_bins_s_seg{segment_index}.npy"))

        return cls(
            displacement,
            temporal_bins_s,
            spatial_bins_um,
            direction=direction,
            interpolation_method=interpolation_method,
        )

    def __eq__(self, other):
        for segment_index in range(self.num_segments):
            if not np.allclose(self.displacement[segment_index], other.displacement[segment_index]):
                return False
            if not np.allclose(self.temporal_bins_s[segment_index], other.temporal_bins_s[segment_index]):
                return False

        if not np.allclose(self.spatial_bins_um, other.spatial_bins_um):
            return False

        return True

    def copy(self):
        return Motion(
            [d.copy() for d in self.displacement],
            [t.copy() for t in self.temporal_bins_s],
            self.spatial_bins_um.copy(),
            direction=self.direction,
            interpolation_method=self.interpolation_method,
        )

    def get_boundaries(self):
        max_ = -np.inf
        min_ = np.inf
        for segment_index, displacement_array in enumerate(self.displacement):
            min_ = min(min_, np.min(displacement_array))
            max_ = max(max_, np.max(displacement_array))
        return min_, max_


def ensure_time_bins(time_bin_centers_s=None, time_bin_edges_s=None):
    """Ensure that both bin edges and bin centers are present

    If either of the inputs are None but not both, the missing is reconstructed
    from the present. Going from edges to centers is done by taking midpoints.
    Going from centers to edges is done by taking midpoints and padding with the
    left and rightmost centers.

    To handle multi segment, this function is working both:
      * array/array input
      * list[array]/list[array] input

    Parameters
    ----------
    time_bin_centers_s : None or np.array or list[np.array]
    time_bin_edges_s : None or np.array or list[np.array]

    Returns
    -------
    time_bin_centers_s, time_bin_edges_s
    """
    if time_bin_centers_s is None and time_bin_edges_s is None:
        raise ValueError("Need at least one of time_bin_centers_s or time_bin_edges_s.")

    if time_bin_centers_s is None:
        if isinstance(time_bin_edges_s, list):
            # multi segment cas
            time_bin_centers_s = []
            for be in time_bin_edges_s:
                bc, _ = ensure_time_bins(time_bin_centers_s=None, time_bin_edges_s=be)
                time_bin_centers_s.append(bc)
        else:
            # simple segment
            assert time_bin_edges_s.ndim == 1 and time_bin_edges_s.size >= 2
            time_bin_centers_s = 0.5 * (time_bin_edges_s[1:] + time_bin_edges_s[:-1])

    if time_bin_edges_s is None:
        if isinstance(time_bin_centers_s, list):
            # multi segment cas
            time_bin_edges_s = []
            for bc in time_bin_centers_s:
                _, be = ensure_time_bins(time_bin_centers_s=bc, time_bin_edges_s=None)
                time_bin_edges_s.append(be)
        else:
            # simple segment
            time_bin_edges_s = np.empty(time_bin_centers_s.shape[0] + 1, dtype=time_bin_centers_s.dtype)
            time_bin_edges_s[[0, -1]] = time_bin_centers_s[[0, -1]]
            if time_bin_centers_s.size > 2:
                time_bin_edges_s[1:-1] = 0.5 * (time_bin_centers_s[1:] + time_bin_centers_s[:-1])

    return time_bin_centers_s, time_bin_edges_s


def ensure_time_bin_edges(time_bin_centers_s=None, time_bin_edges_s=None):
    return ensure_time_bins(time_bin_centers_s, time_bin_edges_s)[1]
