import warnings
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
            displacement=self.displacement,
            temporal_bins_s=self.temporal_bins_s,
            spatial_bins_um=self.spatial_bins_um,
            interpolation_method=self.interpolation_method,
            direction=self.direction,
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


def get_spatial_windows(
    contact_depths,
    spatial_bin_centers,
    rigid=False,
    win_shape="gaussian",
    win_step_um=50.0,
    win_scale_um=150.0,
    win_margin_um=None,
    zero_threshold=None,
):
    """
    Generate spatial windows (taper) for non-rigid motion.
    For rigid motion, this is equivalent to have one unique rectangular window that covers the entire probe.
    The windowing can be gaussian or rectangular.
    Windows are centered between the min/max of contact_depths.
    We can ensure window to not be to close from border with win_margin_um.


    Parameters
    ----------
    contact_depths : np.ndarray
        Position of electrodes of the corection direction shape=(num_channels, )
    spatial_bin_centers : np.array
        The pre-computed spatial bin centers
    rigid : bool, default False
        If True, returns a single rectangular window
    win_shape : str, default "gaussian"
        Shape of the window
        "gaussian" | "rect" | "triangle"
    win_step_um : float
        The steps at which windows are defined
    win_scale_um : float, default 150.
        Sigma of gaussian window if win_shape is gaussian
        Width of the rectangle if win_shape is rect
    win_margin_um : None | float, default None
        The margin to extend (if positive) or shrink (if negative) the probe dimension to compute windows.
        When None, then the margin is set to -win_scale_um./2
    zero_threshold: None | float
        Lower value for thresholding to set zeros.

    Returns
    -------
    windows : 2D arrays
        The scaling for each window. Each element has num_spatial_bins values
        shape: (num_window, spatial_bins)
    window_centers: 1D np.array
        The center of each window

    Notes
    -----
    Note that kilosort2.5 uses overlaping rectangular windows.
    Here by default we use gaussian window.

    """
    n = spatial_bin_centers.size

    if rigid:
        # win_shape = 'rect' is forced
        windows, window_centers = get_rigid_windows(spatial_bin_centers)
    else:
        if win_scale_um <= win_step_um / 5.0:
            warnings.warn(
                f"get_spatial_windows(): spatial windows are probably not overlapping because {win_scale_um=} and {win_step_um=}"
            )

        if win_margin_um is None:
            # this ensure that first/last windows do not overflow outside the probe
            win_margin_um = -win_scale_um / 2.0

        min_ = np.min(contact_depths) - win_margin_um
        max_ = np.max(contact_depths) + win_margin_um
        num_windows = int((max_ - min_) // win_step_um)

        if num_windows < 1:
            raise Exception(
                f"get_spatial_windows(): {win_step_um=}/{win_scale_um=}/{win_margin_um=} are too large for the "
                f"probe size (depth range={np.ptp(contact_depths)}). You can try to reduce them or use rigid motion."
            )
        border = ((max_ - min_) % win_step_um) / 2
        window_centers = np.arange(num_windows + 1) * win_step_um + min_ + border
        windows = []

        for win_center in window_centers:
            if win_shape == "gaussian":
                win = np.exp(-((spatial_bin_centers - win_center) ** 2) / (2 * win_scale_um**2))
            elif win_shape == "rect":
                win = np.abs(spatial_bin_centers - win_center) < (win_scale_um / 2.0)
                win = win.astype("float64")
            elif win_shape == "triangle":
                center_dist = np.abs(spatial_bin_centers - win_center)
                in_window = center_dist <= (win_scale_um / 2.0)
                win = -center_dist
                win[~in_window] = 0
                win[in_window] -= win[in_window].min()
                win[in_window] /= win[in_window].max()
            windows.append(win)

    windows = np.array(windows)

    if zero_threshold is not None:
        windows[windows < zero_threshold] = 0
        windows /= windows.sum(axis=1, keepdims=True)

    return windows, window_centers


def get_rigid_windows(spatial_bin_centers):
    """Generate a single rectangular window for rigid motion."""
    windows = np.ones((1, spatial_bin_centers.size), dtype="float64")
    window_centers = np.array([(spatial_bin_centers[0] + spatial_bin_centers[-1]) / 2.0])
    return windows, window_centers


def get_window_domains(windows):
    """Array of windows -> list of slices where window > 0."""
    slices = []
    for w in windows:
        in_window = np.flatnonzero(w)
        slices.append(slice(in_window[0], in_window[-1] + 1))
    return slices


def scipy_conv1d(input, weights, padding="valid"):
    """SciPy translation of torch F.conv1d"""
    from scipy.signal import correlate

    n, c_in, length = input.shape
    c_out, in_by_groups, kernel_size = weights.shape
    assert in_by_groups == c_in == 1

    if padding == "same":
        mode = "same"
        length_out = length
    elif padding == "valid":
        mode = "valid"
        length_out = length - 2 * (kernel_size // 2)
    elif isinstance(padding, int):
        mode = "valid"
        input = np.pad(input, [*[(0, 0)] * (input.ndim - 1), (padding, padding)])
        length_out = length - (kernel_size - 1) + 2 * padding
    else:
        raise ValueError(f"Unknown 'padding' value of {padding}, 'padding' must be 'same', 'valid' or an integer")

    output = np.zeros((n, c_out, length_out), dtype=input.dtype)
    for m in range(n):
        for c in range(c_out):
            output[m, c] = correlate(input[m, 0], weights[c, 0], mode=mode)

    return output


def get_spatial_bin_edges(recording, direction, hist_margin_um, bin_um):
    # contact along one axis
    probe = recording.get_probe()
    dim = ["x", "y", "z"].index(direction)
    contact_depths = probe.contact_positions[:, dim]

    min_ = np.min(contact_depths) - hist_margin_um
    max_ = np.max(contact_depths) + hist_margin_um
    spatial_bins = np.arange(min_, max_ + bin_um, bin_um)

    return spatial_bins


def make_2d_motion_histogram(
    recording,
    peaks,
    peak_locations,
    weight_with_amplitude=False,
    avg_in_bin=True,
    direction="y",
    bin_s=1.0,
    bin_um=2.0,
    hist_margin_um=50,
    spatial_bin_edges=None,
    depth_smooth_um=None,
    time_smooth_s=None,
):
    """
    Generate 2d motion histogram in depth and time.

    Parameters
    ----------
    recording : BaseRecording
        The input recording
    peaks : np.array
        The peaks array
    peak_locations : np.array
        Array with peak locations
    weight_with_amplitude : bool, default: False
        If True, motion histogram is weighted by amplitudes
    avg_in_bin : bool, default True
        If true, average the amplitudes in each bin.
        This is done only if weight_with_amplitude=True.
    direction : "x" | "y" | "z", default: "y"
        The depth direction
    bin_s : float, default: 1.0
        The temporal bin duration in s
    bin_um : float, default: 2.0
        The spatial bin size in um. Ignored if spatial_bin_edges is given.
    hist_margin_um : float, default: 50
        The margin to add to the minimum and maximum positions before spatial binning.
        Ignored if spatial_bin_edges is given.
    spatial_bin_edges : np.array, default: None
        The pre-computed spatial bin edges
    depth_smooth_um: None or float
        Optional gaussian smoother on histogram on depth axis.
        This is given as the sigma of the gaussian in micrometers.
    time_smooth_s: None or float
        Optional gaussian smoother on histogram on time axis.
        This is given as the sigma of the gaussian in seconds.

    Returns
    -------
    motion_histogram
        2d np.array with motion histogram (num_temporal_bins, num_spatial_bins)
    temporal_bin_edges
        1d array with temporal bin edges
    spatial_bin_edges
        1d array with spatial bin edges
    """
    n_samples = recording.get_num_samples()
    mint_s = recording.sample_index_to_time(0)
    maxt_s = recording.sample_index_to_time(n_samples - 1)
    temporal_bin_edges = np.arange(mint_s, maxt_s + bin_s, bin_s)
    if spatial_bin_edges is None:
        spatial_bin_edges = get_spatial_bin_edges(recording, direction, hist_margin_um, bin_um)
    else:
        bin_um = spatial_bin_edges[1] - spatial_bin_edges[0]

    arr = np.zeros((peaks.size, 2), dtype="float64")
    arr[:, 0] = recording.sample_index_to_time(peaks["sample_index"])
    arr[:, 1] = peak_locations[direction]

    if weight_with_amplitude:
        weights = np.abs(peaks["amplitude"])
    else:
        weights = None

    motion_histogram, edges = np.histogramdd(arr, bins=(temporal_bin_edges, spatial_bin_edges), weights=weights)

    # average amplitude in each bin
    if weight_with_amplitude and avg_in_bin:
        bin_counts, _ = np.histogramdd(arr, bins=(temporal_bin_edges, spatial_bin_edges))
        bin_counts[bin_counts == 0] = 1
        motion_histogram = motion_histogram / bin_counts

    from scipy.ndimage import gaussian_filter1d

    if depth_smooth_um is not None:
        motion_histogram = gaussian_filter1d(motion_histogram, depth_smooth_um / bin_um, axis=1, mode="constant")

    if time_smooth_s is not None:
        motion_histogram = gaussian_filter1d(motion_histogram, time_smooth_s / bin_s, axis=0, mode="constant")

    return motion_histogram, temporal_bin_edges, spatial_bin_edges


def make_3d_motion_histograms(
    recording,
    peaks,
    peak_locations,
    direction="y",
    bin_s=1.0,
    bin_um=2.0,
    hist_margin_um=50,
    num_amp_bins=20,
    log_transform=True,
    spatial_bin_edges=None,
):
    """
    Generate 3d motion histograms in depth, amplitude, and time.
    This is used by the "iterative_template_registration" (Kilosort2.5) method.


    Parameters
    ----------
    recording : BaseRecording
        The input recording
    peaks : np.array
        The peaks array
    peak_locations : np.array
        Array with peak locations
    direction : "x" | "y" | "z", default: "y"
        The depth direction
    bin_s : float, default: 1.0
        The temporal bin duration in s.
    bin_um : float, default: 2.0
        The spatial bin size in um. Ignored if spatial_bin_edges is given.
    hist_margin_um : float, default: 50
        The margin to add to the minimum and maximum positions before spatial binning.
        Ignored if spatial_bin_edges is given.
    log_transform : bool, default: True
        If True, histograms are log-transformed
    spatial_bin_edges : np.array, default: None
        The pre-computed spatial bin edges

    Returns
    -------
    motion_histograms
        3d np.array with motion histogram (num_temporal_bins, num_spatial_bins, num_amp_bins)
    temporal_bin_edges
        1d array with temporal bin edges
    spatial_bin_edges
        1d array with spatial bin edges
    """
    n_samples = recording.get_num_samples()
    mint_s = recording.sample_index_to_time(0)
    maxt_s = recording.sample_index_to_time(n_samples - 1)
    temporal_bin_edges = np.arange(mint_s, maxt_s + bin_s, bin_s)
    if spatial_bin_edges is None:
        spatial_bin_edges = get_spatial_bin_edges(recording, direction, hist_margin_um, bin_um)

    # pre-compute abs amplitude and ranges for scaling
    amplitude_bin_edges = np.linspace(0, 1, num_amp_bins + 1)
    abs_peaks = np.abs(peaks["amplitude"])
    max_peak_amp = np.max(abs_peaks)
    min_peak_amp = np.min(abs_peaks)
    # log amplitudes and scale between 0-1
    abs_peaks_log_norm = (np.log10(abs_peaks) - np.log10(min_peak_amp)) / (
        np.log10(max_peak_amp) - np.log10(min_peak_amp)
    )

    arr = np.zeros((peaks.size, 3), dtype="float64")
    arr[:, 0] = recording.sample_index_to_time(peaks["sample_index"])
    arr[:, 1] = peak_locations[direction]
    arr[:, 2] = abs_peaks_log_norm

    motion_histograms, edges = np.histogramdd(
        arr,
        bins=(
            temporal_bin_edges,
            spatial_bin_edges,
            amplitude_bin_edges,
        ),
    )

    if log_transform:
        motion_histograms = np.log2(1 + motion_histograms)

    return motion_histograms, temporal_bin_edges, spatial_bin_edges
