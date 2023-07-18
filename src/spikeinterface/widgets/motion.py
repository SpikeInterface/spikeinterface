import numpy as np
from warnings import warn

from .base import BaseWidget
from .utils import get_unit_colors


from ..core.template_tools import get_template_extremum_amplitude


class MotionWidget(BaseWidget):
    """
    Plot unit depths

    Parameters
    ----------
    motion_info: dict
        The motion info return by correct_motion() or load back with load_motion_info()
    recording : RecordingExtractor, optional
        The recording extractor object (only used to get "real" times), default None
    sampling_frequency : float, optional
        The sampling frequency (needed if recording is None), default None
    depth_lim : tuple
        The min and max depth to display, default None (min and max of the recording)
    motion_lim : tuple
        The min and max motion to display, default None (min and max of the motion)
    color_amplitude : bool
        If True, the color of the scatter points is the amplitude of the peaks, default False
    scatter_decimate : int
        If > 1, the scatter points are decimated, default None
    amplitude_cmap : str
        The colormap to use for the amplitude, default 'inferno'
    amplitude_clim : tuple
        The min and max amplitude to display, default None (min and max of the amplitudes)
    amplitude_alpha : float
        The alpha of the scatter points, default 0.5
    """

    possible_backends = {}

    def __init__(
        self,
        motion_info,
        recording=None,
        depth_lim=None,
        motion_lim=None,
        color_amplitude=False,
        scatter_decimate=None,
        amplitude_cmap="inferno",
        amplitude_clim=None,
        amplitude_alpha=1,
        backend=None,
        **backend_kwargs,
    ):
        times = recording.get_times() if recording is not None else None

        plot_data = dict(
            sampling_frequency=motion_info["parameters"]["sampling_frequency"],
            times=times,
            depth_lim=depth_lim,
            motion_lim=motion_lim,
            color_amplitude=color_amplitude,
            scatter_decimate=scatter_decimate,
            amplitude_cmap=amplitude_cmap,
            amplitude_clim=amplitude_clim,
            amplitude_alpha=amplitude_alpha,
            **motion_info,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)
