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
    recording : RecordingExtractor
        The recording extractor object
    motion_info: dict
        The motion info return by correct_motion() or load back with load_motion_info()
    depth_lim: tuple
        The min and max depth to display, default None (min and max of the recording)
    motion_lim: tuple
        The min and max motion to display, default None (min and max of the motion)
    color_amplitude: bool
        If True, the color of the scatter points is the amplitude of the peaks, default False
    scatter_decimate: int
        If > 1, the scatter points are decimated, default None
    amplitude_cmap: str
        The colormap to use for the amplitude, default 'inferno'
    """

    possible_backends = {}

    def __init__(
        self,
        recording,
        motion_info,
        depth_lim=None,
        motion_lim=None,
        color_amplitude=False,
        scatter_decimate=None,
        amplitude_cmap="inferno",
        backend=None,
        **backend_kwargs,
    ):
        plot_data = dict(
            rec=recording,
            depth_lim=depth_lim,
            motion_lim=motion_lim,
            color_amplitude=color_amplitude,
            scatter_decimate=scatter_decimate,
            amplitude_cmap=amplitude_cmap,
            **motion_info,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)
