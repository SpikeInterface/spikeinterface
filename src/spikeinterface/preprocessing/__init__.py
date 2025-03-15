from .preprocessinglist import *

from .motion import correct_motion, load_motion_info, save_motion_info, get_motion_parameters_preset, get_motion_presets

""" For this to work, I think `get_spatial_interpolation_kernel` could go to core
    (or elsewhere) to avoid circular imports. Currently  sortingcomponents.motion
    requires it but  inter-session-alignment requires sortingcomponents.motion.

from .inter_session_alignment.session_alignment import (
    get_estimate_histogram_kwargs,
    get_compute_alignment_kwargs,
    get_non_rigid_window_kwargs,
    get_interpolate_motion_kwargs,
    align_sessions,
    align_sessions_after_motion_correction,
    compute_peaks_locations_for_session_alignment,
)
"""
from .preprocessing_tools import get_spatial_interpolation_kernel
from .detect_bad_channels import detect_bad_channels
from .correct_lsb import correct_lsb


# for snippets
from .align_snippets import AlignSnippets
