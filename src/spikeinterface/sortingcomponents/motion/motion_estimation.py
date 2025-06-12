from __future__ import annotations

import warnings
import numpy as np


from spikeinterface.sortingcomponents.tools import make_multi_method_doc

from spikeinterface.core.motion import Motion
from .decentralized import DecentralizedRegistration
from .iterative_template import IterativeTemplateRegistration
from .dredge import DredgeLfpRegistration, DredgeApRegistration
from .medicine import MedicineRegistration


# estimate_motion > infer_motion
def estimate_motion(
    recording,
    peaks=None,
    peak_locations=None,
    direction="y",
    rigid=False,
    win_shape="gaussian",
    win_step_um=200.0,
    win_scale_um=300.0,
    win_margin_um=None,
    method="decentralized",
    extra_outputs=False,
    progress_bar=False,
    verbose=False,
    margin_um=None,
    **method_kwargs,
) -> Motion | tuple[Motion, dict]:
    """
    Estimate motion with several possible methods.

    Most of methods except dredge_lfp needs peaks and after their localization.

    Note that the way you detect peak locations (center of mass/monopolar_triangulation/grid_convolution)
    have an impact on the result.

    Parameters
    ----------
    recording : BaseRecording
        The recording extractor
    peaks : numpy array
        Peak vector (complex dtype).
        Needed for decentralized and iterative_template methods.
    peak_locations : numpy array
        Complex dtype with "x", "y", "z" fields
        Needed for decentralized and iterative_template methods.
    direction : "x" | "y" | "z", default: "y"
        Dimension on which the motion is estimated. "y" is depth along the probe.

    {method_doc}

    rigid : bool, default: False
        Compute rigid (one motion for the entire probe) or non rigid motion
        Rigid computation is equivalent to non-rigid with only one window with rectangular shape.
    win_shape : "gaussian" | "rect" | "triangle", default: "gaussian"
        The shape of the windows for non rigid.
        When rigid this is force to "rect"
        Nonrigid window-related arguments
        The depth domain will be broken up into windows with shape controlled by win_shape,
        spaced by win_step_um at a margin of win_margin_um from the boundary, and with
        width controlled by win_scale_um.
        When win_margin_um is None the margin is automatically set to -win_scale_um/2.
        See get_spatial_windows.
    win_step_um : float, default: 50
        See win_shape
    win_scale_um : float, default: 150
        See win_shape
    win_margin_um : None | float, default: None
        See win_shape
    extra_outputs : bool, default: False
        If True then return an extra dict that contains variables
        to check intermediate steps (motion_histogram, non_rigid_windows, pairwise_displacement)
    progress_bar : bool, default: False
        Display progress bar or not
    verbose : bool, default: False
        If True, output is verbose
    **method_kwargs :

    Returns
    -------
    motion: Motion object
        The motion object.
    extra: dict
        Optional output if `extra_outputs=True`
        This dict contain histogram, pairwise_displacement usefull for ploting.
    """

    if margin_um is not None:
        warnings.warn("estimate_motion() margin_um has been removed used hist_margin_um or win_margin_um")

    # TODO handle multi segment one day : Charlie this is for you
    assert recording.get_num_segments() == 1, "At the moment estimate_motion handle only unique segment"

    method_class = estimate_motion_methods[method]

    if method_class.need_peak_location:
        if peaks is None or peak_locations is None:
            raise ValueError(f"estimate_motion: the method {method} need peaks and peak_locations")

    if extra_outputs:
        extra = {}
    else:
        extra = None

    # run method
    motion = method_class.run(
        recording,
        peaks,
        peak_locations,
        direction,
        rigid,
        win_shape,
        win_step_um,
        win_scale_um,
        win_margin_um,
        verbose,
        progress_bar,
        extra,
        **method_kwargs,
    )

    if extra_outputs:
        return motion, extra
    else:
        return motion


_methods_list = [
    DecentralizedRegistration,
    IterativeTemplateRegistration,
    DredgeLfpRegistration,
    DredgeApRegistration,
    MedicineRegistration,
]
estimate_motion_methods = {m.name: m for m in _methods_list}
method_doc = make_multi_method_doc(_methods_list)
estimate_motion.__doc__ = estimate_motion.__doc__.format(method_doc=method_doc)
