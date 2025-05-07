from __future__ import annotations

import copy
import warnings
import json
import shutil
import time
import inspect
from pathlib import Path
import numpy as np
from typing import Literal

from spikeinterface.core import get_noise_levels, fix_job_kwargs
from spikeinterface.core.job_tools import _shared_job_kwargs_doc
from spikeinterface.core.core_tools import SIJsonEncoder
from spikeinterface.core.job_tools import _shared_job_kwargs_doc
from spikeinterface.core import BaseRecording


motion_options_preset = {
    # dredge
    "dredge": {
        "doc": "Official Dredge preset",
        "detect_kwargs": dict(
            method="locally_exclusive",
            peak_sign="neg",
            detect_threshold=8.0,
            exclude_sweep_ms=0.8,
            radius_um=80.0,
        ),
        "select_kwargs": dict(),
        "localize_peaks_kwargs": dict(
            method="monopolar_triangulation",
        ),
        "estimate_motion_kwargs": dict(
            method="dredge_ap",
            direction="y",
            rigid=False,
            win_shape="gaussian",
            win_step_um=400.0,
            win_scale_um=400.0,
            win_margin_um=None,
        ),
        "interpolate_motion_kwargs": dict(
            border_mode="force_extrapolate", spatial_interpolation_method="kriging", sigma_um=20.0, p=2
        ),
    },
    "medicine": {
        "doc": "Medicine method: https://jazlab.github.io/medicine/",
        "detect_kwargs": dict(
            method="locally_exclusive",
        ),
        "localize_peaks_kwargs": dict(method="monopolar_triangulation"),
        "select_kwargs": dict(),
        "estimate_motion_kwargs": dict(method="medicine"),
        "interpolate_motion_kwargs": dict(),
    },
    # similar than dredge but faster
    "dredge_fast": {
        "doc": "Modified and faster Dredge preset",
        "detect_kwargs": dict(
            method="locally_exclusive",
            peak_sign="neg",
            detect_threshold=8.0,
            exclude_sweep_ms=0.8,
            radius_um=80.0,
        ),
        "select_kwargs": dict(),
        "localize_peaks_kwargs": dict(
            method="grid_convolution",
        ),
        "estimate_motion_kwargs": dict(
            method="dredge_ap",
            direction="y",
            rigid=False,
            win_shape="gaussian",
            win_step_um=400.0,
            win_scale_um=400.0,
            win_margin_um=None,
        ),
        "interpolate_motion_kwargs": dict(
            border_mode="force_extrapolate", spatial_interpolation_method="kriging", sigma_um=20.0, p=2
        ),
    },
    # This preset is the encestor of dredge
    "nonrigid_accurate": {
        "doc": "method by Paninski lab (monopolar_triangulation + decentralized)",
        "detect_kwargs": dict(
            method="locally_exclusive",
            peak_sign="neg",
            detect_threshold=8.0,
            exclude_sweep_ms=0.8,
            radius_um=80.0,
        ),
        "select_kwargs": dict(),
        "localize_peaks_kwargs": dict(method="monopolar_triangulation"),
        "estimate_motion_kwargs": dict(method="decentralized", direction="y", rigid=False),
        "interpolate_motion_kwargs": dict(
            border_mode="remove_channels", spatial_interpolation_method="kriging", sigma_um=20.0, p=2
        ),
    },
    "nonrigid_fast_and_accurate": {
        "doc": "mixed methods by KS & Paninski lab (grid_convolution + decentralized)",
        "detect_kwargs": dict(
            method="locally_exclusive",
            peak_sign="neg",
            detect_threshold=8.0,
            exclude_sweep_ms=0.8,
            radius_um=80.0,
        ),
        "select_kwargs": dict(),
        "localize_peaks_kwargs": dict(method="grid_convolution"),
        "estimate_motion_kwargs": dict(method="decentralized", direction="y", rigid=False),
        "interpolate_motion_kwargs": dict(
            border_mode="remove_channels", spatial_interpolation_method="kriging", sigma_um=20.0, p=2
        ),
    },
    # This preset is a super fast rigid estimation with center of mass
    "rigid_fast": {
        "doc": "Rigid and not super accurate but fast. Use center of mass.",
        "detect_kwargs": dict(
            method="locally_exclusive",
            peak_sign="neg",
            detect_threshold=8.0,
            exclude_sweep_ms=0.1,
            radius_um=75.0,
        ),
        "select_kwargs": dict(),
        # "localize_peaks_kwargs": dict(method="grid_convolution"),
        "localize_peaks_kwargs": dict(method="center_of_mass"),
        "estimate_motion_kwargs": dict(method="dredge_ap", bin_s=5.0, rigid=True),
        "interpolate_motion_kwargs": dict(
            border_mode="remove_channels", spatial_interpolation_method="kriging", sigma_um=20.0, p=2
        ),
    },
    # This preset try to mimic kilosort2.5 motion estimator
    "kilosort_like": {
        "doc": "Mimic the drift correction of kilosort (grid_convolution + iterative_template)",
        "detect_kwargs": dict(
            method="locally_exclusive",
            peak_sign="neg",
            detect_threshold=8.0,
            exclude_sweep_ms=0.1,
            radius_um=50,
        ),
        "select_kwargs": dict(),
        "localize_peaks_kwargs": dict(
            method="grid_convolution",
            weight_method={"mode": "gaussian_2d", "sigma_list_um": np.linspace(5, 25, 5)},
        ),
        "estimate_motion_kwargs": dict(
            method="iterative_template",
            bin_s=2.0,
            rigid=False,
            win_step_um=200.0,
            win_scale_um=400.0,
            hist_margin_um=0,
            win_shape="rect",
        ),
        "interpolate_motion_kwargs": dict(
            border_mode="force_extrapolate", spatial_interpolation_method="kriging", sigma_um=20.0, p=2
        ),
    },
    # empty preset
    "": {
        "detect_kwargs": {},
        "select_kwargs": {},
        "localize_peaks_kwargs": {},
        "estimate_motion_kwargs": {},
        "interpolate_motion_kwargs": {},
    },
}


def _get_default_motion_params():
    # dirty code that inspect class to get parameters
    # when multi method for detect_peak/localize_peak/estimate_motion

    params = dict()

    from spikeinterface.sortingcomponents.peak_detection import detect_peak_methods
    from spikeinterface.sortingcomponents.peak_localization import localize_peak_methods
    from spikeinterface.sortingcomponents.motion.motion_estimation import estimate_motion_methods, estimate_motion

    params["detect_kwargs"] = dict()
    for method_name, method_class in detect_peak_methods.items():
        if hasattr(method_class, "check_params"):
            sig = inspect.signature(method_class.check_params)
            params["detect_kwargs"][method_name] = {
                k: v.default for k, v in sig.parameters.items() if k != "self" and v.default != inspect.Parameter.empty
            }

    # no design by subclass
    params["select_kwargs"] = dict()

    params["localize_peaks_kwargs"] = dict()
    for method_name, method_class in localize_peak_methods.items():
        sig = inspect.signature(method_class.__init__)
        p = {k: v.default for k, v in sig.parameters.items() if k != "self" and v.default != inspect.Parameter.empty}
        p.pop("parents", None)
        p.pop("return_output", None)
        p.pop("return_tensor", None)
        params["localize_peaks_kwargs"][method_name] = p

    params["estimate_motion_kwargs"] = dict()
    for method_name, method_class in estimate_motion_methods.items():
        sig = inspect.signature(estimate_motion)
        p = {k: v.default for k, v in sig.parameters.items() if k != "self" and v.default != inspect.Parameter.empty}
        for k in ("peaks", "peak_locations", "method", "extra_outputs", "verbose", "progress_bar", "margin_um"):
            p.pop(k)

        sig = inspect.signature(method_class.run)
        p.update(
            {k: v.default for k, v in sig.parameters.items() if k != "self" and v.default != inspect.Parameter.empty}
        )
        params["estimate_motion_kwargs"][method_name] = p

    # no design by subclass
    params["interpolate_motion_kwargs"] = dict()

    return params


def get_motion_presets():
    preset_keys = list(motion_options_preset.keys())
    preset_keys.remove("")
    return preset_keys


def get_motion_parameters_preset(preset):
    """
    Get the parameters tree for a given preset for motion correction.

    Parameters
    ----------
    preset : str, default: None
        The preset name. See available presets using `spikeinterface.preprocessing.get_motion_presets()`.
    """
    preset_params = copy.deepcopy(motion_options_preset[preset])
    all_default_params = _get_default_motion_params()
    params = dict()
    for step, step_params in preset_params.items():
        if isinstance(step_params, str):
            # the doc key
            params[step] = step_params

        elif len(step_params) == 0:
            # empty dict with no methods = skip the step (select_peaks for instance)
            params[step] = dict()

        elif isinstance(step_params, dict):
            if "method" in step_params:
                method = step_params["method"]
                params[step] = all_default_params[step][method]
            else:
                params[step] = dict()
            params[step].update(step_params)
        else:
            raise ValueError(f"Preset {preset} is wrong")

    return params


def _update_motion_kwargs(preset, detect_kwargs, select_kwargs, localize_peaks_kwargs, estimate_motion_kwargs):

    params = motion_options_preset[preset]
    detect_kwargs = dict(params["detect_kwargs"], **detect_kwargs)
    select_kwargs = dict(params["select_kwargs"], **select_kwargs)
    localize_peaks_kwargs = dict(params["localize_peaks_kwargs"], **localize_peaks_kwargs)
    estimate_motion_kwargs = dict(params["estimate_motion_kwargs"], **estimate_motion_kwargs)

    return detect_kwargs, select_kwargs, localize_peaks_kwargs, estimate_motion_kwargs


def _update_interpolation_kwargs(preset, interpolation_kwargs):

    params = motion_options_preset[preset]
    interpolation_kwargs = dict(params["interpolate_motion_kwargs"], **interpolation_kwargs)

    return interpolation_kwargs


def compute_motion(
    recording: BaseRecording,
    preset: Literal[
        "dredge",
        "medicine",
        "dredge_fast",
        "nonrigid_accurate",
        "nonrigid_fast_and_accurate",
        "rigid_fast",
        "kilosort_like",
    ] = "dredge_fast",
    detect_kwargs: dict = {},
    select_kwargs: dict = {},
    localize_peaks_kwargs: dict = {},
    estimate_motion_kwargs: dict = {},
    output_motion_info: bool = False,
    folder: str | Path | None = None,
    overwrite: bool = False,
    raise_error: bool = True,
    **job_kwargs,
) -> dict:
    """
    Function to compute motion correction based on a preset e.g. 'dredge' or 'medicine'.

    This function has some intermediate steps that can be controlled one by one with parameters:
      * detect peaks
      * (optional) sub-sample peaks to speed up the localization
      * localize peaks
      * estimate the motion

    The recording must be preprocessed (filter and denoised at least), and we recommend to not use whitening before motion
    estimation. Since the motion interpolation requires a "float" recording, the recording is cast to float32 if necessary.

    Parameters for each step are handled as separate dictionaries.
    For more information please check the documentation of the following functions:

      * :py:func:`~spikeinterface.sortingcomponents.peak_detection.detect_peaks`
      * :py:func:`~spikeinterface.sortingcomponents.peak_selection.select_peaks`
      * :py:func:`~spikeinterface.sortingcomponents.peak_localization.localize_peaks`
      * :py:func:`~spikeinterface.sortingcomponents.motion.motion.estimate_motion`

    {}
    {}
    raise_error : bool, default: True
        If True, an error is raised if motion estimation fails
        If False, the process continues and the peaks and peak_locations are still returned in `motion_info`.

    Returns
    =======
    motion_info : dict
        A dictionary containing a motion objects, peaks, peak locations, run_times and the parameters used to compute these.
    """

    # local import are important because "sortingcomponents" is not important by default
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks, detect_peak_methods
    from spikeinterface.sortingcomponents.peak_selection import select_peaks
    from spikeinterface.sortingcomponents.peak_localization import localize_peaks, localize_peak_methods
    from spikeinterface.core.node_pipeline import ExtractDenseWaveforms, run_node_pipeline
    from spikeinterface.sortingcomponents.motion.motion_estimation import estimate_motion, estimate_motion_methods

    # get preset params and update if necessary
    detect_kwargs, select_kwargs, localize_peaks_kwargs, estimate_motion_kwargs = _update_motion_kwargs(
        preset, detect_kwargs, select_kwargs, localize_peaks_kwargs, estimate_motion_kwargs
    )

    job_kwargs = fix_job_kwargs(job_kwargs)

    # params
    parameters = dict(
        preset=preset,
        detect_kwargs=detect_kwargs,
        select_kwargs=select_kwargs,
        localize_peaks_kwargs=localize_peaks_kwargs,
        estimate_motion_kwargs=estimate_motion_kwargs,
        job_kwargs=job_kwargs,
        sampling_frequency=recording.sampling_frequency,
    )

    progress_bar = job_kwargs.get("progress_bar", False)

    noise_levels = get_noise_levels(recording, return_scaled=False)

    if folder is not None:
        folder = Path(folder)
        if overwrite:
            if folder.is_dir():
                import shutil

                shutil.rmtree(folder)
        else:
            assert not folder.is_dir(), f"Folder {folder} already exists"

    no_selection_kwargs = len(select_kwargs) == 0

    if no_selection_kwargs:
        # maybe do this directly in the folder when not None, but might be slow on external storage
        gather_mode = "memory"
        # node detect
        detect_peaks_method = detect_kwargs["method"]
        method_class = detect_peak_methods[detect_peaks_method]
        detect_kwargs_without_method = {
            key: detect_kwarg for key, detect_kwarg in detect_kwargs.items() if key != "method"
        }
        node0 = method_class(recording, **detect_kwargs_without_method)

        node1 = ExtractDenseWaveforms(recording, parents=[node0], ms_before=0.1, ms_after=0.3)

        # node detect + localize
        method = localize_peaks_kwargs["method"]
        method_class = localize_peak_methods[method]
        localize_peaks_kwargs_without_method = {
            key: localize_peaks_kwarg for key, localize_peaks_kwarg in localize_peaks_kwargs.items() if key != "method"
        }
        node2 = method_class(
            recording, parents=[node0, node1], return_output=True, **localize_peaks_kwargs_without_method
        )
        pipeline_nodes = [node0, node1, node2]
        t0 = time.perf_counter()
        peaks, peak_locations = run_node_pipeline(
            recording,
            pipeline_nodes,
            job_kwargs,
            job_name="detect and localize",
            gather_mode=gather_mode,
            gather_kwargs=None,
            squeeze_output=False,
            folder=None,
            names=None,
        )
        t1 = time.perf_counter()
        run_times = dict(detect_and_localize=t1 - t0)
    else:
        # localization is done after select_peaks()
        pipeline_nodes = None

        t0 = time.perf_counter()
        peaks = detect_peaks(recording, noise_levels=noise_levels, pipeline_nodes=None, **detect_kwargs, **job_kwargs)
        t1 = time.perf_counter()
        # select some peaks
        peaks = select_peaks(peaks, **select_kwargs, **job_kwargs)
        t2 = time.perf_counter()
        peak_locations = localize_peaks(recording, peaks, **localize_peaks_kwargs, **job_kwargs)
        t3 = time.perf_counter()

        run_times = dict(
            detect_peaks=t1 - t0,
            select_peaks=t2 - t1,
            localize_peaks=t3 - t2,
        )

    t0 = time.perf_counter()
    try:
        motion = estimate_motion(recording, peaks, peak_locations, progress_bar=progress_bar, **estimate_motion_kwargs)
    except Exception as err:
        error_message = f"Motion estimation failed. Error given: {err}."
        if raise_error:
            raise RuntimeError(error_message)
        else:
            warnings.warn(error_message)
        motion = None
    t1 = time.perf_counter()
    run_times["estimate_motion"] = t1 - t0

    if recording.get_dtype().kind != "f":
        recording = recording.astype("float32")

    motion_info = dict(
        parameters=parameters,
        run_times=run_times,
        peaks=peaks,
        peak_locations=peak_locations,
        motion=motion,
    )
    if folder is not None:
        save_motion_info(motion_info, folder, overwrite=overwrite)

    if output_motion_info:
        return motion, motion_info
    else:
        return motion


def correct_motion(
    recording: BaseRecording,
    preset: Literal[
        "dredge",
        "medicine",
        "dredge_fast",
        "nonrigid_accurate",
        "nonrigid_fast_and_accurate",
        "rigid_fast",
        "kilosort_like",
    ] = "dredge_fast",
    folder: str | Path | None = None,
    output_motion: bool = False,
    output_motion_info: bool = False,
    overwrite: bool = False,
    detect_kwargs: dict = {},
    select_kwargs: dict = {},
    localize_peaks_kwargs: dict = {},
    estimate_motion_kwargs: dict = {},
    interpolate_motion_kwargs: dict = {},
    **job_kwargs,
):
    """
    High-level function that estimates the motion and interpolates the recording.

    This function has some intermediate steps that can be controlled one by one with parameters:
      * detect peaks
      * (optional) sub-sample peaks to speed up the localization
      * localize peaks
      * estimate the motion
      * create and return a `InterpolateMotionRecording` recording object

    Even if this function is convenient, we recommend to run all step separately for fine tuning.

    Optionally, this function can create a folder with files and figures ready to check.

    This function depends on several modular components of :py:mod:`spikeinterface.sortingcomponents`.

    If `select_kwargs` is None then all peak are used for localized.

    The recording must be preprocessed (filter and denoised at least), and we recommend to not use whitening before motion
    estimation. Since the motion interpolation requires a "float" recording, the recording is cast to float32 if necessary.

    Parameters for each step are handled as separate dictionaries.
    For more information please check the documentation of the following functions:

      * :py:func:`~spikeinterface.sortingcomponents.peak_detection.detect_peaks`
      * :py:func:`~spikeinterface.sortingcomponents.peak_selection.select_peaks`
      * :py:func:`~spikeinterface.sortingcomponents.peak_localization.localize_peaks`
      * :py:func:`~spikeinterface.sortingcomponents.motion.motion.estimate_motion`
      * :py:func:`~spikeinterface.sortingcomponents.motion.motion.interpolate_motion`


    Possible presets : {}

    {}
    output_motion : bool, default: False
        It True, the function returns a `motion` object.

    {}

    Returns
    -------
    recording_corrected : Recording
        The motion corrected recording
    motion : Motion
        Optional output if `output_motion=True`.
    motion_info : dict
        Optional output if `output_motion_info=True`. This dict contains several variable for
        for plotting. See `plot_motion_info()`
    """

    from spikeinterface.sortingcomponents.motion import interpolate_motion

    detect_kwargs, select_kwargs, localize_peaks_kwargs, estimate_motion_kwargs = _update_motion_kwargs(
        preset, detect_kwargs, select_kwargs, localize_peaks_kwargs, estimate_motion_kwargs
    )
    interpolate_motion_kwargs = _update_interpolation_kwargs(preset, interpolate_motion_kwargs)

    motion, motion_info = compute_motion(
        recording,
        preset=preset,
        folder=folder,
        overwrite=overwrite,
        detect_kwargs=detect_kwargs,
        select_kwargs=select_kwargs,
        localize_peaks_kwargs=localize_peaks_kwargs,
        estimate_motion_kwargs=estimate_motion_kwargs,
        output_motion_info=True,
        **job_kwargs,
    )

    recording_corrected = interpolate_motion(recording, motion, **interpolate_motion_kwargs)

    if not output_motion and not output_motion_info:
        return recording_corrected

    out = (recording_corrected,)
    if output_motion:
        out += (motion,)
    if output_motion_info:
        out += (motion_info,)
    return out


def save_motion_info(motion_info, folder, overwrite=False):
    folder = Path(folder)
    if folder.is_dir():
        if not overwrite:
            raise FileExistsError(f"Folder {folder} already exists. Use `overwrite=True` to overwrite.")
        else:
            shutil.rmtree(folder)
    folder.mkdir(exist_ok=True, parents=True)

    (folder / "parameters.json").write_text(
        json.dumps(motion_info["parameters"], indent=4, cls=SIJsonEncoder), encoding="utf8"
    )
    (folder / "run_times.json").write_text(json.dumps(motion_info["run_times"], indent=4), encoding="utf8")

    np.save(folder / "peaks.npy", motion_info["peaks"])
    np.save(folder / "peak_locations.npy", motion_info["peak_locations"])

    motion = motion_info["motion"]
    if motion is not None:
        motion.save(folder / "motion")


def load_motion_info(folder):
    from spikeinterface.core.motion import Motion

    folder = Path(folder)

    motion_info = {}

    with open(folder / "parameters.json") as f:
        motion_info["parameters"] = json.load(f)

    with open(folder / "run_times.json") as f:
        motion_info["run_times"] = json.load(f)

    array_names = ("peaks", "peak_locations")
    for name in array_names:
        if (folder / f"{name}.npy").exists():
            motion_info[name] = np.load(folder / f"{name}.npy")
        else:
            motion_info[name] = None

    if (folder / "motion").is_dir():
        motion = Motion.load(folder / "motion")
    else:

        is_legacy_format = True
        required_files = ["spatial_bins.npy", "temporal_bins.npy", "motion.npy"]
        for required_file in required_files:
            if not (folder / required_file).is_file():
                is_legacy_format = False

        if is_legacy_format:

            warnings.warn("Trying to load Motion from the legacy format")

            spatial_bins_um = np.load(folder / "spatial_bins.npy")
            temporal_bins_s = [np.load(folder / "temporal_bins.npy")]
            displacement = [np.load(folder / "motion.npy")]

            motion = Motion(displacement=displacement, temporal_bins_s=temporal_bins_s, spatial_bins_um=spatial_bins_um)

        else:

            warnings.warn("No `motion` object in folder. `motion_info` is loaded with `motion` equal to `None`.")

            motion = None

    motion_info["motion"] = motion
    return motion_info


#########################################
# Docstrings
#########################################


_doc_presets = "\n"
for k, v in motion_options_preset.items():
    if k == "":
        continue
    doc = v["doc"]
    _doc_presets = _doc_presets + f"      * {k}: {doc}\n"

_common_motion_parameters = """Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to be transformed
    preset : str, default: "nonrigid_accurate"
        The preset name
    folder : Path str or None, default: None
        If not None then intermediate motion info are saved into a folder
    overwrite : bool, default: False
        If True and folder is given, overwrite the folder if it already exists
    detect_kwargs : dict
        Optional parameters to overwrite the ones in the preset for "detect" step.
    select_kwargs : dict
        If not None, optional parameters to overwrite the ones in the preset for "select" step.
        If None, the "select" step is skipped.
    localize_peaks_kwargs : dict
        Optional parameters to overwrite the ones in the preset for "localize" step.
    estimate_motion_kwargs : dict
        Optional parameters to overwrite the ones in the preset for "estimate_motion" step.
    interpolate_motion_kwargs : dict
        Optional parameters to overwrite the ones in the preset for "detect" step.
    output_motion_info : bool, default: False
        If True, then the function returns a `motion_info` dictionary that contains variables
        to check intermediate steps (motion_histogram, non_rigid_windows, pairwise_displacement)
        This dictionary is the same when reloaded from the folder."""


correct_motion.__doc__ = correct_motion.__doc__.format(_doc_presets, _common_motion_parameters, _shared_job_kwargs_doc)
compute_motion.__doc__ = compute_motion.__doc__.format(_doc_presets, _common_motion_parameters)
