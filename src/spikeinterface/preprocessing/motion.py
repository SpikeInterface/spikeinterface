import time
from pathlib import Path

import numpy as np
import json
import copy

from spikeinterface.core import get_noise_levels, fix_job_kwargs
from spikeinterface.core.job_tools import _shared_job_kwargs_doc
from spikeinterface.core.core_tools import SIJsonEncoder

motion_options_preset = {
    # This preset should be the most acccurate
    "nonrigid_accurate": {
        "doc": "method by Paninski lab (monopolar_triangulation + decentralized)",
        "detect_kwargs": dict(
            method="locally_exclusive",
            peak_sign="neg",
            detect_threshold=8.0,
            exclude_sweep_ms=0.1,
            local_radius_um=50,
        ),
        "select_kwargs": None,
        "localize_peaks_kwargs": dict(
            method="monopolar_triangulation",
            local_radius_um=75.0,
            max_distance_um=150.0,
            optimizer="minimize_with_log_penality",
            enforce_decrease=True,
            # feature="peak_voltage",
            feature="ptp",
        ),
        "estimate_motion_kwargs": dict(
            method="decentralized",
            direction="y",
            bin_duration_s=2.0,
            rigid=False,
            bin_um=5.0,
            margin_um=0.0,
            # win_shape="gaussian",
            # win_step_um=50.0,
            # win_sigma_um=150.0,
            win_shape="gaussian",
            win_step_um=100.0,
            win_sigma_um=200.0,
            histogram_depth_smooth_um=5.0,
            histogram_time_smooth_s=None,
            pairwise_displacement_method="conv",
            max_displacement_um=100.0,
            weight_scale="linear",
            error_sigma=0.2,
            conv_engine=None,
            torch_device=None,
            batch_size=1,
            corr_threshold=0.0,
            time_horizon_s=None,
            convergence_method="lsmr",
            soft_weights=False,
            normalized_xcorr=True,
            centered_xcorr=True,
            temporal_prior=True,
            spatial_prior=False,
            force_spatial_median_continuity=False,
            reference_displacement="median",
            reference_displacement_time_s=0,
            robust_regression_sigma=2,
            weight_with_amplitude=False,
        ),
        "interpolate_motion_kwargs": dict(
            direction=1,
            border_mode="remove_channels",
            spatial_interpolation_method="idw",
            num_closest=3,
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
            local_radius_um=50,
        ),
        "select_kwargs": None,
        "localize_peaks_kwargs": dict(
            method="center_of_mass",
            local_radius_um=75.0,
            feature="ptp",
        ),
        "estimate_motion_kwargs": dict(
            method="decentralized",
            bin_duration_s=10.0,
            rigid=True,
        ),
        "interpolate_motion_kwargs": dict(
            direction=1,
            border_mode="remove_channels",
            spatial_interpolation_method="idw",
            num_closest=3,
        ),
    },
    # This preset try to mimic kilosort2.5 motion estimator
    "kilosort_like": {
        "doc": "Mimic the drift correction of kilosrt (grid_convolution + iterative_template)",
        "detect_kwargs": dict(
            method="locally_exclusive",
            peak_sign="neg",
            detect_threshold=8.0,
            exclude_sweep_ms=0.1,
            local_radius_um=50,
        ),
        "select_kwargs": None,
        "localize_peaks_kwargs": dict(
            method="grid_convolution",
            local_radius_um=40.0,
            upsampling_um=5.0,
            sigma_um=np.linspace(5.0, 25.0, 5),
            sigma_ms=0.25,
            margin_um=30.0,
            prototype=None,
            percentile=10.0,
            sparsity_threshold=0.01,
        ),
        "estimate_motion_kwargs": dict(
            method="iterative_template",
            bin_duration_s=2.0,
            rigid=False,
            win_step_um=50.0,
            win_sigma_um=150.0,
            margin_um=0,
            win_shape="rect",
        ),
        "interpolate_motion_kwargs": dict(
            direction=1,
            border_mode="force_extrapolate",
            spatial_interpolation_method="kriging",
            sigma_um=[20.0, 30],
            p=1,
        ),
    },
    # empty preset
    "": {
        "detect_kwargs": {},
        "select_kwargs": None,
        "localize_peaks_kwargs": {},
        "estimate_motion_kwargs": {},
        "interpolate_motion_kwargs": {},
    },
}


def correct_motion(
    recording,
    preset="nonrigid_accurate",
    folder=None,
    output_motion_info=False,
    detect_kwargs={},
    select_kwargs=None,
    localize_peaks_kwargs={},
    estimate_motion_kwargs={},
    interpolate_motion_kwargs={},
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

    Even if this function is convinient, we recommend to run all step separately for fine tuning.

    Optionally, this function can create a folder with files and figures ready to check.

    This function depends on several modular components of :py:mod:`spikeinterface.sortingcomponents`.

    If select_kwargs is None then all peak are used for localized.

    The recording must be preprocessed (filter and denoised at least), and we recommend to not use whithening before motion
    estimation.

    Parameters for each step are handled as separate dictionaries.
    For more information please check the documentation of the following functions:
      * :py:func:`~spikeinterface.sortingcomponents.peak_detection.detect_peaks'
      * :py:func:`~spikeinterface.sortingcomponents.peak_selection.select_peaks'
      * :py:func:`~spikeinterface.sortingcomponents.peak_localization.localize_peaks'
      * :py:func:`~spikeinterface.sortingcomponents.motion_estimation.estimate_motion'
      * :py:func:`~spikeinterface.sortingcomponents.motion_interpolation.interpolate_motion'


    Possible presets: {}

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
    preset: str
        The preset name. Default "nonrigid_accurate".
    folder: Path str or None
        If not None then intermediate motion info are saved into a folder. Default None
    output_motion_info: bool
        If True, then the function returns a `motion_info` dictionary that contains variables
        to check intermediate steps (motion_histogram, non_rigid_windows, pairwise_displacement)
        This dictionary is the same when reloaded from the folder. Default False
    detect_kwargs: dict
        Optional parameters to overwrite the ones in the preset for "detect" step.
    select_kwargs: dict
        If not None, optional parameters to overwrite the ones in the preset for "select" step.
        If None, the "select" step is skipped.
    localize_peaks_kwargs: dict
        Optional parameters to overwrite the ones in the preset for "localize" step.
    estimate_motion_kwargs: dict
        Optional parameters to overwrite the ones in the preset for "estimate_motion" step.
    interpolate_motion_kwargs: dict
        Optional parameters to overwrite the ones in the preset for "detect" step.

    {}

    Returns
    -------
    recording_corrected: Recording
        The motion corrected recording
    motion_info: dict
        Optional output if `output_motion_info=True`
    """

    # local import are important because "sortingcomponents" is not important by default
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks, detect_peak_methods
    from spikeinterface.sortingcomponents.peak_selection import select_peaks
    from spikeinterface.sortingcomponents.peak_localization import localize_peaks, localize_peak_methods
    from spikeinterface.sortingcomponents.motion_estimation import estimate_motion
    from spikeinterface.sortingcomponents.motion_interpolation import InterpolateMotionRecording
    from spikeinterface.sortingcomponents.peak_pipeline import ExtractDenseWaveforms, run_node_pipeline

    # get preset params and update if necessary
    params = motion_options_preset[preset]
    detect_kwargs = dict(params["detect_kwargs"], **detect_kwargs)
    if params["select_kwargs"] is None:
        select_kwargs = None
    else:
        select_kwargs = dict(params["select_kwargs"], **select_kwargs)
    localize_peaks_kwargs = dict(params["localize_peaks_kwargs"], **localize_peaks_kwargs)
    estimate_motion_kwargs = dict(params["estimate_motion_kwargs"], **estimate_motion_kwargs)
    interpolate_motion_kwargs = dict(params["interpolate_motion_kwargs"], **interpolate_motion_kwargs)

    if output_motion_info:
        motion_info = {}
    else:
        motion_info = None

    job_kwargs = fix_job_kwargs(job_kwargs)

    noise_levels = get_noise_levels(recording, return_scaled=False)

    if select_kwargs is None:
        # maybe do this directly in the folder when not None
        gather_mode = "memory"

        # node detect
        method = detect_kwargs.pop("method", "locally_exclusive")
        method_class = detect_peak_methods[method]
        node0 = method_class(recording, **detect_kwargs)

        node1 = ExtractDenseWaveforms(recording, parents=[node0], ms_before=0.1, ms_after=0.3)

        # node nolcalize
        method = localize_peaks_kwargs.pop("method", "center_of_mass")
        method_class = localize_peak_methods[method]
        node2 = method_class(recording, parents=[node0, node1], return_output=True, **localize_peaks_kwargs)
        pipeline_nodes = [node0, node1, node2]
        t0 = time.perf_counter()
        peaks, peak_locations = run_node_pipeline(
            recording,
            pipeline_nodes,
            job_kwargs,
            job_name="detect and localize",
            gather_mode=gather_mode,
            squeeze_output=False,
            folder=None,
            names=None,
        )
        t1 = time.perf_counter()
        run_times = dict(
            detect_and_localize=t1 - t0,
        )
    else:
        # lcalization is done after select_peaks()
        pipeline_nodes = None

        t0 = time.perf_counter()
        peaks = detect_peaks(recording, noise_levels=noise_levels, pipeline_nodes=None, **detect_kwargs, **job_kwargs)
        t1 = time.perf_counter()
        # salect some peaks
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
    motion, temporal_bins, spatial_bins = estimate_motion(recording, peaks, peak_locations, **estimate_motion_kwargs)
    t1 = time.perf_counter()
    run_times["estimate_motion"] = t1 - t0

    recording_corrected = InterpolateMotionRecording(
        recording, motion, temporal_bins, spatial_bins, **interpolate_motion_kwargs
    )

    if folder is not None:
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)

        # params and run times
        parameters = dict(
            detect_kwargs=detect_kwargs,
            select_kwargs=select_kwargs,
            localize_peaks_kwargs=localize_peaks_kwargs,
            estimate_motion_kwargs=estimate_motion_kwargs,
            interpolate_motion_kwargs=interpolate_motion_kwargs,
            job_kwargs=job_kwargs,
            sampling_frequency=recording.sampling_frequency,
        )
        (folder / "parameters.json").write_text(json.dumps(parameters, indent=4, cls=SIJsonEncoder), encoding="utf8")
        (folder / "run_times.json").write_text(json.dumps(run_times, indent=4), encoding="utf8")
        if recording.check_if_json_serializable():
            recording.dump_to_json(folder / "recording.json")

        np.save(folder / "peaks.npy", peaks)
        np.save(folder / "peak_locations.npy", peak_locations)
        np.save(folder / "temporal_bins.npy", temporal_bins)
        np.save(folder / "motion.npy", motion)
        np.save(folder / "peak_locations.npy", peak_locations)
        if spatial_bins is not None:
            np.save(folder / "spatial_bins.npy", spatial_bins)

    if output_motion_info:
        motion_info = dict(
            parameters=parameters,
            run_times=run_times,
            peaks=peaks,
            peak_locations=peak_locations,
            temporal_bins=temporal_bins,
            spatial_bins=spatial_bins,
            motion=motion,
        )
        return recording_corrected, motion_info
    else:
        return recording_corrected


_doc_presets = "\n"
for k, v in motion_options_preset.items():
    if k == "":
        continue
    doc = v["doc"]
    _doc_presets = _doc_presets + f"      * {k}: {doc}\n"

correct_motion.__doc__ = correct_motion.__doc__.format(_doc_presets, _shared_job_kwargs_doc)


def load_motion_info(folder):
    folder = Path(folder)

    motion_info = {}

    with open(folder / "parameters.json") as f:
        motion_info["parameters"] = json.load(f)

    with open(folder / "run_times.json") as f:
        motion_info["run_times"] = json.load(f)

    array_names = ("peaks", "peak_locations", "temporal_bins", "spatial_bins", "motion")
    for name in array_names:
        if (folder / f"{name}.npy").exists():
            motion_info[name] = np.load(folder / f"{name}.npy")
        else:
            motion_info[name] = None

    return motion_info
