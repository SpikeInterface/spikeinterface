import time
from pathlib import Path

import numpy as np
import json
import copy

from spikeinterface.core import get_noise_levels, fix_job_kwargs


motion_options_preset = {
    "rigid_fast": {
        "detect_kwargs": {},
        "select_kwargs": None,
        "localize_peaks_kwargs": {},
        "estimate_motion_kwargs": {},
        "correct_motion_kwargs": {},
    },
    "kilosort_like": {
        "detect_kwargs": dict(
            method="localy_exclusive",
            peak_sign="neg",
            detect_threshold=8.0,
            exclude_sweep_ms=0.1,
            local_radius_um=50,
        ),
        "select_kwargs": None,
        "localize_peaks_kwargs": dict(
            method="grid_convolution",
            local_radius_um=50.0,
            upsampling_um=5.0,
            sigma_um=np.linspace(10, 50.0, 5),
            sigma_ms=0.25,
            margin_um=50.0,
            prototype=None,
            percentile=10.0,
        ),
        "estimate_motion_kwargs": {},
        "correct_motion_kwargs": {},
    },
    "": {
        "detect_kwargs": {},
        "select_kwargs": None,
        "localize_peaks_kwargs": {},
        "estimate_motion_kwargs": {},
        "correct_motion_kwargs": {},
    },
}


def correct_motion(
    recording,
    preset="",
    folder=None,
    detect_kwargs={},
    select_kwargs=None,
    localize_peaks_kwargs={},
    estimate_motion_kwargs={},
    correct_motion_kwargs={},
    output_extra_check=False,
    **job_kwargs,
):
    """
    Top level function that estimate and interpolate the motion for a recording.

    This function have some intermediate steps that should be all controlled one by one carfully:
      * detect peaks
      * optionaly sample some peaks to speed up the localization
      * localize peaks
      * estimate the motion vector
      * create and return a `InterpolateMotionRecording` recording object

    Even this function is convinient to begin with, we highly recommend to run all step manually and
    separatly to accuratly check then.

    Optionaly this function create a folder with files and figures ready to check.

    This function depend on several modular components of :py:mod:`spikeinterface.sortingcomponents`

    if select_kwargs is None then all peak are used for localized.

    Parameters of steps are handled in a separate dict. For more information please check doc of the following
    functions:
      * :py:func:`~spikeinterface.sortingcomponents.peak_detection.detect_peaks'
      * :py:func:`~spikeinterface.sortingcomponents.peak_selection.select_peaks'
      * :py:func:`~spikeinterface.sortingcomponents.peak_localization.localize_peaks'
      * :py:func:`~spikeinterface.sortingcomponents.motion_estimation.estimate_motion'
      * :py:func:`~spikeinterface.sortingcomponents.motion_correction.CorrectMotionRecording'



    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed

    output_extra_check: bool
        If True then return an extra dict that contains variables
        to check intermediate steps (motion_histogram, non_rigid_windows, pairwise_displacement)

    Returns
    -------
    recording_corrected: Recording
        The motion corrected recording
    extra_check: dict
        Optional output if `output_extra_check=True`




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
    print(params)
    detect_kwargs = dict(params["detect_kwargs"], **detect_kwargs)
    if params["select_kwargs"] is None:
        select_kwargs = None
    else:
        select_kwargs = dict(params["select_kwargs"], **select_kwargs)
    localize_peaks_kwargs = dict(params["localize_peaks_kwargs"], **localize_peaks_kwargs)
    estimate_motion_kwargs = dict(params["estimate_motion_kwargs"], **estimate_motion_kwargs)
    correct_motion_kwargs = dict(params["correct_motion_kwargs"], **correct_motion_kwargs)

    if output_extra_check:
        extra_check = {}
    else:
        extra_check = None

    job_kwargs = fix_job_kwargs(job_kwargs)

    noise_levels = get_noise_levels(recording, return_scaled=False)

    if select_kwargs is None:
        # maybe do this directly in the folderwhen not None
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
        print(pipeline_nodes)
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
        t3 = t2 = t1 = time.perf_counter()
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

    motion, temporal_bins, spatial_bins = estimate_motion(recording, peaks, peak_locations, **estimate_motion_kwargs)
    t4 = time.perf_counter()

    recording_corrected = InterpolateMotionRecording(
        recording, motion, temporal_bins, spatial_bins, **correct_motion_kwargs
    )

    run_times = dict(
        detect_peaks=t1 - t0,
        select_peaks=t2 - t1,
        localize_peaks=t3 - t2,
        estimate_motion=t4 - t3,
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
            correct_motion_kwargs=correct_motion_kwargs,
            job_kwargs=job_kwargs,
        )
        (folder / "parameters.json").write_text(json.dumps(parameters, indent=4), encoding="utf8")
        (folder / "run_times.json").write_text(json.dumps(run_times, indent=4), encoding="utf8")

        np.save(folder / "peaks.npy", peaks)
        np.save(folder / "peak_locations.npy", peak_locations)
        np.save(folder / "temporal_bins.npy", temporal_bins)
        np.save(folder / "peak_locations.npy", peak_locations)
        if spatial_bins is not None:
            np.save(folder / "spatial_bins.npy", spatial_bins)

    if output_extra_check:
        extra_check = dict(
            parameters=parameters,
            run_times=run_times,
            peaks=peaks,
            peak_locations=peak_locations,
            spatial_bins=None,
        )
        return recording_corrected, extra_check
    else:
        return recording_corrected
