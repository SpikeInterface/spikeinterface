
import time
from pathlib import Path

import numpy as np
import json

from spikeinterface.core import get_noise_levels, fix_job_kwargs


motion_options_preset = {
    'rigid_simple': {
        'detect_kwargs': {},
        'select_kwargs': None,
        'localize_peaks_kwargs': {},
        'estimate_motion_kwargs': {},
        'correct_motion_kwargs': {},    
    },
    'kilosort_like': {
        'detect_kwargs': {},
        'select_kwargs': None,
        'localize_peaks_kwargs': {},
        'estimate_motion_kwargs': {},
        'correct_motion_kwargs': {},
    },


}


def estimate_and_correct_motion(recording,
                                preset='',
                                folder=None,  
                              detect_kwargs={},
                              select_kwargs=None,
                              localize_peaks_kwargs={},
                              estimate_motion_kwargs={},
                              correct_motion_kwargs={},
                              job_kwargs={},
                              
                              ):
    """
    Top level function that estimate and correct the motion for a recording.

    This function have some intermediate steps that should be all controlled one by one carfully:
      * detect peaks
      * optionaly sample some peaks to speed up the localization
      * localize peaks
      * estimate the motion vector
      * create and return a `CorrectMotionRecording` recording object

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
    
    Returns
    -------
    recording_corrected: Recording
        The motion corrected recording



    """

    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    from spikeinterface.sortingcomponents.peak_selection import select_peaks
    from spikeinterface.sortingcomponents.peak_localization import localize_peaks, localize_peak_methods
    from spikeinterface.sortingcomponents.motion_estimation import estimate_motion
    from spikeinterface.sortingcomponents.motion_correction import CorrectMotionRecording

    from spikeinterface.sortingcomponents.peak_pipeline import ExtractDenseWaveforms

    job_kwargs = fix_job_kwargs(job_kwargs)

    noise_levels = get_noise_levels(recording, return_scaled=False)

    if select_kwargs is None:
        # localize is done during detect_peaks()
        method = localize_peaks_kwargs.pop('method', 'center_of_mass')
        method_class = localize_peak_methods[method]
        node0 = ExtractDenseWaveforms(recording, name='waveforms',ms_before=0.1, ms_after=0.3)
        node1 = method_class(recording, parents='waveforms', return_ouput=True, **localize_peaks_kwargs)
        pipeline_nodes = [node0, node1]
    else:
        # lcalization is done after select_peaks()
        pipeline_nodes = None

    print(pipeline_nodes)

    t0 = time.perf_counter()
    peaks = detect_peaks(recording, noise_levels=noise_levels, pipeline_nodes=pipeline_nodes,
                         **detect_kwargs, **job_kwargs)
    t1 = time.perf_counter()

    if select_kwargs is None:
        # computed during detect_peaks()
        t3 = t2 = time.perf_counter()
        peaks, peak_locations = peaks

    else:
        # salect some peaks
        peaks = select_peaks(peaks, **select_kwargs, **job_kwargs)
        t2 = time.perf_counter()
        peak_locations = localize_peaks(recording, peaks,
                                        **localize_peaks_kwargs, **job_kwargs)
        t3 = time.perf_counter()
      

    motion, temporal_bins, spatial_bins = estimate_motion(recording, peaks, peak_locations, 
                                                          **estimate_motion_kwargs)
    t4 = time.perf_counter()

    recording_corrected = CorrectMotionRecording(recording, motion, temporal_bins, spatial_bins, 
                                                 **correct_motion_kwargs)

    run_times = dict(
        detect_peaks=t1 -t0,
        select_peaks=t2 - t1,
        localize_peaks=t3 - t2,
        estimate_motion= t4 - t3,
    )

    if folder is not None:
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)

        # params and run times
        parameters = dict(detect_kwargs=detect_kwargs, select_kwargs=select_kwargs, 
                          localize_peaks_kwargs=localize_peaks_kwargs, estimate_motion_kwargs=estimate_motion_kwargs,
                          correct_motion_kwargs=correct_motion_kwargs, job_kwargs=job_kwargs)
        (folder / 'parameters.json').write_text(json.dumps(parameters, indent=4), encoding='utf8')
        (folder / 'run_times.json').write_text(json.dumps(run_times, indent=4), encoding='utf8')


        np.save(folder / 'peaks.npy', peaks)
        np.save(folder / 'peak_locations.npy', peak_locations)
        np.save(folder / 'temporal_bins.npy', temporal_bins)
        np.save(folder / 'peak_locations.npy', peak_locations)
        if spatial_bins is not None:
            np.save(folder / 'spatial_bins.npy', spatial_bins)


    return recording_corrected