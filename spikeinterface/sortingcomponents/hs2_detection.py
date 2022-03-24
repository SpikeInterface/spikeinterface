"""Sorting components: Herding Spikes 2 detection."""

import herdingspikes
import numpy as np

_default_params = {
    # core params
    'left_cutout_time': 0.3,  # 0.2,
    'right_cutout_time': 1.8,  # 0.8,
    'detect_threshold': 20,  # 24, #15,

    # extra probe params
    'probe_masked_channels': [],
    'probe_inner_radius': 70,
    'probe_neighbor_radius': 90,
    'probe_event_length': 0.26,
    'probe_peak_jitter': 0.2,

    # extra detection params
    't_inc': 100000,
    'num_com_centers': 1,
    'maa': 12,
    'ahpthr': 11,
    'out_file_name': "HS2_detected",
    'decay_filtering': False,
    'save_all': False,
    'amp_evaluation_time': 0.4,  # 0.14,
    'spk_evaluation_time': 1.0,
    'to_localize': True,
    'generate_shapes': True
}

_params_description = {
    # core params
    'left_cutout_time': "Cutout size before peak (ms).",
    'right_cutout_time': "Cutout size after peak (ms).",
    'detect_threshold': "Detection threshold",

    # extra probe params
    'probe_masked_channels': "Masked channels",
    'probe_inner_radius': "Radius of area around probe channel for localization",
    'probe_neighbor_radius': "Radius of area around probe channel for neighbor classification.",
    'probe_event_length': "Duration of a spike event (ms)",
    'probe_peak_jitter': "Maximum peak misalignment for synchronous spike (ms)",

    # extra detection params
    't_inc': "Number of samples per chunk during detection.",
    'num_com_centers': "Number of centroids to average when localizing.",
    'maa': "Minimum summed spike amplitude for spike acceptance.",
    'ahpthr': "Requires magnitude of spike rebound for acceptance",
    'out_file_name': "File name for storage of unclustered detected spikes",
    'decay_filtering': "Experimental: Set to True at your risk",
    'save_all': "Save all working files after sorting (slow)",
    'amp_evaluation_time': "Amplitude evaluation time (ms)",
    'spk_evaluation_time': "Spike evaluation time (ms)",
    'to_localize': "Run spike localisation",
    
    # To be added in HS2
    'generate_shapes': "Run Shape generation"
}

def hs2_detect_spikes(recording, parameters=_default_params):
    for _default_key in _default_params:
        if _default_key not in parameters:
            parameters[_default_key] = _default_params[_default_key]
    
    p = parameters

    if p['to_localize'] == True:
        hs2_spike_dtype = [('sample_ind', 'int64'), ('channel_ind', 'int64'),
                        ('amplitude', 'float64'), ('segment_ind', 'int64'),
                        ('x', 'float64'), ('y', 'float64')]
    else:
        hs2_spike_dtype = [('sample_ind', 'int64'), ('channel_ind', 'int64'),
                ('amplitude', 'float64'), ('segment_ind', 'int64')]

    peaks = np.zeros(0, dtype=hs2_spike_dtype)

    total_segments = recording.get_num_segments()
    for segment_ind in range(total_segments):
        if total_segments == 1:
            recording_segment = recording
        else:
            recording_segment = recording._recording_segments[segment_ind].parent_segment.parent_extractor

        Probe = herdingspikes.probe.RecordingExtractor(
            recording_segment,
            masked_channels=p['probe_masked_channels'],
            inner_radius=p['probe_inner_radius'],
            neighbor_radius=p['probe_neighbor_radius'],
            event_length=p['probe_event_length'],
            peak_jitter=p['probe_peak_jitter'],
        )

        H = herdingspikes.HSDetection(
            Probe, file_directory_name='.',
            left_cutout_time=p['left_cutout_time'],
            right_cutout_time=p['right_cutout_time'],
            threshold=p['detect_threshold'],
            to_localize=p['to_localize'],
            num_com_centers=p['num_com_centers'],
            maa=p['maa'],
            ahpthr=p['ahpthr'],
            out_file_name=p['out_file_name'],
            decay_filtering=p['decay_filtering'],
            save_all=p['save_all'],
            amp_evaluation_time=p['amp_evaluation_time'],
            spk_evaluation_time=p['spk_evaluation_time']
        )

        H.DetectFromRaw(load=True, tInc=int(p['t_inc']))

        crt_peaks_num = H.spikes['t'].values.size
        crt_peaks = np.zeros(crt_peaks_num, dtype=hs2_spike_dtype)
        crt_peaks['sample_ind'] = H.spikes['t'].values
        crt_peaks['channel_ind'] = H.spikes['ch'].values
        crt_peaks['amplitude'] = H.spikes['Amplitude'].values
        crt_peaks['segment_ind'] = np.full(crt_peaks_num, segment_ind)

        if p['to_localize'] == True:
            crt_peaks['x'] = H.spikes['x'].values
            crt_peaks['y'] = H.spikes['y'].values

        peaks = np.append(peaks, crt_peaks)

    return peaks