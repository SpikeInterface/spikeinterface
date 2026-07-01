from pathlib import Path
import spikeinterface.full as si
rec = si.read_openephys('/Users/christopherhalcrow/Work/Harry_Project/fast_curate_demo/analyzers/1742_2024-03-15_11-01-52_obj')

rec = rec.frame_slice(start_frame=0, end_frame=30_000)

preprocessing_dict = {
    'bandpass_filter': {'freq_min': 250},
    'common_reference': {'operator': 'median', 'reference': 'global'},
    'detect_and_remove_artifacts': {'recording_to_detect': 'pipeline[raw]'},
}

pipeline = si.PreprocessingPipeline(preprocessing_dict)
pp_rec = si.apply_preprocessing_pipeline(rec, pipeline)
pp_rec.dump("test_rec_raw_metadata/recording_bp.pickle")

loaded_rec = si.get_preprocessing_dict_from_file('test_rec_raw_metadata/recording_bp.pickle')

#loaded_rec
