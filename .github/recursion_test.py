import tempfile
import os
import shutil
import platform

from pathlib import Path
from tempfile import tempdir
from tqdm import tqdm

from spikeinterface.core.datasets import download_dataset
from spikeinterface.extractors import SpikeGLXRecordingExtractor
from spikeinterface.core.segmentutils import concatenate_recordings
from spikeinterface.preprocessing import bandpass_filter
from spikeinterface.core.core_tools import write_binary_recording


def copy_folder_n_times(directory_path, n_times):
    parent_directory = directory_path.parent
    folder_with_copies = parent_directory / "folder_with_copies"
    folder_with_copies.mkdir(exist_ok=True, parents=True)
    for i in tqdm(range(n_times), desc="Copying folder"):
        copy_name = f"copy_{i}"
        copy_path = folder_with_copies / copy_name
        shutil.copytree(directory_path, copy_path)

    return folder_with_copies


if __name__ == "__main__":


    
    number_of_files = 750
    
    job_kwargs = {
        "chunk_duration": "1s",
        "n_jobs": -1,
        "mp_context": "spawn",
        "max_threads_per_process": 1,
    }

    number_of_preprocessing_steps = 20
    pre_processing_function = bandpass_filter

    with tempfile.TemporaryDirectory() as tmpdir:
        # Use the temporary directory for the duration of the program

        repo = "https://gin.g-node.org/NeuralEnsemble/ephy_testing_data"
        remote_path = "spikeglx/Noise4Sam_g0"
        local_folder = Path(tmpdir) / "a_fine_test_folder"
        local_path = download_dataset(repo=repo, remote_path=remote_path, local_folder=local_folder, unlock=True)

        folder_with_copies = copy_folder_n_times(local_path, n_times=number_of_files)
        print("Done copying files")
        # Display cumulative size of folder_with_copies
        print(f"Size of folder_with_copies: {sum(f.stat().st_size for f in folder_with_copies.glob('**/*') if f.is_file()) / 1e9} GB")
        
        spikeglx_folder_list = [path for path in folder_with_copies.iterdir() if path.is_dir()]
        stream_id = "imec0.ap"
        extractor = SpikeGLXRecordingExtractor
        extractor_list = [extractor(folder_path=path, stream_id=stream_id) for path in spikeglx_folder_list]
        num_frames = extractor_list[0].get_num_frames()
        extractor_list = [recording.select_segments(segment_indices=[0]) for recording in extractor_list]
        extractor_list = [recording.frame_slice(start_frame=0, end_frame=num_frames) for recording in extractor_list]
        concatenated_recording = concatenate_recordings(extractor_list)
        
        print(f"Folder {tmpdir}")
        print(f"{concatenated_recording}")
        
        pre_processed_recording = concatenated_recording
        for i in range(number_of_preprocessing_steps):
            pre_processed_recording = pre_processing_function(pre_processed_recording)

        binary_file_file_path = Path(tmpdir) / "a_fine_test.bin"
        cached_recording = write_binary_recording(
            pre_processed_recording, 
            file_paths=[binary_file_file_path], 
            dtype=concatenated_recording.get_dtype(), 
            verbose=True,
            **job_kwargs,
        )
        print(f"cached recording {cached_recording}")
