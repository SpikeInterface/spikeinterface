import copy
from pathlib import Path
import os
import sys
from typing import Union
import shutil
import numpy as np

from ..basesorter import BaseSorter
from ..kilosortbase import KilosortBase
from ..utils import ShellScript, get_git_commit

from spikeinterface.extractors import BinaryRecordingExtractor, KiloSortSortingExtractor


def check_if_installed(kilosort_path: Union[str, None]):
    if kilosort_path is None:
        return False
    assert isinstance(kilosort_path, str)

    if kilosort_path.startswith('"'):
        kilosort_path = kilosort_path[1:-1]
    kilosort_path = str(Path(kilosort_path).absolute())

    if (Path(kilosort_path) / 'preprocessData.m').is_file():
        return True
    else:
        return False


class KilosortSorter(KilosortBase, BaseSorter):
    """
    """

    sorter_name: str = 'kilosort'
    kilosort_path: Union[str, None] = os.getenv('KILOSORT_PATH', None)

    requires_locations = False
    docker_requires_gpu = True

    _default_params = {
        'detect_threshold': 6,
        'car': True,
        'useGPU': True,
        'freq_min': 300,
        'freq_max': 6000,
        'ntbuff': 64,
        'Nfilt': None,
        'NT': None,
        'total_memory': '500M',
        'n_jobs_bin': 1
    }

    _params_description = {
        'detect_threshold': "Threshold for spike detection",
        'car': "Enable or disable common reference",
        'useGPU': "Enable or disable GPU usage",
        'freq_min': "High-pass filter cutoff frequency",
        'freq_max': "Low-pass filter cutoff frequency",
        'ntbuff': "Samples of symmetrical buffer for whitening and spike detection",
        'Nfilt': "Number of clusters to use (if None it is automatically computed)",
        'NT': "Batch size (if None it is automatically computed)",
        'total_memory': "Chunk size in Mb for saving to binary format (default 500Mb)",
        'n_jobs_bin': "Number of jobs for saving to binary format (Default 1)"
    }

    sorter_description = """Kilosort is a GPU-accelerated and efficient template-matching spike sorter. 
    For more information see https://papers.nips.cc/paper/6326-fast-and-accurate-spike-sorting-of-high-channel-count-probes-with-kilosort"""

    installation_mesg = """\nTo use Kilosort run:\n
        >>> git clone https://github.com/cortex-lab/KiloSort
    and provide the installation path by setting the KILOSORT_PATH
    environment variables or using KilosortSorter.set_kilosort_path().\n\n

    More information on KiloSort at:
        https://github.com/cortex-lab/KiloSort
    """

    handle_multi_segment = False

    @classmethod
    def is_installed(cls):
        return check_if_installed(cls.kilosort_path)

    @classmethod
    def get_sorter_version(cls):
        commit = get_git_commit(os.getenv('KILOSORT_PATH', None))
        if commit is None:
            return 'unknown'
        else:
            return 'git-' + commit

    @classmethod
    def set_kilosort_path(cls, kilosort_path: str):
        kilosort_path = str(Path(kilosort_path).absolute())
        KilosortSorter.kilosort_path = kilosort_path
        try:
            print("Setting KILOSORT_PATH environment variable for subprocess calls to:", kilosort_path)
            os.environ["KILOSORT_PATH"] = kilosort_path
        except Exception as e:
            print("Could not set KILOSORT_PATH environment variable:", e)

    @classmethod
    def _check_params(cls, recording, output_folder, params):
        p = params
        nchan = recording.get_num_channels()
        if p['Nfilt'] is None:
            p['Nfilt'] = (nchan // 32) * 32 * 8
        else:
            p['Nfilt'] = p['Nfilt'] // 32 * 32
        if p['Nfilt'] == 0:
            p['Nfilt'] = nchan * 8
        if p['NT'] is None:
            p['NT'] = 64 * 1024 + p['ntbuff']
        else:
            p['NT'] = p['NT'] // 32 * 32  # make sure is multiple of 32
        return p

    @classmethod
    def _setup_recording(cls, recording, output_folder, params, verbose):
        p = params

        source_dir = Path(__file__).parent

        # prepare electrode positions for this group (only one group, the split is done in basesorter)
        groups = [1] * recording.get_num_channels()
        positions = np.array(recording.get_channel_locations())
        if positions.shape[1] != 2:
            raise RuntimeError("3D 'location' are not supported. Set 2D locations instead")

        # save binary file : handle only one segment
        input_file_path = output_folder / 'recording.dat'
        BinaryRecordingExtractor.write_recording(recording, file_paths=[input_file_path],
                                                 dtype='int16', total_memory=p["total_memory"],
                                                 n_jobs=p["n_jobs_bin"], verbose=False, progress_bar=verbose)

        # set up kilosort config files and run kilosort on data
        with (source_dir / 'kilosort_master.m').open('r') as f:
            kilosort_master_txt = f.read()
        with (source_dir / 'kilosort_config.m').open('r') as f:
            kilosort_config_txt = f.read()
        with (source_dir / 'kilosort_channelmap.m').open('r') as f:
            kilosort_channelmap_txt = f.read()

        nchan = recording.get_num_channels()

        if p['useGPU']:
            useGPU = 1
        else:
            useGPU = 0

        if p['car']:
            use_car = 1
        else:
            use_car = 0

        # make substitutions in txt files
        kilosort_master_txt = kilosort_master_txt.format(
            kilosort_path=str(Path(KilosortSorter.kilosort_path).absolute()),
            output_folder=str(output_folder.absolute()),
            channel_path=str((output_folder / 'kilosort_channelmap.m').absolute()),
            config_path=str((output_folder / 'kilosort_config.m').absolute()),
            useGPU=useGPU,
        )

        kilosort_config_txt = kilosort_config_txt.format(
            nchanTOT=recording.get_num_channels(),
            nchan=recording.get_num_channels(),
            sample_rate=recording.get_sampling_frequency(),
            dat_file=str((output_folder / 'recording.dat').absolute()),
            Nfilt=int(p['Nfilt']),
            ntbuff=int(p['ntbuff']),
            NT=int(p['NT']),
            kilo_thresh=p['detect_threshold'],
            use_car=use_car,
            freq_min=p['freq_min'],
            freq_max=p['freq_max']
        )

        kilosort_channelmap_txt = kilosort_channelmap_txt.format(
            nchan=recording.get_num_channels(),
            sample_rate=recording.get_sampling_frequency(),
            xcoords=[p[0] for p in positions],
            ycoords=[p[1] for p in positions],
            kcoords=groups
        )

        for fname, value in zip(['kilosort_master.m', 'kilosort_config.m',
                                 'kilosort_channelmap.m'],
                                [kilosort_master_txt, kilosort_config_txt,
                                 kilosort_channelmap_txt]):
            with (output_folder / fname).open('w') as f:
                f.writelines(value)

        shutil.copy(str(source_dir.parent / 'utils' / 'writeNPY.m'), str(output_folder))
        shutil.copy(str(source_dir.parent / 'utils' / 'constructNPYheader.m'), str(output_folder))
