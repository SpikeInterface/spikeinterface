from pathlib import Path
import os
import sys
import numpy as np
from typing import Union
import shutil
import json

from ..basesorter import BaseSorter
from ..kilosortbase import KilosortBase
from ..utils import get_git_commit, ShellScript

from spikeinterface.extractors import BinaryRecordingExtractor, KiloSortSortingExtractor

PathType = Union[str, Path]


def check_if_installed(kilosort2_5_path: Union[str, None]):
    if kilosort2_5_path is None:
        return False
    assert isinstance(kilosort2_5_path, str)

    if kilosort2_5_path.startswith('"'):
        kilosort2_5_path = kilosort2_5_path[1:-1]
    kilosort2_5_path = str(Path(kilosort2_5_path).absolute())

    if (Path(kilosort2_5_path) / 'master_kilosort.m').is_file() or (
            Path(kilosort2_5_path) / 'main_kilosort.m').is_file():
        return True
    else:
        return False


class Kilosort2_5Sorter(KilosortBase, BaseSorter):
    """
    """

    sorter_name: str = 'kilosort2_5'
    kilosort2_5_path: Union[str, None] = os.getenv('KILOSORT2_5_PATH', None)
    requires_locations = False
    docker_requires_gpu = True

    _default_params = {
        'detect_threshold': 6,
        'projection_threshold': [10, 4],
        'preclust_threshold': 8,
        'car': True,
        'minFR': 0.1,
        'minfr_goodchannels': 0.1,
        'nblocks': 5,
        'sig': 20,
        'freq_min': 150,
        'sigmaMask': 30,
        'nPCs': 3,
        'ntbuff': 64,
        'nfilt_factor': 4,
        'NT': None,
        'keep_good_only': False,
        'total_memory': '500M',
        'n_jobs_bin': 1
    }

    _params_description = {
        'detect_threshold': "Threshold for spike detection",
        'projection_threshold': "Threshold on projections",
        'preclust_threshold': "Threshold crossings for pre-clustering (in PCA projection space)",
        'car': "Enable or disable common reference",
        'minFR': "Minimum spike rate (Hz), if a cluster falls below this for too long it gets removed",
        'minfr_goodchannels': "Minimum firing rate on a 'good' channel",
        'nblocks': "blocks for registration. 0 turns it off, 1 does rigid registration. Replaces 'datashift' option.",
        'sig': "spatial smoothness constant for registration",
        'freq_min': "High-pass filter cutoff frequency",
        'sigmaMask': "Spatial constant in um for computing residual variance of spike",
        'nPCs': "Number of PCA dimensions",
        'ntbuff': "Samples of symmetrical buffer for whitening and spike detection",
        'nfilt_factor': "Max number of clusters per good channel (even temporary ones) 4",
        'NT': "Batch size (if None it is automatically computed)",
        'keep_good_only': "If True only 'good' units are returned",
        'total_memory': "Chunk size in Mb for saving to binary format (default 500Mb)",
        'n_jobs_bin': "Number of jobs for saving to binary format (Default 1)"
    }

    sorter_description = """Kilosort2_5 is a GPU-accelerated and efficient template-matching spike sorter. On top of its 
    predecessor Kilosort, it implements a drift-correction strategy. Kilosort2.5 improves on Kilosort2 primarily in the 
    type of drift correction we use. Where Kilosort2 modified templates as a function of time/drift (a drift tracking 
    approach), Kilosort2.5 corrects the raw data directly via a sub-pixel registration process (a drift correction 
    approach). Kilosort2.5 has not been as broadly tested as Kilosort2, but is expected to work out of the box on 
    Neuropixels 1.0 and 2.0 probes, as well as other probes with vertical pitch <=40um. For other recording methods, 
    like tetrodes or single-channel recordings, you should test empirically if v2.5 or v2.0 works better for you (use 
    the "releases" on the github page to download older versions).
    For more information see https://github.com/MouseLand/Kilosort"""

    installation_mesg = """\nTo use Kilosort2.5 run:\n
        >>> git clone https://github.com/MouseLand/Kilosort
    and provide the installation path by setting the KILOSORT2_5_PATH
    environment variables or using Kilosort2_5Sorter.set_kilosort2_5_path().\n\n

    More information on Kilosort2.5 at:
        https://github.com/MouseLand/Kilosort
    """

    handle_multi_segment = False

    @classmethod
    def is_installed(cls):
        return check_if_installed(cls.kilosort2_5_path)

    @staticmethod
    def get_sorter_version():
        commit = get_git_commit(os.getenv('KILOSORT2_5_PATH', None))
        if commit is None:
            return 'unknown'
        else:
            return 'git-' + commit

    @staticmethod
    def set_kilosort2_5_path(kilosort2_5_path: PathType):
        kilosort2_5_path = str(Path(kilosort2_5_path).absolute())
        Kilosort2_5Sorter.kilosort2_5_path = kilosort2_5_path
        try:
            print("Setting KILOSORT2_5_PATH environment variable for subprocess calls to:", kilosort2_5_path)
            os.environ["KILOSORT2_5_PATH"] = kilosort2_5_path
        except Exception as e:
            print("Could not set KILOSORT2_5_PATH environment variable:", e)

    @classmethod
    def _check_params(cls, recording, output_folder, params):
        p = params
        if p['NT'] is None:
            p['NT'] = 64 * 1024 + p['ntbuff']
        else:
            p['NT'] = p['NT'] // 32 * 32  # make sure is multiple of 32
        return p

    @classmethod
    def _setup_recording(cls, recording, output_folder, params, verbose):
        p = params

        source_dir = Path(Path(__file__).parent)

        # prepare electrode positions for this group (only one group, the split is done in basesorter)
        groups = [1] * recording.get_num_channels()
        positions = np.array(recording.get_channel_locations())
        if positions.shape[1] != 2:
            raise RuntimeError("3D 'location' are not supported. Set 2D locations instead")

        # save binary file
        input_file_path = output_folder / 'recording.dat'
        BinaryRecordingExtractor.write_recording(recording, file_paths=[input_file_path],
                                                 dtype='int16', total_memory=p["total_memory"],
                                                 n_jobs=p["n_jobs_bin"], verbose=False, progress_bar=verbose)

        if p['car']:
            use_car = 1
        else:
            use_car = 0

        # read the template txt files
        with (source_dir / 'kilosort2_5_master.m').open('r') as f:
            kilosort2_5_master_txt = f.read()
        with (source_dir / 'kilosort2_5_config.m').open('r') as f:
            kilosort2_5_config_txt = f.read()
        with (source_dir / 'kilosort2_5_channelmap.m').open('r') as f:
            kilosort2_5_channelmap_txt = f.read()

        # make substitutions in txt files
        kilosort2_5_master_txt = kilosort2_5_master_txt.format(
            kilosort2_5_path=str(Path(Kilosort2_5Sorter.kilosort2_5_path).absolute()),
            output_folder=str(output_folder.absolute()),
            channel_path=str((output_folder / 'kilosort2_5_channelmap.m').absolute()),
            config_path=str((output_folder / 'kilosort2_5_config.m').absolute()),
        )

        kilosort2_5_config_txt = kilosort2_5_config_txt.format(
            nchan=recording.get_num_channels(),
            sample_rate=recording.get_sampling_frequency(),
            dat_file=str((output_folder / 'recording.dat').absolute()),
            nblocks=p['nblocks'],
            sig=p['sig'],
            projection_threshold=p['projection_threshold'],
            preclust_threshold=p['preclust_threshold'],
            minfr_goodchannels=p['minfr_goodchannels'],
            minFR=p['minFR'],
            freq_min=p['freq_min'],
            sigmaMask=p['sigmaMask'],
            kilo_thresh=p['detect_threshold'],
            use_car=use_car,
            nPCs=int(p['nPCs']),
            ntbuff=int(p['ntbuff']),
            nfilt_factor=int(p['nfilt_factor']),
            NT=int(p['NT'])
        )

        kilosort2_5_channelmap_txt = kilosort2_5_channelmap_txt.format(
            nchan=recording.get_num_channels(),
            sample_rate=recording.get_sampling_frequency(),
            xcoords=[p[0] for p in positions],
            ycoords=[p[1] for p in positions],
            kcoords=groups
        )

        for fname, txt in zip(['kilosort2_5_master.m', 'kilosort2_5_config.m',
                               'kilosort2_5_channelmap.m'],
                              [kilosort2_5_master_txt, kilosort2_5_config_txt,
                               kilosort2_5_channelmap_txt]):
            with (output_folder / fname).open('w') as f:
                f.write(txt)

        shutil.copy(str(source_dir.parent / 'utils' / 'writeNPY.m'), str(output_folder))
        shutil.copy(str(source_dir.parent / 'utils' / 'constructNPYheader.m'), str(output_folder))
