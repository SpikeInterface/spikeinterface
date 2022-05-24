from pathlib import Path
import os
from typing import Union
import sys
import shutil

from .kilosort3_config import generate_ops_file
from ..basesorter import BaseSorter
from ..kilosortbase import KilosortBase
from ..utils import get_git_commit, ShellScript

PathType = Union[str, Path]


def check_if_installed(kilosort3_path: Union[str, None]):
    if kilosort3_path is None:
        return False
    assert isinstance(kilosort3_path, str)

    if kilosort3_path.startswith('"'):
        kilosort3_path = kilosort3_path[1:-1]
    kilosort3_path = str(Path(kilosort3_path).absolute())

    if (Path(kilosort3_path) / 'main_kilosort3.m').is_file():
        return True
    else:
        return False


class Kilosort3Sorter(KilosortBase, BaseSorter):
    """Kilosort3 Sorter object."""

    sorter_name: str = 'kilosort3'
    compiled_name: str = 'ks3_compiled'
    kilosort3_path: Union[str, None] = os.getenv('KILOSORT3_PATH', None)
    requires_locations = False
    docker_requires_gpu = True

    _default_params = {
        'detect_threshold': 6,
        'projection_threshold': [9, 9],
        'preclust_threshold': 8,
        'car': True,
        'minFR': 0.2,
        'minfr_goodchannels': 0.2,
        'nblocks': 5,
        'sig': 20,
        'freq_min': 300,
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

    sorter_description = """Kilosort3 is a GPU-accelerated and efficient template-matching spike sorter. On top of its
    predecessor Kilosort, it implements a drift-correction strategy. Kilosort3 improves on Kilosort2 primarily in the
    type of drift correction we use. Where Kilosort2 modified templates as a function of time/drift (a drift tracking
    approach), Kilosort3 corrects the raw data directly via a sub-pixel registration process (a drift correction
    approach). Kilosort3 has not been as broadly tested as Kilosort2, but is expected to work out of the box on
    Neuropixels 1.0 and 2.0 probes, as well as other probes with vertical pitch <=40um. For other recording methods,
    like tetrodes or single-channel recordings, you should test empirically if v3 or v2.0 works better for you (use
    the "releases" on the github page to download older versions).
    For more information see https://github.com/MouseLand/Kilosort"""

    installation_mesg = """\nTo use Kilosort3 run:\n
        >>> git clone https://github.com/MouseLand/Kilosort
    and provide the installation path by setting the KILOSORT3_PATH
    environment variables or using Kilosort3Sorter.set_kilosort3_path().\n\n

    More information on Kilosort3 at:
        https://github.com/MouseLand/Kilosort
    """

    handle_multi_segment = False

    @classmethod
    def is_installed(cls):
        if cls.check_compiled():
            return True
        return check_if_installed(cls.kilosort3_path)

    @classmethod
    def get_sorter_version(cls):
        if cls.check_compiled():
            return 'compiled'
        commit = get_git_commit(os.getenv('KILOSORT3_PATH', None))
        if commit is None:
            return 'unknown'
        else:
            return 'git-' + commit

    @staticmethod
    def set_kilosort3_path(kilosort3_path: PathType):
        kilosort3_path = str(Path(kilosort3_path).absolute())
        Kilosort3Sorter.kilosort3_path = kilosort3_path
        try:
            print("Setting KILOSORT3_PATH environment variable for subprocess calls to:", kilosort3_path)
            os.environ["KILOSORT3_PATH"] = kilosort3_path
        except Exception as e:
            print("Could not set KILOSORT3_PATH environment variable:", e)

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

        cls._generate_channel_map_file(recording, output_folder)

        cls._write_recording(recording, output_folder, params, verbose)

        generate_ops_file(recording, params, output_folder)

        source_dir = Path(Path(__file__).parent)
        shutil.copy(str(source_dir / 'kilosort3_master.m'), str(output_folder))
        shutil.copy(str(source_dir.parent / 'utils' / 'writeNPY.m'), str(output_folder))
        shutil.copy(str(source_dir.parent / 'utils' / 'constructNPYheader.m'), str(output_folder))

    # TODO: This is a copy/adaptation from KilosortBase
    # If all versions of kilosort are changed according to this approach,
    # _run_from_folder should be moved to KilosortBase again
    @classmethod
    def _run_from_folder(cls, output_folder, params, verbose):

        output_folder = output_folder.absolute()
        if cls.check_compiled():
            shell_cmd = f'''
                #!/bin/bash
                ks3_compiled {output_folder}
            '''
        elif 'win' in sys.platform and sys.platform != 'darwin':
            kilosort3_path = Path(Kilosort3Sorter.kilosort3_path).absolute()
            disk_move = str(output_folder)[:2]
            shell_cmd = f'''
                        {disk_move}
                        cd {output_folder}
                        matlab -nosplash -wait -r "{cls.sorter_name}_master('{output_folder}', '{kilosort3_path}')"
                    '''
        else:
            kilosort3_path = Path(Kilosort3Sorter.kilosort3_path).absolute()
            shell_cmd = f'''
                        #!/bin/bash
                        cd "{output_folder}"
                        matlab -nosplash -nodisplay -r "{cls.sorter_name}_master('{output_folder}', '{kilosort3_path}')"
                    '''
        shell_script = ShellScript(shell_cmd, script_path=output_folder / f'run_{cls.sorter_name}',
                                   log_path=output_folder / f'{cls.sorter_name}.log', verbose=verbose)
        shell_script.start()
        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception(f'{cls.sorter_name} returned a non-zero exit code')
