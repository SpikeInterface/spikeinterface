from pathlib import Path
import os
from typing import Union
import shutil
import sys
import json

import scipy.io

from ..basesorter import BaseSorter
from ..utils import ShellScript

from spikeinterface.extractors import WaveClusSortingExtractor
from spikeinterface.extractors import WaveClusSnippetsExtractor
PathType = Union[str, Path]


def check_if_installed(waveclus_path: Union[str, None]):
    if waveclus_path is None:
        return False
    assert isinstance(waveclus_path, str)

    if waveclus_path.startswith('"'):
        waveclus_path = waveclus_path[1:-1]
    waveclus_path = str(Path(waveclus_path).absolute())

    if (Path(waveclus_path) / 'wave_clus.m').is_file():
        return True
    else:
        return False


class WaveClusSnippetsSorter(BaseSorter):
    """WaveClus Sorter object."""

    sorter_name: str = 'waveclus_snippets'
    compiled_name: str = 'waveclus_snippets_compiled'
    waveclus_path: Union[str, None] = os.getenv('WAVECLUS_PATH', None)
    requires_locations = False

    _default_params = {
        'feature_type': 'wav',
        'scales': 4,
        'min_clus': 20,
        'maxtemp': 0.251,
        'template_sdnum': 3,
        'mintemp': 0,
        'stdmax': 50,
        'max_spk': 40000,
        'keep_good_only': True,
        'chunk_memory': '500M'
    }

    _params_description = {
        'feature_type': "wav (for wavelets) or pca, type of feature extraction applied to the spikes",
        'scales': "Levels of the wavelet decomposition used as features",
        'min_clus': "Minimum increase of cluster sizes used by the peak selection on the temperature map",
        'maxtemp': "Maximum temperature calculated by the SPC method",
        'template_sdnum': "Maximum distance (in total variance of the cluster) from the mean waveform to force a "
                          "spike into a cluster",
        'mintemp': "Minimum temperature calculated by the SPC algorithm",
        'stdmax': "The events with a value over this number of noise standard deviations will be discarded",
        'max_spk': "Maximum number of spikes used by the SPC algorithm",
        'keep_good_only': "If True only 'good' units are returned",
        'chunk_memory': 'Chunk size in Mb to write h5 file (default 500Mb)'
    }

    sorter_description = """Wave Clus combines a wavelet-based feature extraction and paramagnetic clustering with a
    template-matching approach. It is mainly designed for monotrodes and low-channel count probes.
    For more information see https://doi.org/10.1152/jn.00339.2018"""

    installation_mesg = """\nTo use WaveClus run:\n
        >>> git clone https://github.com/csn-le/wave_clus
    and provide the installation path by setting the WAVECLUS_PATH
    environment variables or using WaveClusSorter.set_waveclus_path().\n\n

    More information on WaveClus at:
        https://github.com/csn-le/wave_clus/wiki
    """

    @classmethod
    def is_installed(cls):
        if cls.check_compiled():
            return True
        return check_if_installed(cls.waveclus_path)

    @classmethod
    def get_sorter_version(cls):
        if cls.check_compiled():
            return 'compiled'
        p = os.getenv('WAVECLUS_PATH', None)
        if p is None:
            return 'unknown'
        else:
            with open(str(Path(p) / 'version.txt'), mode='r', encoding='utf8') as f:
                version = f.readline()
        return version

    @classmethod
    def set_waveclus_path(cls, waveclus_path: PathType):
        waveclus_path = str(Path(waveclus_path).absolute())
        WaveClusSnippetsSorter.waveclus_path = waveclus_path
        try:
            print("Setting WAVECLUS_PATH environment variable for subprocess calls to:", waveclus_path)
            os.environ["WAVECLUS_PATH"] = waveclus_path
        except Exception as e:
            print("Could not set WAVECLUS_PATH environment variable:", e)

    @classmethod
    def _check_apply_filter_in_params(cls, params):
        return False

    @classmethod
    def _setup_recording(cls, snippets, sorter_output_folder, params, verbose):
        # Generate mat files in the dataset directory

        WaveClusSnippetsExtractor.write_snippets(snippets, sorter_output_folder / 'results_spikes.mat')

        if verbose:
            num_snippets = snippets.get_total_snippets()
            num_channels = snippets.get_num_channels()
            print('Num. channels = {}, Num. snippets = {}'.format(
                num_channels, num_snippets))

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        sorter_output_folder = sorter_output_folder.absolute()

        cls._generate_par_file(params, sorter_output_folder)
        if verbose:
            print(f'Running waveclus in {sorter_output_folder}...')

        if cls.check_compiled():
            shell_cmd = f'''
                #!/bin/bash
                {cls.compiled_name} {sorter_output_folder}
            '''
        else:
            source_dir = Path(__file__).parent
            shutil.copy(str(source_dir / f'waveclus_snippets_master.m'), str(sorter_output_folder))

            sorter_path = Path(cls.waveclus_path).absolute()
            if 'win' in sys.platform and sys.platform != 'darwin':
                disk_move = str(sorter_output_folder.absolute())[:2]
                shell_cmd = f'''
                    {disk_move}
                    cd {sorter_output_folder}
                    matlab -nosplash -wait -log -r "waveclus_snippets_master('{sorter_output_folder}', '{sorter_path}')"
                '''
            else:
                shell_cmd = f'''
                    #!/bin/bash
                    cd "{sorter_output_folder}"
                    matlab -nosplash -nodisplay -log -r "waveclus_snippets_master('{sorter_output_folder}', '{sorter_path}')"
                '''
        shell_cmd = ShellScript(shell_cmd, script_path=sorter_output_folder / f'run_{cls.sorter_name}',
                                log_path=sorter_output_folder / f'{cls.sorter_name}.log', verbose=verbose)
        shell_cmd.start()
        retcode = shell_cmd.wait()

        if retcode != 0:
            raise Exception('waveclus returned a non-zero exit code')

        result_fname = sorter_output_folder / 'times_results.mat'
        if not result_fname.is_file():
            raise Exception(f'Result file does not exist: {result_fname}')

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        sorter_output_folder = Path(sorter_output_folder)
        result_fname = str(sorter_output_folder / 'times_results.mat')
        if (sorter_output_folder.parent / 'spikeinterface_params.json').is_file():
            params_file = sorter_output_folder.parent / 'spikeinterface_params.json'
        else:
            # back-compatibility
            params_file = sorter_output_folder / 'spikeinterface_params.json'
        with params_file.open('r') as f:
            sorter_params = json.load(f)['sorter_params']
        keep_good_only = sorter_params.get('keep_good_only', True)
        sorting = WaveClusSortingExtractor(
            file_path=result_fname, keep_good_only=keep_good_only)
        return sorting

    @staticmethod
    def _generate_par_file(params, sorter_output_folder):
        """
        This function generates parameters data for waveclus and saves as `par_input.mat`

        Loading example in Matlab (shouldn't be assigned to a variable):
        >> load('/sorter_output_folder/par_input.mat');

        Parameters
        ----------
        params: dict
            Custom parameters dictionary for waveclus
        sorter_output_folder: pathlib.Path
            Path object to save `par_input.mat`
        """
        p = params.copy()

        par_renames = {'feature_type': 'features'}
        par_input = {}
        for key, value in p.items():
            if type(value) == bool:
                value = '{}'.format(value).lower()
            if key in par_renames:
                key = par_renames[key]
            par_input[key] = value

        # Converting integer values into float
        # matlab interprets numerical fields as double by default
        for k, v in par_input.items():
            if isinstance(v, int):
                par_input[k] = float(v)

        par_input = {'par_input': par_input}
        scipy.io.savemat(str(sorter_output_folder / 'par_input.mat'), par_input)
