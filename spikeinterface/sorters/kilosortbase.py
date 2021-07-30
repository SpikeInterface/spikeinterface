from pathlib import Path
import sys
import numpy as np
import json

from .utils import get_git_commit, ShellScript

from spikeinterface.extractors import KiloSortSortingExtractor


class KilosortBase:
    """
    Shared class for all kilosort implementation:
      * _run_from_folder
      * get_result_from_folder
    """

    @classmethod
    def _run_from_folder(cls, output_folder, params, verbose):

        print('KilosortBase._run_from_folder', cls)

        if 'win' in sys.platform and sys.platform != 'darwin':
            disk_move = str(output_folder)[:2]
            shell_cmd = f'''
                        {disk_move}
                        cd {output_folder}
                        matlab -nosplash -wait -log -r {cls.sorter_name}_master
                    '''
        else:
            shell_cmd = f'''
                        #!/bin/bash
                        cd "{output_folder}"
                        matlab -nosplash -nodisplay -log -r {cls.sorter_name}_master
                    '''
        shell_script = ShellScript(shell_cmd, script_path=output_folder / f'run_{cls.sorter_name}',
                                   log_path=output_folder / f'{cls.sorter_name}.log', verbose=verbose)
        shell_script.start()
        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception(f'{cls.sorter_name} returned a non-zero exit code')

    @classmethod
    def _get_result_from_folder(cls, output_folder):
        output_folder = Path(output_folder)
        with (output_folder / 'spikeinterface_params.json').open('r') as f:
            sorter_params = json.load(f)['sorter_params']
        keep_good_only = sorter_params.get('keep_good_only', False)
        sorting = KiloSortSortingExtractor(folder_path=output_folder, keep_good_only=keep_good_only)
        return sorting
