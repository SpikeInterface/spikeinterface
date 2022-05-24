from pathlib import Path
import json
import sys

import numpy as np
import scipy.io

from .utils import ShellScript
from spikeinterface.extractors import KiloSortSortingExtractor, BinaryRecordingExtractor


class KilosortBase:
    """
    Shared class for all kilosort implementation:
        * generate_channel_map_file
        * _run_from_folder
        * get_result_from_folder
    """

    @staticmethod
    def _generate_channel_map_file(recording, output_folder):
        """
        This function generates channel map data for kilosort and saves as `chanMap.mat`

        Loading example in Matlab (shouldn't be assigned to a variable):
        >> load('/path/to/output_folder/chanMap.mat');

        Parameters
        ----------
            recording: BaseRecording
                The recording to generate the channel map file
            output_folder: pathlib.Path
                Path object to save `chanMap.mat` file
        """
        # prepare electrode positions for this group (only one group, the split is done in basesorter)
        groups = [1] * recording.get_num_channels()
        positions = np.array(recording.get_channel_locations())
        if positions.shape[1] != 2:
            raise RuntimeError("3D 'location' are not supported. Set 2D locations instead")

        nchan = recording.get_num_channels()
        xcoords = [p[0] for p in positions],
        ycoords = [p[1] for p in positions],
        kcoords = groups,

        channel_map = {}
        channel_map['connected'] = np.full((nchan, 1), True)
        channel_map['chanMap0ind'] = np.arange(nchan)
        channel_map['chanMap'] = channel_map['chanMap0ind'] + 1

        channel_map['xcoords'] = np.array(xcoords).astype(float)
        channel_map['ycoords'] = np.array(ycoords).astype(float)
        channel_map['kcoords'] = np.array(kcoords).astype(float)

        sample_rate = recording.get_sampling_frequency()
        channel_map['fs'] = float(sample_rate)
        scipy.io.savemat(str(output_folder / 'chanMap.mat'), channel_map)

    @staticmethod
    def _write_recording(recording, output_folder, params, verbose):
        # save binary file
        BinaryRecordingExtractor.write_recording(recording, file_paths=output_folder / 'recording.dat',
                                                 dtype='int16', total_memory=params["total_memory"],
                                                 n_jobs=params["n_jobs_bin"], verbose=False, progress_bar=verbose)

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
