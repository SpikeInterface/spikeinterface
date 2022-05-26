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
      * _generate_channel_map_file
      * _generate_ops_file
      * _run_from_folder
      * _get_result_from_folder
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
    def _generate_ops_file(cls, recording, params, output_folder):
        """
        This function generates ops (configs) data for kilosort and saves as `ops.mat`

        Loading example in Matlab (should be assigned to a variable called `ops`):
        >> ops = load('/output_folder/ops.mat');

        Parameters
        ----------
        recording: BaseRecording
            The recording to generate the channel map file
        params: dict
            Custom parameters dictionary for kilosort3
        output_folder: pathlib.Path
            Path object to save `ops.mat`
        """
        ops = {}

        nchan = recording.get_num_channels()
        ops['NchanTOT'] = nchan  # total number of channels (omit if already in chanMap file)
        ops['Nchan'] = nchan  # number of active channels (omit if already in chanMap file)

        ops['datatype'] = 'dat'  # binary ('dat', 'bin') or 'openEphys'
        ops['fbinary'] = str((output_folder / 'recording.dat').absolute())  # will be created for 'openEphys'
        ops['fproc'] = str((output_folder / 'temp_wh.dat').absolute())  # residual from RAM of preprocessed data
        ops['root'] = str(output_folder.absolute())  # 'openEphys' only: where raw files are
        ops['trange'] = [0, np.Inf] #  time range to sort
        ops['chanMap'] = str((output_folder / 'chanMap.mat').absolute())

        # sample rate
        ops['fs'] = recording.get_sampling_frequency()

        ops = cls._get_specific_options(ops, params)

        # Converting integer values into float
        # Kilosort interprets ops fields as double
        for k, v in ops.items():
            if isinstance(v, int):
                ops[k] = float(v)

        scipy.io.savemat(str(output_folder / 'ops.mat'), ops)

    @classmethod
    def _run_from_folder(cls, output_folder, params, verbose):
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

    @classmethod
    def _get_specific_options(cls, ops, params):
        return ops
