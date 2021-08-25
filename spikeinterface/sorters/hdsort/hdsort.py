from pathlib import Path
import os
from typing import Union
import numpy as np
import sys

from spikeinterface.core.core_tools import write_to_h5_dataset_format
from ..basesorter import BaseSorter
from ..utils import ShellScript

# from spikeinterface.extractors import MaxOneRecordingExtractor
from spikeinterface.extractors import HDSortSortingExtractor

PathType = Union[str, Path]


def check_if_installed(hdsort_path: Union[str, None]):
    if hdsort_path is None:
        return False
    assert isinstance(hdsort_path, str)

    if hdsort_path.startswith('"'):
        hdsort_path = hdsort_path[1:-1]
    hdsort_path = str(Path(hdsort_path).absolute())
    if (Path(hdsort_path) / '+hdsort').is_dir():
        return True
    else:
        return False


class HDSortSorter(BaseSorter):
    """
    """

    sorter_name: str = 'hdsort'
    hdsort_path: Union[str, None] = os.getenv('HDSORT_PATH', None)
    requires_locations = False
    _default_params = {
        'detect_threshold': 4.2,
        'detect_sign': -1,  # -1 - 1
        'filter': True,
        'parfor': True,
        'freq_min': 300,
        'freq_max': 7000,
        'max_el_per_group': 9,
        'min_el_per_group': 1,
        'add_if_nearer_than': 20,
        'max_distance_within_group': 52,
        'n_pc_dims': 6,
        'chunk_size': 500000,
        'loop_mode': 'local_parfor',
        'chunk_memory': '500M'
    }

    _params_description = {
        'detect_threshold': "Threshold for spike detection",
        'detect_sign': "Use -1 (negative) or 1 (positive) depending "
                       "on the sign of the spikes in the recording",
        'filter': "Enable or disable filter",
        'parfor': "If True, the Matlab parfor is used",
        'freq_min': "High-pass filter cutoff frequency",
        'freq_max': "Low-pass filter cutoff frequency",
        'max_el_per_group': "Maximum number of channels per electrode group",
        'min_el_per_group': "Minimum number of channels per electrode group",
        'add_if_nearer_than': "Minimum distance to add electrode to an electrode group",
        'max_distance_within_group': "Maximum distance within an electrode group",
        'n_pc_dims': "Number of principal components dimensions to perform initial clustering",
        'chunk_size': "Chunk size in number of frames for template-matching",
        'loop_mode': "Loop mode: 'loop', 'local_parfor', 'grid' (requires a grid architecture)",
        'chunk_memory': "Chunk size in Mb for saving to binary format (default 500Mb)",
    }

    sorter_description = """HDSort is a template-matching spike sorter designed for high density micro-electrode arrays. 
    For more information see https://doi.org/10.1152/jn.00803.2017"""

    installation_mesg = """\nTo use HDSort run:\n
        >>> git clone https://git.bsse.ethz.ch/hima_public/HDsort.git
    and provide the installation path by setting the HDSORT_PATH
    environment variables or using HDSortSorter.set_hdsort_path().\n\n

    More information on HDSort at:
        https://git.bsse.ethz.ch/hima_public/HDsort.git
    """

    handle_multi_segment = False

    @classmethod
    def is_installed(cls):
        return check_if_installed(cls.hdsort_path)

    @staticmethod
    def get_sorter_version():
        p = os.getenv('HDSORT_PATH', None)
        if p is None:
            return 'unknown'
        else:
            with open(str(Path(p) / 'version.txt'), mode='r', encoding='utf8') as f:
                version = f.readline()
        return version

    @staticmethod
    def set_hdsort_path(hdsort_path: PathType):
        HDSortSorter.hdsort_path = str(Path(hdsort_path).absolute())
        try:
            print("Setting HDSORT_PATH environment variable for subprocess calls to:", hdsort_path)
            os.environ["HDSORT_PATH"] = hdsort_path
        except Exception as e:
            print("Could not set HDSORT_PATH environment variable:", e)

    @classmethod
    def _check_apply_filter_in_params(cls, params):
        return params['filter']

    @classmethod
    def _setup_recording(cls, recording, output_folder, params, verbose):
        source_dir = Path(__file__).parent
        utils_path = source_dir.parent / 'utils'

        #  if isinstance(recording, MaxOneRecordingExtractor):
        if False:  # TODO
            # ~ self.params['file_name'] = str(Path(recording._file_path).absolute())
            trace_file_name = str(Path(recording._file_path).absolute())
            # ~ self.params['file_format'] =  'maxone'
            file_format = 'maxone'
            if verbose:
                print('Using MaxOne format directly')
        else:
            # Generate three files dataset in Mea1k format
            trace_file_name = cls.write_hdsort_input_format(recording,
                                                            str(output_folder / 'recording.h5'),
                                                            chunk_memory=params["chunk_memory"])
            # ~ self.params['file_format'] = 'mea1k'
            file_format = 'mea1k'

        p = params
        # ~ p['sort_name'] = 'hdsort_output'
        sort_name = 'hdsort_output'

        # read the template txt files
        with (source_dir / 'hdsort_master.m').open('r') as f:
            hdsort_master_txt = f.read()
        with (source_dir / 'hdsort_config.m').open('r') as f:
            hdsort_config_txt = f.read()

        # make substitutions in txt files
        hdsort_master_txt = hdsort_master_txt.format(
            hdsort_path=str(
                Path(HDSortSorter.hdsort_path).absolute()),
            utils_path=str(utils_path.absolute()),
            config_path=str((output_folder / 'hdsort_config.m').absolute()),
            # ~ file_name=p['file_name'],
            file_name=trace_file_name,
            # ~ file_format=p['file_format'],
            file_format=file_format,
            # ~ sort_name=p['sort_name'],
            sort_name=sort_name,
            chunk_size=p['chunk_size'],
            loop_mode=p['loop_mode']
        )

        if p['filter']:
            filter = 1
        else:
            filter = 0

        if p['parfor']:
            parfor = 'true'
        else:
            parfor = 'false'

        hdsort_config_txt = hdsort_config_txt.format(
            filter=filter,
            parfor=parfor,
            hpf=p['freq_min'],
            lpf=p['freq_max'],
            max_el_per_group=p['max_el_per_group'],
            min_el_per_group=p['min_el_per_group'],
            add_if_nearer_than=p['add_if_nearer_than'],
            max_distance_within_group=p['max_distance_within_group'],
            detect_threshold=p['detect_threshold'],
            n_pc_dims=p['n_pc_dims'],
        )

        for fname, txt in zip(['hdsort_master.m', 'hdsort_config.m'],
                              [hdsort_master_txt, hdsort_config_txt]):
            with (output_folder / fname).open('w') as f:
                f.write(txt)

        # store sample rate in a file
        samplerate = recording.get_sampling_frequency()
        samplerate_fname = str(output_folder / 'samplerate.txt')
        with open(samplerate_fname, 'w') as f:
            f.write('{}'.format(samplerate))

    @classmethod
    def _run_from_folder(cls, output_folder, params, verbose):
        tmpdir = output_folder

        if "win" in sys.platform and sys.platform != 'darwin':
            shell_cmd = '''
                        {disk_move}
                        cd {tmpdir}
                        matlab -nosplash -wait -r hdsort_master
                    '''.format(disk_move=str(output_folder)[:2], tmpdir=output_folder)
        else:
            shell_cmd = '''
                        #!/bin/bash
                        cd "{tmpdir}"
                        matlab -nosplash -nodisplay -r hdsort_master
                    '''.format(tmpdir=output_folder)

        shell_script = ShellScript(shell_cmd, script_path=output_folder / f'run_{cls.sorter_name}',
                                   log_path=output_folder / f'{cls.sorter_name}.log', verbose=verbose)
        shell_script.start()

        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception('HDsort returned a non-zero exit code')

    @classmethod
    def _get_result_from_folder(cls, output_folder):
        output_folder = Path(output_folder)
        sorting = HDSortSortingExtractor(file_path=str(output_folder / 'hdsort_output' /
                                                       'hdsort_output_results.mat'))
        return sorting

    @classmethod
    def write_hdsort_input_format(cls, recording, save_path, chunk_memory='500M'):
        try:
            import h5py
        except:
            raise Exception("To use HDSort, install h5py: pip install h5py")

        # check if already in write format
        write_file = True
        if hasattr(recording, '_file_path'):
            if Path(recording._file_path).suffix in ['.h5', '.hdf5']:
                with h5py.File(recording._file_path, 'r') as f:
                    keys = f.keys()
                    if "version" in keys and "ephys" in keys and "mapping" in keys and "frame_rate" in keys \
                            and "frame_numbers" in keys:
                        if "sig" in f["ephys"].keys():
                            write_file = False
                            trace_file_name = str(Path(recording._file_path).absolute())

        if write_file:
            save_path = Path(save_path)
            if save_path.suffix == '':
                save_path = Path(str(save_path) + '.h5')
            mapping_dtype = np.dtype([('electrode', np.int32), ('x', np.float64), ('y', np.float64),
                                      ('channel', np.int32)])

            locations = recording.get_property('location')
            assert locations is not None, "'location' property is needed to run HDSort"

            with h5py.File(save_path, 'w') as f:
                f.create_group('ephys')
                f.create_dataset('version', data=str(20161003))
                ephys = f['ephys']
                ephys.create_dataset('frame_rate', data=recording.get_sampling_frequency())
                ephys.create_dataset('frame_numbers', data=np.arange(recording.get_num_frames(segment_index=0)))
                # save mapping
                mapping = np.empty(recording.get_num_channels(), dtype=mapping_dtype)
                x = locations[:, 0]
                y = locations[:, 1]
                # channel should be from 0 to num_channel - 1
                for i, ch in enumerate(recording.get_channel_ids()):
                    mapping[i] = (ch, x[i], y[i], i)
                ephys.create_dataset('mapping', data=mapping)
                # save traces
                segment_index = 0
                write_to_h5_dataset_format(recording, dataset_path='/ephys/signal', segment_index=0,
                                           file_handle=f, time_axis=1, chunk_memory=chunk_memory)

            trace_file_name = str(save_path.absolute())

        return trace_file_name
