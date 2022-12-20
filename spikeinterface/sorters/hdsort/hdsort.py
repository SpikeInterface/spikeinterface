from pathlib import Path
import os
from typing import Union
import shutil
import sys

import numpy as np
import scipy.io

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
    """HDSort Sorter object."""

    sorter_name: str = 'hdsort'
    compiled_name: str = 'hdsort_compiled'
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
        if cls.check_compiled():
            return True
        return check_if_installed(cls.hdsort_path)

    @classmethod
    def get_sorter_version(cls):
        if cls.check_compiled():
            return 'compiled'
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

    @staticmethod
    def _generate_configs_file(sorter_output_folder, params, file_name, file_format):
        P = {}

        # preprocess
        P['filter'] = 1.0 if params['filter'] else 0.0
        P['parfor'] = True if params['parfor'] else False
        P['hpf'] = float(params['freq_min'])
        P['lpf'] = float(params['freq_max'])

        # leg creationg
        P['legs'] = {
            'maxElPerGroup': float(params['max_el_per_group']),
            'minElPerGroup': float(params['min_el_per_group']),
            'addIfNearerThan': float(params['add_if_nearer_than']),  # always add direct neighbors
            'maxDistanceWithinGroup': float(params['max_distance_within_group'])
        }

        # spike detection
        P['spikeDetection'] = {
            'method': '-',
            'thr': float(params['detect_threshold'])
        }
        P['artefactDetection'] = {'use': 0.0}

        # pre-clustering
        P['noiseEstimation'] = {'minDistFromSpikes': 80.0}
        P['spikeAlignment'] = {
            'initAlignment': '-',
            'maxSpikes': 50000.0  # so many spikes will be clustered
        }
        P['featureExtraction'] = {'nDims': float(params['n_pc_dims'])}  # 6
        P['clustering'] = {
            'maxSpikes': 50000.0,  # dont align spikes you dont cluster...
            'meanShiftBandWidthFactor': 1.8
            # 'meanShiftBandWidth': sqrt(1.8*6)  # todo: check this!
        }

        # template matching
        P['botm'] = {
            'run': 0.0,
            'Tf': 75.0,
            'cutLeft': 20.0
        }
        P['spikeCutting'] = {
            'maxSpikes': 200000000000.0,  # Set this to basically inf
            'blockwise': False
        }
        P['templateEstimation'] = {
            'cutLeft': 10.0,
            'Tf': 55.0,
            'maxSpikes': 100.0
        }

        # merging
        P['mergeTemplates'] = {
            'merge': 1.0,
            'upsampleFactor': 3.0,
            'atCorrelation': .93,  # DONT SET THIS TOO LOW! USE OTHER ELECTRODES ON FULL FOOTPRINT TO MERGE
            'ifMaxRelDistSmallerPercent': 30.0

        }

        # configs
        sort_name = 'hdsort_output'
        cfgs = {}
        cfgs['rawFile'] = file_name
        cfgs['sortingName'] = sort_name
        cfgs['fileFormat'] = file_format
        cfgs['chunkSize'] = float(params['chunk_size'])
        cfgs['loopMode'] = params['loop_mode']

        data = {
            'P': P,
            **cfgs
        }

        scipy.io.savemat(str(sorter_output_folder / 'configsParams.mat'), data)

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
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
                                                            str(sorter_output_folder / 'recording.h5'),
                                                            chunk_memory=params["chunk_memory"])
            # ~ self.params['file_format'] = 'mea1k'
            file_format = 'mea1k'

        cls._generate_configs_file(sorter_output_folder, params, trace_file_name, file_format)

        # store sample rate in a file
        samplerate = recording.get_sampling_frequency()
        samplerate_fname = str(sorter_output_folder / 'samplerate.txt')
        with open(samplerate_fname, 'w') as f:
            f.write('{}'.format(samplerate))

        source_dir = Path(Path(__file__).parent)
        shutil.copy(str(source_dir / 'hdsort_master.m'), str(sorter_output_folder))

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        if cls.check_compiled():
            shell_cmd = f'''
                #!/bin/bash
                {cls.compiled_name} {sorter_output_folder}
            '''
        else:
            sorter_output_folder = sorter_output_folder.absolute()
            hdsort_path = Path(cls.hdsort_path).absolute()

            if "win" in sys.platform and sys.platform != 'darwin':
                disk_move = str(sorter_output_folder)[:2]
                shell_cmd = f'''
                            {disk_move}
                            cd {sorter_output_folder}
                            matlab -nosplash -wait -r "{cls.sorter_name}_master('{sorter_output_folder}', '{hdsort_path}')"
                        '''
            else:
                shell_cmd = f'''
                            #!/bin/bash
                            cd "{sorter_output_folder}"
                            matlab -nosplash -nodisplay -r "{cls.sorter_name}_master('{sorter_output_folder}', '{hdsort_path}')"
                        '''
        shell_script = ShellScript(shell_cmd, script_path=sorter_output_folder / f'run_{cls.sorter_name}',
                                   log_path=sorter_output_folder / f'{cls.sorter_name}.log', verbose=verbose)
        shell_script.start()
        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception('HDsort returned a non-zero exit code')

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        sorter_output_folder = Path(sorter_output_folder)
        sorting = HDSortSortingExtractor(file_path=str(sorter_output_folder / 'hdsort_output' /
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
