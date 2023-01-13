import copy
from pathlib import Path
import os
import numpy as np
from numpy.lib.format import open_memmap
import sys

from spikeinterface.extractors import SpykingCircusSortingExtractor
from ..basesorter import BaseSorter
from ..utils import ShellScript

from probeinterface import write_prb


class SpykingcircusSorter(BaseSorter):
    """SpykingCircus Sorter object."""

    sorter_name = 'spykingcircus'
    requires_locations = False

    _default_params = {
        'detect_sign': -1,  # -1 - 1 - 0
        'adjacency_radius': 100,  # Channel neighborhood adjacency radius corresponding to geom file
        'detect_threshold': 6,  # Threshold for detection
        'template_width_ms': 3,  # Spyking circus parameter
        'filter': True,
        'merge_spikes': True,
        'auto_merge': 0.75,
        'num_workers': None,
        'whitening_max_elts': 1000,  # I believe it relates to subsampling and affects compute time
        'clustering_max_elts': 10000,  # I believe it relates to subsampling and affects compute time
    }

    _params_description = {
        'detect_sign': "Use -1 (negative), 1 (positive) or 0 (both) depending "
                       "on the sign of the spikes in the recording",
        'adjacency_radius': "Radius in um to build channel neighborhood",
        'detect_threshold': "Threshold for spike detection",
        'template_width_ms': "Template width in ms. Recommended values: 3 for in vivo - 5 for in vitro",
        'filter': "Enable or disable filter",
        'merge_spikes': "Enable or disable automatic mergind",
        'auto_merge': "Automatic merging threshold",
        'num_workers': "Number of workers (if None, half of the cpu number is used)",
        'whitening_max_elts': "Max number of events per electrode for whitening",
        'clustering_max_elts': "Max number of events per electrode for clustering",
    }

    sorter_description = """Spyking Circus uses a smart clustering and a greedy template matching approach for
    spike sorting. For more information see https://doi.org/10.7554/eLife.34518"""

    installation_mesg = """\nTo use Spyking-Circus run:\n
        >>> pip install spyking-circus

        Need MPICH working, for ubuntu do:
            sudo apt install libmpich-dev mpich

        More information on Spyking-Circus at:
            https://spyking-circus.readthedocs.io/en/latest/
    """

    handle_multi_segment = False

    @classmethod
    def is_installed(cls):
        try:
            import circus
            HAVE_SC = True
        except ImportError:
            HAVE_SC = False
        return HAVE_SC

    @staticmethod
    def get_sorter_version():
        import circus
        return circus.__version__

    @classmethod
    def _check_params(cls, recording, sorter_output_folder, params):
        # check and re dump params
        p = params
        if p['num_workers'] is None:
            p['num_workers'] = np.maximum(1, int(os.cpu_count() / 2))
        return p

    @classmethod
    def _check_apply_filter_in_params(cls, params):
        return params['filter']

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        p = params

        if p['detect_sign'] < 0:
            detect_sign = 'negative'
        elif p['detect_sign'] > 0:
            detect_sign = 'positive'
        else:
            detect_sign = 'both'
        if p['merge_spikes']:
            auto = p['auto_merge']
        else:
            auto = 0

        source_dir = Path(__file__).parent

        # save prb file
        # note: only one group here, the split is done in basesorter
        prb_file = sorter_output_folder / 'probe.prb'
        probegroup = recording.get_probegroup()
        write_prb(prb_file, probegroup,
                  total_nb_channels=recording.get_num_channels(),
                  radius=p['adjacency_radius'])

        # save binary file
        file_name = 'recording'
        # We should make this copy more efficient with chunks

        n_chan = recording.get_num_channels()
        n_frames = recording.get_num_frames(segment_index=0)
        chunk_size = 2 ** 24 // n_chan
        npy_file = str(sorter_output_folder / file_name) + '.npy'
        data_file = open_memmap(npy_file, shape=(n_frames, n_chan), dtype=np.float32, mode='w+')
        nb_chunks = n_frames // chunk_size
        for i in range(nb_chunks + 1):
            start_frame = i * chunk_size
            end_frame = min((i + 1) * chunk_size, n_frames)
            data = recording.get_traces(start_frame=start_frame, end_frame=end_frame).astype('float32')
            data_file[start_frame:end_frame, :] = data

        sample_rate = float(recording.get_sampling_frequency())

        # set up spykingcircus config file
        with (source_dir / 'config_default.params').open('r') as f:
            circus_config = f.readlines()
        circus_config = ''.join(circus_config).format(sample_rate, prb_file, p['template_width_ms'],
                                                      p['detect_threshold'], detect_sign, p['filter'],
                                                      p['whitening_max_elts'],
                                                      p['clustering_max_elts'], auto)
        with (sorter_output_folder / (file_name + '.params')).open('w') as f:
            f.writelines(circus_config)

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        sorter_name = cls.sorter_name

        num_workers = params['num_workers']

        if 'win' in sys.platform and sys.platform != 'darwin':
            shell_cmd = '''
                        spyking-circus {recording} -c {num_workers}
                    '''.format(recording=sorter_output_folder / 'recording.npy', num_workers=num_workers)
        else:
            shell_cmd = '''
                        #!/bin/bash
                        spyking-circus {recording} -c {num_workers}
                    '''.format(recording=sorter_output_folder / 'recording.npy', num_workers=num_workers)

        shell_script = ShellScript(shell_cmd, script_path=sorter_output_folder / f'run_{sorter_name}',
                                   log_path=sorter_output_folder / f'{sorter_name}.log', verbose=verbose)
        shell_script.start()

        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception('spykingcircus returned a non-zero exit code')

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        sorting = SpykingCircusSortingExtractor(folder_path=Path(sorter_output_folder) / 'recording')
        return sorting
