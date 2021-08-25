from pathlib import Path
import numpy as np

from spikeinterface.core import load_extractor
from spikeinterface.extractors import BinaryRecordingExtractor
from ..basesorter import BaseSorter

try:
    import pykilosort
    from pykilosort import Bunch, add_default_handler, run

    HAVE_PYKILOSORT = True
except ImportError:
    HAVE_PYKILOSORT = False


class PyKilosortSorter(BaseSorter):
    """
    """

    sorter_name = 'pykilosort'
    requires_locations = False
    docker_requires_gpu = True
    compatible_with_parallel = {'loky': True, 'multiprocessing': False, 'threading': False}

    _default_params = {
        "nfilt_factor": 8,
        "AUCsplit": 0.85,
        "nskip": 5
    }

    _params_description = {
    }

    sorter_description = """pykilosort is a port of kilosort to python"""

    installation_mesg = """\nTo use pykilosort:\n
       >>> pip install cupy
        >>> git clone https://github.com/MouseLand/pykilosort
        >>> cd pykilosort
        >>>python setup.py install    
    More info at https://github.com/MouseLand/pykilosort#installation
    """

    #
    handle_multi_segment = False

    @classmethod
    def is_installed(cls):
        return HAVE_PYKILOSORT

    @classmethod
    def get_sorter_version(cls):
        return pykilosort.__version__

    @classmethod
    def _check_params(cls, recording, output_folder, params):
        return params

    @classmethod
    def _setup_recording(cls, recording, output_folder, params, verbose):
        probe = recording.get_probe()

        # local copy
        recording.save(format='binary', folder=output_folder / 'bin_folder')

    @classmethod
    def _run_from_folder(cls, output_folder, params, verbose):
        recording = load_extractor(output_folder / 'spikeinterface_recording.json')

        assert isinstance(recording, BinaryRecordingExtractor)
        assert recording.get_num_segments() == 1
        dat_path = recording._kwargs['file_paths'][0]
        print('dat_path', dat_path)

        num_chans = recording.get_num_channels()
        locations = recording.get_channel_locations()
        print(locations)
        print(type(locations))

        # ks_probe is not probeinterface Probe at all
        ks_probe = Bunch()
        ks_probe.NchanTOT = num_chans
        ks_probe.chanMap = np.arange(num_chans)
        ks_probe.kcoords = np.ones(num_chans)
        ks_probe.xc = locations[:, 0]
        ks_probe.yc = locations[:, 1]

        run(
            dat_path,
            params=params,
            probe=ks_probe,
            dir_path=output_folder,
            n_channels=num_chans,
            dtype=recording.get_dtype(),
            sample_rate=recording.get_sampling_frequency(),
        )

    @classmethod
    def _get_result_from_folder(cls, output_folder):
        raise NotImplementedError
        # return sorting
