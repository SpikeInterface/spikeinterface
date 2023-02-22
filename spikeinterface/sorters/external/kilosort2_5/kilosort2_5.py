from pathlib import Path
import os
from typing import Union

from ..basesorter import BaseSorter
from ..kilosortbase import KilosortBase
from ..utils import get_git_commit


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
    """Kilosort2.5 Sorter object."""

    sorter_name: str = 'kilosort2_5'
    compiled_name: str = 'ks2_5_compiled'
    kilosort2_5_path: Union[str, None] = os.getenv('KILOSORT2_5_PATH', None)
    requires_locations = False

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
        'do_correction': True,
        'wave_length': 61,
        'keep_good_only': False,
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
        "do_correction": "If True drift registration is applied",
        'NT': "Batch size (if None it is automatically computed)",
        'keep_good_only': "If True only 'good' units are returned",
        'wave_length': "size of the waveform extracted around each detected peak, (Default 61, maximum 81)",
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
        if cls.check_compiled():
            return True
        return check_if_installed(cls.kilosort2_5_path)

    @classmethod
    def get_sorter_version(cls):
        if cls.check_compiled():
            return 'compiled'
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
        if p['wave_length'] % 2 != 1:
            p['wave_length'] = p['wave_length'] + 1 # The wave_length must be odd
        if p['wave_length'] > 81:
            p['wave_length'] = 81 # The wave_length must be less than 81.
        return p

    @classmethod
    def _get_specific_options(cls, ops, params):
        """
        Adds specific options for Kilosort2_5 in the ops dict and returns the final dict

        Parameters
        ----------
        ops: dict
            options data
        params: dict
            Custom parameters dictionary for kilosort3

        Returns
        ----------
        ops: dict
            Final ops data
        """
        # frequency for high pass filtering (300)
        ops['fshigh'] = params['freq_min']

        # minimum firing rate on a "good" channel (0 to skip)
        ops['minfr_goodchannels'] = params['minfr_goodchannels']

        # number of blocks to use for nonrigid registration
        ops['nblocks'] = params['nblocks']

        # spatial smoothness constant for registration
        ops['sig'] = params['sig']

        projection_threshold = [float(pt) for pt in params['projection_threshold']]
        # threshold on projections (like in Kilosort1, can be different for last pass like [10 4])
        ops['Th'] = projection_threshold

        # how important is the amplitude penalty (like in Kilosort1, 0 means not used, 10 is average, 50 is a lot)
        ops['lam'] = 10.0

        # splitting a cluster at the end requires at least this much isolation for each sub-cluster (max = 1)
        ops['AUCsplit'] = 0.9

        # minimum spike rate (Hz), if a cluster falls below this for too long it gets removed
        ops['minFR'] = params['minFR']

        # number of samples to average over (annealed from first to second value)
        ops['momentum'] = [20.0, 400.0]

        # spatial constant in um for computing residual variance of spike
        ops['sigmaMask'] = params['sigmaMask']

        # threshold crossings for pre-clustering (in PCA projection space)
        ops['ThPre'] = params['preclust_threshold']

        ## danger, changing these settings can lead to fatal errors
        # options for determining PCs
        ops['spkTh'] = -params['detect_threshold']  # spike threshold in standard deviations (-6)
        ops['reorder'] = 1.0  # whether to reorder batches for drift correction.
        ops['nskip'] = 25.0  # how many batches to skip for determining spike PCs

        ops['GPU'] = 1.0  # has to be 1, no CPU version yet, sorry
        # ops['Nfilt'] = 1024 # max number of clusters
        ops['nfilt_factor'] = params['nfilt_factor']  # max number of clusters per good channel (even temporary ones)
        ops['ntbuff'] = params['ntbuff']  # samples of symmetrical buffer for whitening and spike detection
        ops['NT'] = params['NT']  # must be multiple of 32 + ntbuff. This is the batch size (try decreasing if out of memory).
        ops['whiteningRange'] = 32.0  # number of channels to use for whitening each channel
        ops['nSkipCov'] = 25.0  # compute whitening matrix from every N-th batch
        ops['scaleproc'] = 200.0  # int16 scaling of whitened data
        ops['nPCs'] = params['nPCs']  # how many PCs to project the spikes into
        ops['useRAM'] = 0.0  # not yet available

        # drift correction
        ops['do_correction'] = params['do_correction']

        ## option for wavelength
        ops['nt0'] = params['wave_length'] # size of the waveform extracted around each detected peak. Be sure to make it odd to make alignment easier.
        return ops
