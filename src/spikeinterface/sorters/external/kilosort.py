from __future__ import annotations

from pathlib import Path
import os
from typing import Union
import numpy as np

from spikeinterface.sorters.basesorter import BaseSorter
from .kilosortbase import KilosortBase
from spikeinterface.sorters.utils import get_git_commit


def check_if_installed(kilosort_path: Union[str, None]):
    if kilosort_path is None:
        return False
    assert isinstance(kilosort_path, str)

    if kilosort_path.startswith('"'):
        kilosort_path = kilosort_path[1:-1]
    kilosort_path = str(Path(kilosort_path).absolute())

    if (Path(kilosort_path) / "preprocessData.m").is_file():
        return True
    else:
        return False


class KilosortSorter(KilosortBase, BaseSorter):
    """Kilosort Sorter object."""

    sorter_name: str = "kilosort"
    compiled_name: str = "ks_compiled"
    kilosort_path: Union[str, None] = os.getenv("KILOSORT_PATH", None)
    requires_locations = False
    requires_gpu = "nvidia-optional"

    _default_params = {
        "detect_threshold": 6,
        "car": True,
        "useGPU": True,
        "freq_min": 300,
        "freq_max": 6000,
        "ntbuff": 64,
        "Nfilt": None,
        "NT": None,
        "wave_length": 61,
        "delete_tmp_files": ("matlab_files",),
        "delete_recording_dat": False,
    }

    _params_description = {
        "detect_threshold": "Threshold for spike detection",
        "car": "Enable or disable common reference",
        "useGPU": "Enable or disable GPU usage",
        "freq_min": "High-pass filter cutoff frequency",
        "freq_max": "Low-pass filter cutoff frequency",
        "ntbuff": "Samples of symmetrical buffer for whitening and spike detection",
        "Nfilt": "Number of clusters to use (if None it is automatically computed)",
        "NT": "Batch size (if None it is automatically computed--recommended Kilosort behavior if ntbuff also not changed)",
        "wave_length": "size of the waveform extracted around each detected peak, (Default 61, maximum 81)",
        "delete_tmp_files": "Delete temporary files created during sorting (matlab files and the `temp_wh.dat` file that "
        "contains kilosort-preprocessed data). Accepts `False` (deletes no files), `True` (deletes all files) "
        "or a Tuple containing the files to delete. Options are: ('temp_wh.dat', 'matlab_files')",
        "delete_recording_dat": "Whether to delete the 'recording.dat' file after a successful run",
    }

    sorter_description = """Kilosort is a GPU-accelerated and efficient template-matching spike sorter.
    For more information see https://papers.nips.cc/paper/6326-fast-and-accurate-spike-sorting-of-high-channel-count-probes-with-kilosort"""

    installation_mesg = """\nTo use Kilosort run:\n
        >>> git clone https://github.com/cortex-lab/KiloSort
    and provide the installation path by setting the KILOSORT_PATH
    environment variables or using KilosortSorter.set_kilosort_path().\n\n

    More information on KiloSort at:
        https://github.com/cortex-lab/KiloSort
    """

    handle_multi_segment = False

    @classmethod
    def is_installed(cls):
        if cls.check_compiled():
            return True
        return check_if_installed(cls.kilosort_path)

    @classmethod
    def get_sorter_version(cls):
        if cls.check_compiled():
            return "compiled"
        commit = get_git_commit(os.getenv("KILOSORT_PATH", None))
        if commit is None:
            return "unknown"
        else:
            return "git-" + commit

    @classmethod
    def use_gpu(cls, params):
        if "useGPU" in params:
            return params["useGPU"]
        return cls.default_params()["useGPU"]

    @classmethod
    def set_kilosort_path(cls, kilosort_path: str):
        kilosort_path = str(Path(kilosort_path).absolute())
        KilosortSorter.kilosort_path = kilosort_path
        try:
            print("Setting KILOSORT_PATH environment variable for subprocess calls to:", kilosort_path)
            os.environ["KILOSORT_PATH"] = kilosort_path
        except Exception as e:
            print("Could not set KILOSORT_PATH environment variable:", e)

    @classmethod
    def _check_params(cls, recording, output_folder, params):
        p = params
        nchan = recording.get_num_channels()
        if p["Nfilt"] is None:
            p["Nfilt"] = (nchan // 32) * 32 * 8
        else:
            p["Nfilt"] = p["Nfilt"] // 32 * 32
        if p["Nfilt"] == 0:
            p["Nfilt"] = nchan * 8
        if p["NT"] is None:
            p["NT"] = 64 * 1024 + p["ntbuff"]
        else:
            p["NT"] = p["NT"] // 32 * 32  # make sure is multiple of 32
        if p["wave_length"] % 2 != 1:
            p["wave_length"] = p["wave_length"] + 1  # The wave_length must be odd
        if p["wave_length"] > 81:
            p["wave_length"] = 81  # The wave_length must be less than 81.
        return p

    @classmethod
    def _get_specific_options(cls, ops, params) -> dict:
        """
        Adds specific options for Kilosort in the ops dict and returns the final dict

        Parameters
        ----------
        ops : dict
            options data
        params : dict
            Custom parameters dictionary for kilosort3

        Returns
        ----------
        ops : dict
            Final ops data
        """

        # TODO: Check GPU option!
        ops["GPU"] = params["useGPU"]  # whether to run this code on an Nvidia GPU (much faster, mexGPUall first)
        ops["parfor"] = 0.0  # whether to use parfor to accelerate some parts of the algorithm
        ops["verbose"] = 1.0  # whether to print command line progress
        ops["showfigures"] = 0.0  # whether to plot figures during optimization

        ops["Nfilt"] = params[
            "Nfilt"
        ]  # number of clusters to use (2-4 times more than Nchan, should be a multiple of 32)
        ops["nNeighPC"] = min(
            12.0, ops["Nchan"]
        )  # visualization only (Phy): number of channnels to mask the PCs, leave empty to skip (12)
        ops["nNeigh"] = 16.0  # visualization only (Phy): number of neighboring templates to retain projections of (16)

        # options for channel whitening
        ops["whitening"] = (
            "full"  # type of whitening (default 'full', for 'noSpikes' set options for spike detection below)
        )
        ops["nSkipCov"] = 1.0  # compute whitening matrix from every N-th batch (1)
        ops["whiteningRange"] = (
            32.0  # how many channels to whiten together (Inf for whole probe whitening, should be fine if Nchan<=32)
        )

        # ops['criterionNoiseChannels'] = 0.2  # fraction of "noise" templates allowed to span all channel groups (see createChannelMapFile for more info).

        # other options for controlling the model and optimization
        ops["Nrank"] = 3.0  # matrix rank of spike template model (3)
        ops["nfullpasses"] = 6.0  # number of complete passes through data during optimization (6)
        ops["maxFR"] = 20000  # maximum number of spikes to extract per batch (20000)
        ops["fshigh"] = params["freq_min"]  # frequency for high pass filtering
        ops["fslow"] = params["freq_max"]  # frequency for low pass filtering (optional)
        ops["ntbuff"] = params["ntbuff"]  # samples of symmetrical buffer for whitening and spike detection
        ops["scaleproc"] = 200.0  # int16 scaling of whitened data
        ops["NT"] = params["NT"]  # 32*1024+ ops.ntbuff;
        # this is the batch size (try decreasing if out of memory)
        # for GPU should be multiple of 32 + ntbuff

        # the following options can improve/deteriorate results.
        # when multiple values are provided for an option, the first two are beginning and ending anneal values,
        # the third is the value used in the final pass.
        ops["Th"] = [4.0, 10.0, 10.0]  # threshold for detecting spikes on template-filtered data ([6 12 12])
        ops["lam"] = [5.0, 5.0, 5.0]  # large means amplitudes are forced around the mean ([10 30 30])
        ops["nannealpasses"] = 4.0  # should be less than nfullpasses (4)
        ops["momentum"] = [1 / 20, 1 / 400]  # start with high momentum and anneal (1./[20 1000])
        ops["shuffle_clusters"] = 1.0  # allow merges and splits during optimization (1)
        ops["mergeT"] = 0.1  # upper threshold for merging (.1)
        ops["splitT"] = 0.1  # lower threshold for splitting (.1)

        ops["initialize"] = "fromData"  # 'fromData' or 'no'
        ops["spkTh"] = -params["detect_threshold"]  # spike threshold in standard deviations (-6)
        ops["loc_range"] = [3.0, 1.0]  # ranges to detect peaks; plus/minus in time and channel ([3 1])
        ops["long_range"] = [30.0, 6.0]  # ranges to detect isolated peaks ([30 6])
        ops["maskMaxChannels"] = 5.0  # how many channels to mask up/down ([5])
        ops["crit"] = 0.65  # upper criterion for discarding spike repeates (0.65)
        ops["nFiltMax"] = 10000.0  # maximum "unique" spikes to consider (10000)

        # options for posthoc merges (under construction)
        ops["fracse"] = 0.1  # binning step along discriminant axis for posthoc merges (in units of sd)
        ops["epu"] = np.inf

        ops["ForceMaxRAMforDat"] = 20e9  # maximum RAM the algorithm will try to use; on Windows it will autodetect.

        ## option for wavelength
        ops["nt0"] = params[
            "wave_length"
        ]  # size of the waveform extracted around each detected peak. Be sure to make it odd to make alignment easier.
        return ops
