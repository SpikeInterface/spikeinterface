from __future__ import annotations

from pathlib import Path
from packaging.version import parse
import importlib.util

from spikeinterface.preprocessing import bandpass_filter, whiten
from spikeinterface.sorters.basesorter import BaseSorter
from spikeinterface.core.old_api_utils import NewToOldRecording
from spikeinterface.extractors import NpzSortingExtractor, NumpySorting


class Mountainsort4Sorter(BaseSorter):
    """Mountainsort4 Sorter object."""

    sorter_name = "mountainsort4"
    requires_locations = False
    compatible_with_parallel = {"loky": True, "multiprocessing": False, "threading": False}

    _default_params = {
        "detect_sign": -1,  # Use -1, 0, or 1, depending on the sign of the spikes in the recording
        "adjacency_radius": -1,  # Use -1 to include all channels in every neighborhood
        "freq_min": 300,  # Use None for no bandpass filtering
        "freq_max": 6000,
        "filter": True,
        "whiten": True,  # Whether to do channel whitening as part of preprocessing
        "num_workers": 1,
        "clip_size": 50,
        "detect_threshold": 3,
        "detect_interval": 10,  # Minimum number of timepoints between events detected on the same channel
        "tempdir": None,
    }

    _params_description = {
        "detect_sign": "Use -1 (negative) or 1 (positive) depending " "on the sign of the spikes in the recording",
        # Use -1, 0, or 1, depending on the sign of the spikes in the recording
        "adjacency_radius": "Radius in um to build channel neighborhood "
        "(Use -1 to include all channels in every neighborhood)",
        # Use -1 to include all channels in every neighborhood
        "freq_min": "High-pass filter cutoff frequency",
        "freq_max": "Low-pass filter cutoff frequency",
        "filter": "Enable or disable filter",
        "whiten": "Enable or disable whitening",
        "num_workers": "Number of workers (if None, half of the cpu number is used)",
        "clip_size": "Number of samples per waveform",
        "detect_threshold": "Threshold for spike detection",
        "detect_interval": "Minimum number of timepoints between events detected on the same channel",
        "tempdir": "Temporary directory for mountainsort (available for ms4 >= 1.0.2)s",
    }

    sorter_description = """Mountainsort4 is a fully automatic density-based spike sorter using the isosplit clustering
    method and automatic curation procedures. For more information see https://doi.org/10.1016/j.neuron.2017.08.030"""

    installation_mesg = """\nTo use Mountainsort4 run:\n
       >>> pip install mountainsort4

    More information on mountainsort at:
      * https://github.com/flatironinstitute/mountainsort
    """

    @classmethod
    def is_installed(cls):

        ms4_spec = importlib.util.find_spec("mountainsort4")
        if ms4_spec is not None:
            HAVE_MS4 = True
        else:
            HAVE_MS4 = False
        return HAVE_MS4

    @staticmethod
    def get_sorter_version():
        import mountainsort4

        if hasattr(mountainsort4, "__version__"):
            return mountainsort4.__version__
        return "unknown"

    @classmethod
    def _check_apply_filter_in_params(cls, params):
        return params["filter"]

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        pass

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        import mountainsort4

        recording = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)

        # alias to params
        p = params

        samplerate = recording.get_sampling_frequency()

        # Bandpass filter
        if p["filter"] and p["freq_min"] is not None and p["freq_max"] is not None:
            if verbose:
                print("filtering")
            recording = bandpass_filter(recording=recording, freq_min=p["freq_min"], freq_max=p["freq_max"])

        # Whiten
        if p["whiten"]:
            if verbose:
                print("whitening")
            recording = whiten(recording=recording, dtype="float32")

        print("Mountainsort4 use the OLD spikeextractors mapped with NewToOldRecording")
        old_api_recording = NewToOldRecording(recording)

        ms4_params = dict(
            recording=old_api_recording,
            detect_sign=p["detect_sign"],
            adjacency_radius=p["adjacency_radius"],
            clip_size=p["clip_size"],
            detect_threshold=p["detect_threshold"],
            detect_interval=p["detect_interval"],
            num_workers=p["num_workers"],
            verbose=verbose,
        )

        # temporary folder
        ms4_version = Mountainsort4Sorter.get_sorter_version()

        if ms4_version != "unknown" and parse(ms4_version) >= parse("1.0.3"):
            if p["tempdir"] is not None:
                p["tempdir"] = str(p["tempdir"])
            if verbose:
                print(f'Using temporary directory {p["tempdir"]}')
            ms4_params.update(tempdir=p["tempdir"])

        # Check location no more needed done in basesorter
        old_api_sorting = mountainsort4.mountainsort4(**ms4_params)

        # convert sorting to new API and save it
        unit_ids = old_api_sorting.get_unit_ids()
        units_dict_list = [{u: old_api_sorting.get_unit_spike_train(u) for u in unit_ids}]
        new_api_sorting = NumpySorting.from_unit_dict(units_dict_list, samplerate)
        NpzSortingExtractor.write_sorting(new_api_sorting, str(sorter_output_folder / "firings.npz"))

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        sorter_output_folder = Path(sorter_output_folder)
        result_fname = sorter_output_folder / "firings.npz"
        sorting = NpzSortingExtractor(result_fname)
        return sorting
