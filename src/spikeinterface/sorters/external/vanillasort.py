import importlib.util
import sys
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path

from ..basesorter import BaseSorter
from ...core import load


class _LogAndStdout:
    """Keep the runtime log while echoing verbose package progress."""

    def __init__(self, log, stdout):
        self.log = log
        self.stdout = stdout

    def write(self, text):
        written = self.log.write(text)
        self.stdout.write(text)
        return written

    def flush(self):
        self.log.flush()
        self.stdout.flush()


class VanillaSortSorter(BaseSorter):
    """Run the VanillaSort package's public recording-to-sorting API."""

    sorter_name = "vanillasort"
    requires_locations = True
    handle_multi_segment = True
    gpu_capability = "nvidia-optional"
    compatible_with_parallel = {"loky": False, "multiprocessing": False, "threading": False}

    _default_params = {
        "model": "default",
        "model_path": None,
        "device": "auto",
        "seed": 0,
        "components": 22,
        "profile": "d1",
        "detector_batch": 8,
        "embedding_batch": 128,
    }

    _params_description = {
        "model": "Model name; 'default' downloads pinned Hugging Face checkpoints once and reuses the local cache.",
        "model_path": "Optional local combined .pt checkpoint or directory containing config.json and both models.",
        "device": "'auto' selects CUDA when available, otherwise CPU; also accepts 'cpu', 'cuda', or 'cuda:N'.",
        "seed": "Random seed for model initialization and GMM clustering.",
        "components": "GMM component count per neighborhood/segment; choose K for the recording (no automatic estimation).",
        "profile": "Published preprocessing/threshold profile: 'd1' or 'canonical'.",
        "detector_batch": "Number of 2500-sample detector chunks per batch; reduced on CUDA out-of-memory errors.",
        "embedding_batch": "Number of waveforms per HuiduRep batch; reduced on CUDA out-of-memory errors.",
    }

    sorter_description = """VanillaSort combines VanillaDet detection, HuiduRep waveform embeddings and
    relative-amplitude features with Gaussian-mixture clustering and template-residual refinement.
    The pretrained models require 30 kHz recordings and 2D channel locations, with at least four channels
    per group/shank. Larger probes use experimental local neighborhoods. Segments have independent unit IDs.
    See https://github.com/IgarashiAkatuki/VanillaSort and https://doi.org/10.64898/2026.09.18.752552.
    """

    installation_mesg = """
    Install or upgrade VanillaSort from PyPI:

        pip install --upgrade vanillasort

    Follow the repository README for CPU/CUDA PyTorch installation instructions.
    """

    @classmethod
    def is_installed(cls):
        return importlib.util.find_spec("vanillasort") is not None

    @staticmethod
    def get_sorter_version():
        return version("vanillasort")

    @classmethod
    def use_gpu(cls, params):
        # Express GPU intent without importing torch on the host. The container
        # runner checks NVIDIA availability for nvidia-optional sorters, and
        # VanillaSort resolves "auto" in the environment that runs inference.
        device = params.get("device", "auto")
        return device == "auto" or str(device).startswith("cuda")

    @classmethod
    def _check_params(cls, recording, output_folder, params):
        if params["model_path"] is not None:
            params["model_path"] = str(Path(params["model_path"]).expanduser().resolve())
        return params

    @classmethod
    def _check_apply_filter_in_params(cls, params):
        return True

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        # BaseSorter serializes the recording; the package reads its traces and geometry.
        pass

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        import vanillasort

        recording = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)
        if recording is None:
            raise RuntimeError("Unable to load the recording for VanillaSort")
        # Keep progress in the runtime trace and echo it for verbose runs.
        # redirect_stdout restores sys.stdout even on failure. Its process-wide
        # effect is why this wrapper disables threaded runs.
        with (sorter_output_folder / "vanillasort.log").open("w", encoding="utf-8", buffering=1) as log:
            stream = _LogAndStdout(log, sys.stdout) if verbose else log
            with redirect_stdout(stream):
                print(f"VanillaSort {vanillasort.__version__}", flush=True)
                sorting = vanillasort.sort(
                    recording, output_folder=sorter_output_folder / "inference", verbose=verbose, **params
                )
                print(f"Completed: {sorting.get_num_units()} units, {sorting.get_num_segments()} segments", flush=True)

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        return load(Path(sorter_output_folder) / "inference" / "sorting")
