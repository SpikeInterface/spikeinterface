import importlib.util
import sys
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path

from packaging.version import parse

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
        "enable_gate": True,
        "threshold": None,
        "direct_keep": None,
        "primary_snr": None,
        "secondary_snr": None,
        "min_secondary_channels": None,
        "detector_batch": 8,
        "embedding_batch": 128,
        "neighborhood_mode": "dense",
        "target_units": None,
        "postprocessing": None,
    }

    _params_description = {
        "model": "Model name; 'default' downloads pinned Hugging Face checkpoints once and reuses the local cache.",
        "model_path": "Optional local combined .pt checkpoint or directory containing config.json and both models.",
        "device": "'auto' selects CUDA when available, otherwise CPU; also accepts 'cpu', 'cuda', or 'cuda:N'.",
        "seed": "Random seed for model initialization and GMM clustering.",
        "components": (
            "GMM component count per neighborhood/segment, at least 2; "
            "independent of the output target_units. No automatic K estimation."
        ),
        "profile": "Published preprocessing/threshold profile: 'd1' or 'canonical'.",
        "enable_gate": "Enable the built-in SNR gate; False retains score thresholding, peak/NMS and ownership checks.",
        "threshold": "Minimum detector sigmoid score in [0, 1]; None uses the loaded profile's default.",
        "direct_keep": "Eligible scores at or above this value bypass SNR checks; None uses the profile's default.",
        "primary_snr": "Nonnegative strongest-channel event SNR minimum; None uses the model's default.",
        "secondary_snr": "Nonnegative additional-channel event SNR minimum; None uses the model's default.",
        "min_secondary_channels": "Number of additional channels meeting secondary_snr, from 1 to 3; None uses the model.",
        "detector_batch": "Number of 2500-sample detector chunks per batch; reduced on CUDA out-of-memory errors.",
        "embedding_batch": "Number of waveforms per HuiduRep batch; reduced on CUDA out-of-memory errors.",
        "neighborhood_mode": "'dense' preserves original neighborhoods; 'cover' uses fewer passes and may change accuracy.",
        "target_units": "Optional positive output unit count per segment after merging; None disables fixed-K processing.",
        "postprocessing": (
            "Optional fixed-K options dictionary; requires target_units. "
            "on_shortfall='return_available' permits fewer units when reliable anchors are insufficient."
        ),
    }

    sorter_description = """VanillaSort combines VanillaDet detection, HuiduRep waveform embeddings and
    relative-amplitude features with Gaussian-mixture clustering and template-residual refinement.
    The pretrained models use multichannel recordings with 2D channel locations and at least four channels
    per group/shank. Larger probes use experimental local neighborhoods. Segments have independent unit IDs.
    Parameters and examples: https://github.com/IgarashiAkatuki/VanillaSort/blob/main/docs/spikeinterface.md
    Reference: https://doi.org/10.64898/2026.09.18.752552.
    """

    installation_mesg = """
    This wrapper requires VanillaSort >= 0.2.1. Install or upgrade from PyPI:

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
        installed_version = cls.get_sorter_version()
        if parse(installed_version) < parse("0.2.1"):
            raise RuntimeError(
                f"VanillaSort >= 0.2.1 is required; found {installed_version}. "
                "Upgrade with 'pip install --upgrade vanillasort'."
            )
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
