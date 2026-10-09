from contextlib import redirect_stdout
from pathlib import Path

from ..basesorter import BaseSorter
from ...core import load


class VanillaSortSorter(BaseSorter):
    """Run the VanillaSort package's public recording-to-sorting API."""

    sorter_name = "vanillasort"
    requires_locations = True
    handle_multi_segment = True
    gpu_capability = "optional"
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
    Install VanillaSort >= 0.1.0 from PyPI:

        pip install vanillasort

    Follow the repository README for CPU/CUDA PyTorch installation instructions.
    """

    @classmethod
    def is_installed(cls):
        try:
            import vanillasort
            from packaging.version import parse

            return callable(getattr(vanillasort, "sort", None)) and parse(vanillasort.__version__) >= parse("0.1.0")
        except (ImportError, AttributeError):
            return False

    @staticmethod
    def get_sorter_version():
        import vanillasort

        return vanillasort.__version__

    @classmethod
    def use_gpu(cls, params):
        device = params.get("device", "auto")
        if device == "auto":
            import torch

            return torch.cuda.is_available()
        return str(device).startswith("cuda")

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
        # BaseSorter includes this trace in spikeinterface_log.json, including on failure.
        with (sorter_output_folder / "vanillasort.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            print(f"VanillaSort {vanillasort.__version__}", flush=True)
            sorting = vanillasort.sort(
                recording, output_folder=sorter_output_folder / "inference", verbose=verbose, **params
            )
            print(f"Completed: {sorting.get_num_units()} units, {sorting.get_num_segments()} segments", flush=True)

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        return load(Path(sorter_output_folder) / "inference" / "sorting")
