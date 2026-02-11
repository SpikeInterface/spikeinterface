from pathlib import Path
from packaging.version import parse

from ..basesorter import BaseSorter
from ...core import NumpyFolderSorting

class DartsortSorter(BaseSorter):
    """Dasrtsort wrapper"""

    sorter_name = "dartsort"
    requires_locations = False
    compatible_with_parallel = {"loky": False, "multiprocessing": False, "threading": False}

    sorter_description = "Dartsort is the Columbia university sorter made with love by Charlie Windolf and Liam Paninski's team."

    _default_params = {
    }

    _params_description = {
    }

    @classmethod
    def _dynamic_params(cls):
        from darsort import DARTsortUserConfig
        from pydantic import RootModel
        # the trick is to transform the DARTsortUserConfig  (a pydantic.dataclass) into a pydantic model
        model = RootModel[DARTsortUserConfig](DARTsortUserConfig())
        default_params = model.model_dump(mode='python')
        # default_params_descriptions = 

        return default_params, default_params_descriptions



    

    installation_mesg = """\nTo use dartsort run:\n
       >>> pip install dartsort

    More information on mountainsort5 at:
      * https://github.com/cwindolf/dartsort
    """

    @classmethod
    def is_installed(cls):
        try:
            import dartsort

            HAVE_DARTSORT = True
        except ImportError:
            HAVE_DARTSORT = False

        return HAVE_DARTSORT

    @staticmethod
    def get_sorter_version():
        import dartsort

        if hasattr(dartsort, "__version__"):
            return dartsort.__version__
        return "unknown"

    @classmethod
    def _setup_recording(cls, recording, sorter_output_folder, params, verbose):
        pass

    @classmethod
    def _run_from_folder(cls, sorter_output_folder, params, verbose):
        from dartsort.main import dartsort as dartsort_main
        from dartsort.config import WaveformConfig, FeaturizationConfig, SubtractionConfig, TemplateConfig, MatchingConfig

        recording = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)

        # dartsort config are set using dataclass we need to map this
        
        waveform_config = WaveformConfig(**p["featurization"])
        trough_offset_samples = waveform_config.trough_offset_samples
        featurization_config = FeaturizationConfig(**p["featurization"])
        subtraction_config = SubtractionConfig(trough_offset_samples=trough_offset_samples, **p["subtraction"])
        template_config = TemplateConfig(trough_offset_samples=trough_offset_samples, **p["template"])
        matching_config = MatchingConfig(trough_offset_samples=trough_offset_samples, **p["matching"])

        sorting= dartsort_main(
            recording,
            sorter_output_folder,

            featurization_config=featurization_config,
            subtraction_config=subtraction_config,
            

            n_jobs=p["n_jobs"],
            overwrite=False,
            show_progress=verbose,
            device=p["device"],
        )
        
        NumpyFolderSorting.write_sorting(sorting, sorter_output_folder / "final_sorting")

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        sorter_output_folder = Path(sorter_output_folder)
        sorting = NumpyFolderSorting(sorter_output_folder / "final_sorting")
        return sorting
