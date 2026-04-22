from pathlib import Path
from packaging.version import parse

from ..basesorter import BaseSorter
from ...core import NumpyFolderSorting, NumpySorting

class DartsortSorter(BaseSorter):
    """Dartsort wrapper"""

    sorter_name = "dartsort"
    requires_locations = False
    compatible_with_parallel = {"loky": False, "multiprocessing": False, "threading": False}
    sorter_description = "Dartsort is the Columbia university sorter made with love by Charlie Windolf and Liam Paninski's team."
    installation_mesg = """\nTo use dartsort run:\n
       >>> pip install dartsort

    More information on mountainsort5 at:
      * https://github.com/cwindolf/dartsort
    """

    _default_params = {
    }

    _params_description = {
    }

    @classmethod
    def _dynamic_params(cls):
        from dartsort import DARTsortUserConfig
        from pydantic import RootModel
        # the trick is to transform the DARTsortUserConfig  (a pydantic.dataclass) into a pydantic model
        Model = RootModel[DARTsortUserConfig]
        # so we can dump to dict
        cfg = Model(DARTsortUserConfig())
        default_params = cfg.model_dump(mode='python')
        # and retrieve properties
        schema = Model.model_json_schema()
        default_params_descriptions = {}
        for k, props in schema['$defs']['DARTsortUserConfig']['properties'].items():
            default_params_descriptions[k] = props['title']

        return default_params, default_params_descriptions

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
        from dartsort import dartsort as dartsort_main
        from dartsort import DARTsortUserConfig

        recording = cls.load_recording_from_folder(sorter_output_folder.parent, with_warnings=False)

        # dartsort config are set using dataclass we need to map this
        cfg = DARTsortUserConfig(**params)
        
        ret = dartsort_main(
            recording,
            sorter_output_folder,
            cfg,
        )
        # the dartsort_sorting is not the spikeinterface sorting!!!
        dartsort_sorting = ret['sorting']

        times_samples = dartsort_sorting.times_samples
        labels = dartsort_sorting.labels
        mask = labels >= 0

        sorting = NumpySorting.from_samples_and_labels(
            [times_samples[mask]], [labels[mask]], dartsort_sorting.sampling_frequency
        )
        
        NumpyFolderSorting.write_sorting(sorting, sorter_output_folder / "final_darsort_sorting")

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        sorter_output_folder = Path(sorter_output_folder)
        sorting = NumpyFolderSorting(sorter_output_folder / "final_darsort_sorting")
        return sorting
