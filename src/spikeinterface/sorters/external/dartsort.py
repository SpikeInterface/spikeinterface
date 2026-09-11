from pathlib import Path

import numpy as np

from ..basesorter import BaseSorter
from ...core import load


class DartsortSorter(BaseSorter):
    """Dartsort wrapper"""

    sorter_name = "dartsort"
    requires_locations = False
    compatible_with_parallel = {"loky": False, "multiprocessing": False, "threading": False}
    sorter_description = """dartsort is a modular, drift-aware spike sorter developed in the Paninski lab. For installation and documentation, see https://dartsort.github.io"""
    installation_mesg = """\nTo use dartsort run:\n
       >>> pip install dartsort

    More information about installing dartsort at:
      * https://dartsort.github.io
    """

    _default_params = {}

    _params_description = {}

    @classmethod
    def _dynamic_params(cls):
        from dartsort import DARTsortUserConfig
        from pydantic import RootModel

        # the trick is to transform the DARTsortUserConfig  (a pydantic.dataclass) into a pydantic model
        Model = RootModel[DARTsortUserConfig]
        # so we can dump to dict
        cfg = Model(DARTsortUserConfig())
        default_params = cfg.model_dump(mode="python")
        # and retrieve properties
        schema = Model.model_json_schema()
        default_params_descriptions = {}
        for k, props in schema["$defs"]["DARTsortUserConfig"]["properties"].items():
            default_params_descriptions[k] = props["title"]

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

        # Dartsort can be given the motion object optionaly
        motion = params.pop("motion", None)

        # dartsort config are set using dataclass we need to map this
        cfg = DARTsortUserConfig(**params)

        ret = dartsort_main(
            recording,
            sorter_output_folder,
            cfg,
            motion=motion,
        )
        # the DARTsortSorting is not the spikeinterface sorting
        dartsort_sorting = ret["sorting"]
        sorting = dartsort_sorting.to_numpy_sorting()

        # Add main_channel_id property by taking mode of channels from spikes
        labels = dartsort_sorting.labels
        spike_channels = dartsort_sorting.channels
        main_channel_indices = [np.bincount(spike_channels[labels == unit_id]).argmax() for unit_id in sorting.unit_ids]
        main_channel_ids = recording.channel_ids[main_channel_indices]
        sorting.set_property("main_channel_id", main_channel_ids)
        # We save to the final_darsort_sorting folder to propagate the main_channel_id property
        sorting.save(folder=sorter_output_folder / "final_darsort_sorting")

    @classmethod
    def _get_result_from_folder(cls, sorter_output_folder):
        sorter_output_folder = Path(sorter_output_folder)
        sorting = load(sorter_output_folder / "final_darsort_sorting")
        return sorting
