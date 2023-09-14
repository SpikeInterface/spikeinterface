from spikeinterface.core import load_extractor, NumpyRecording

from spikeinterface.sorters import BaseSorter


class ComponentsBasedSorter(BaseSorter):
    """
    This is a based class for sorter based on spikeinterface.sortingcomponents
    """

    @classmethod
    def is_installed(cls):
        return True

    @classmethod
    def _setup_recording(cls, recording, output_folder, params, verbose):
        # Some recording not json serializable but they can be saved to pickle
        #   * NoiseGeneratorRecording or InjectTemplatesRecording: we force a pickle because this is light
        #   * for NumpyRecording (this is a bit crazy because it flush the entire buffer!!)
        if recording.check_if_dumpable() and not isinstance(recording, NumpyRecording):
            rec_file = output_folder.parent / "spikeinterface_recording.pickle"
            recording.dump_to_pickle(rec_file)
        # TODO (hard) : find a solution for NumpyRecording without any dump
        # this will need an internal API change I think
        # because the run_sorter is from the "folder" (because of container mainly and also many other reasons)
        # and not from the recording itself

    @classmethod
    def _get_result_from_folder(cls, output_folder):
        sorting = load_extractor(output_folder / "sorting")
        return sorting

    @classmethod
    def _check_apply_filter_in_params(cls, params):
        return False
