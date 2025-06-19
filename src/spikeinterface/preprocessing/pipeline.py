from __future__ import annotations
from pathlib import Path
import inspect
from spikeinterface.core import BaseRecording
from spikeinterface.core.core_tools import is_dict_extractor, is_path_remote
from spikeinterface.core.zarrextractors import super_zarr_open
from spikeinterface.preprocessing.preprocessing_classes import preprocessor_dict, _all_preprocesser_dict


pp_names_to_functions = {preprocessor.__name__: preprocessor for preprocessor in preprocessor_dict.values()}
pp_names_to_classes = {pp_function.__name__: pp_class for pp_class, pp_function in _all_preprocesser_dict.items()}


class PreprocessingPipeline:
    """
    A preprocessing pipeline, containing ordered preprocessing steps.

    Parameters
    ----------
    preprocessor_dict : dict
        Dictionary containing preprocessing steps and their kwargs

    Examples
    --------
    Generate a `PreprocessingPipeline` containing a `bandpass_filter` then a
    `common_reference` step. Then apply this to a recording

    >>> from spikeinterface.preprocessing import PreprocessingPipeline
    >>> preprocessor_dict = {'bandpass_filter': {'freq_max': 3000}, 'common_reference': {}}
    >>> my_pipeline = PreprocessingPipeline(preprocessor_dict)
    PreprocessingPipeline:  Raw Recording → bandpass_filter → common_reference → Preprocessed Recording
    >>> my_pipeline._apply(recording)

    """

    def __init__(self, preprocessor_dict):

        non_supported_preprocessors = []
        for preprocessor in preprocessor_dict:
            if preprocessor not in pp_names_to_functions.keys():
                non_supported_preprocessors.append(preprocessor)

        if len(non_supported_preprocessors) > 0:
            raise TypeError(
                f"The preprocessors '{non_supported_preprocessors}' are not supported by the `PreprocessingPipeline`. \
To see the list of supported steps, run:\n>>> from spikeinterface.preprocessing.pipeline import pp_names_to_functions \
\n>>> print(pp_names_to_functions.keys())"
            )

        self.preprocessor_dict = preprocessor_dict

    def __repr__(self):
        txt = "PreprocessingPipeline: \tRaw Recording \u2192 "
        for preprocessor in self.preprocessor_dict:
            txt += str(preprocessor) + " \u2192 "
        txt += "Preprocessed Recording"
        return txt

    def _repr_html_(self):

        all_kwargs = _get_all_kwargs_and_values(self)

        html_text = "<div'>"
        html_text += "<strong>PreprocessingPipeline</strong>"
        html_text += "<div style='border:1px solid #ccc; padding:10px;'><strong>Initial Recording</strong></div>"
        html_text += "<div style='margin: auto; text-indent: 30px;'>&#x2193;</div>"

        for a, (preprocessor, kwargs) in enumerate(all_kwargs.items()):
            html_text += "<details style='border:1px solid #ddd; padding:5px;'>"
            html_text += f"<summary><strong>{preprocessor}</strong></summary>"

            html_text += "<ul>"
            for kwarg, value in kwargs.items():
                html_text += f"<li><strong>{kwarg}</strong>: {value}</li>"
            html_text += "</ul>"
            html_text += "</details>"

        html_text += """<div style='margin: auto; text-indent: 30px;'>&#x2193;</div>"""
        html_text += "<div style='border:1px solid #ccc; padding:10px;'><strong>Preprocessed Recording</strong></div>"
        html_text += "</div>"

        return html_text

    def _apply(self, recording, apply_precomputed_kwargs=False):
        """
        Creates a preprocessed recording by applying the `PreprocessingPipeline` to
        `recording`.

        Parameters
        ----------
        recording : BaseRecording
            The initial recording
        apply_precomputed_kwargs : bool, default: False
            Some preprocessing steps (e.g. Whitening) contain arguments which are computed
            during preprocessing. If True, we use the arguments which have already been
            computed. If False, we recompute them on application of the pipeline.

        Returns
        -------
        preprocessed_recording : BaseRecording
            Preprocessed recording

        """

        for preprocessor_name, kwargs in self.preprocessor_dict.items():

            dont_apply_kwargs = ["recording", "parent_recording"]

            if not apply_precomputed_kwargs:
                preprocessor_class = pp_names_to_classes[preprocessor_name]
                precomputable_kwarg_names = preprocessor_class._precomputable_kwarg_names
                dont_apply_kwargs += precomputable_kwarg_names

            non_rec_kwargs = {key: value for key, value in kwargs.items() if key not in dont_apply_kwargs}
            pp_output = pp_names_to_functions[preprocessor_name](recording, **non_rec_kwargs)
            recording = pp_output

        return recording


def apply_preprocessing_pipeline(
    recording: BaseRecording, pipeline_or_dict: PreprocessingPipeline | dict, apply_precomputed_kwargs=True
):
    """
    Creates a preprocessed recording by applying the preprocessing steps in
    `preprocessor_dict` to `recording`.

    Parameters
    ----------
    recording : BaseRecording
        The initial recording
    pipeline_or_dict : PreprocessingPipeline | dict
        Dictionary containing preprocessing steps and their kwargs, or a pipeline object.
        If None, the original recording is returned.
    apply_precomputed_kwargs : Bool, default: False
        Some preprocessing steps (e.g. Whitening) contain arguments which are computed
        during preprocessing. If True, we use the arguments which have already been
        computed. If False, we recompute them on application of the pipeline.

    Returns
    -------
    preprocessed_recording : BaseRecording
        Preprocessed recording

    Examples
    --------
    Create a preprocessed recording from a generated recording and a preprocessor_dict

    >>> from spikeinterface.preprocessing import create_preprocessed
    >>> from spikeinterface.generation import generate_recording
    >>> recording = generate_recording()
    >>> preprocessor_dict = {'bandpass_filter': {'freq_max': 3000}, 'common_reference': {}}
    >>> preprocessed_recording = apply_preprocessing_pipeline(recording, preprocessor_dict)
    """

    if isinstance(pipeline_or_dict, PreprocessingPipeline):
        pipeline = pipeline_or_dict
    elif isinstance(pipeline_or_dict, dict):
        pipeline = PreprocessingPipeline(pipeline_or_dict)
    else:
        raise TypeError("`pipeline_or_dict` must be a `PreprocessingPipeline` or a dict")

    preprocessed_recording = pipeline._apply(recording, apply_precomputed_kwargs)
    return preprocessed_recording


def get_preprocessing_dict_from_analyzer(analyzer_folder, format="auto", backend_options=None):
    """
    Generates a dictionary from a saved analyzer. The dictionary can be passed to the
    `PreprocessingPipeline` class to create a preprocessing pipeline.

    Parameters
    ----------
    analyzer_folder : str or Path
        Path to the analyzer.
    format : "auto" | "binary_folder" | "zarr", default: "auto"
        The format of the folder. If "auto", tries to guess format using filename.
    backend_options : dict | None, default: None
        The backend options for the backend.

    Returns
    -------
    preprocessing_dict : dict
        The preprocessing dict extracted from the analyzer's recording.
    """
    if not is_path_remote(analyzer_folder):
        analyzer_folder = Path(analyzer_folder)

    if format == "auto":
        if str(analyzer_folder).endswith(".zarr"):
            format = "zarr"
        else:
            format = "binary_folder"

    if format == "binary_folder":
        recording_files = list(analyzer_folder.glob("*recording.*"))
        if len(recording_files) == 0:
            raise FileNotFoundError(f"Cannot find `recording.*` file in {analyzer_folder}.")
        else:
            recording_file = recording_files[0]
            preprocessing_dict = get_preprocessing_dict_from_file(recording_file)

    elif format == "zarr":
        backend_options = {} if backend_options is None else backend_options
        storage_options = backend_options.get("storage_options", {})
        zarr_root = super_zarr_open(str(analyzer_folder), mode="r", storage_options=storage_options)

        rec_field = zarr_root.get("recording")
        if rec_field is not None:
            recording_dict = rec_field[0]
        else:
            recording_dict = {}

        preprocessing_dict = _make_pipeline_dict_from_recording_dict(recording_dict)

    return preprocessing_dict


def get_preprocessing_dict_from_file(recording_dictionary_path):
    """
    Generates a preprocessing dict, passable to `apply_preprocessing_pipeline` function and
    `PreprocessPipeline` class, from a recording dictionary.

    Only extracts preprocessing steps which can be applied "globally" to any recording.
    Hence this does not extract `ChannelSlice` and `FrameSlice` steps.

    Parameters
    ----------
    recording_dictionary_path : str or Path
        Path to the `.json` or `.pkl` output from a saved recording.

    Returns
    -------
    preprocessor_dict : dict
        Dictionary containing preprocessing steps and their kwargs

    """

    if str(recording_dictionary_path).endswith(".json"):
        import json

        with open(recording_dictionary_path, "r") as f:
            recording_dict = json.load(f)
    elif str(recording_dictionary_path).endswith(".pkl") or str(recording_dictionary_path).endswith(".pickle"):
        import pickle

        with open(recording_dictionary_path, "rb") as f:
            recording_dict = pickle.load(f)

    pipeline_dict = _make_pipeline_dict_from_recording_dict(recording_dict)
    return pipeline_dict


def _make_pipeline_dict_from_recording_dict(recording_dict):
    """
    Transforms a recording dict (created by the `dump` method of `BaseRecording`)
    into a preprocessing pipeline dict.
    """

    pipeline_dict_from_file = {}
    _ = _load_pp_from_dict(recording_dict, pipeline_dict_from_file)

    pipeline_dict = {}
    for preprocessor in pipeline_dict_from_file:

        preprocessor_class_name = preprocessor.split(".")[-1]

        preprocessor_function = preprocessor_dict.get(preprocessor_class_name)
        if preprocessor_function is None:
            continue

        pp_kwargs = {
            key: value
            for key, value in pipeline_dict_from_file[preprocessor].items()
            if key not in ["recording", "parent_recording"]
        }

        pipeline_dict[preprocessor_function.__name__] = pp_kwargs

    return pipeline_dict


def _load_pp_from_dict(prov_dict, kwargs_dict):
    """
    Recursive function used to iterate through a recording dictionary,
    extract preprocessing steps and their kwargs, and add them to `kwargs_dict`.
    Based on `_load_extractor_from_dict` from spikeinterface.core.base.

    Parameters
    ----------
    prov_dict : dict
        The dictionary created when a recording is saved by the
        `save_to_folder` method from `spikeinterface.core.base`.
    kwargs_dict : dict
        A dictionary just containing the preprocessing step names and their kwargs,
        extracted from prov_dict.

    Returns
    -------
    current_level_kwargs
        The kwargs of the preprocessing step at the current level of the recursion.
    """
    this_level_kwargs = dict()

    prov_dict_to_kwargs_dict = lambda x: _load_pp_from_dict(x, kwargs_dict) if is_dict_extractor(x) else x

    for name, value in prov_dict["kwargs"].items():
        if is_dict_extractor(value):
            this_level_kwargs[name] = _load_pp_from_dict(value, kwargs_dict)
        elif isinstance(value, dict):
            this_level_kwargs[name] = {k: prov_dict_to_kwargs_dict(v) for k, v in value.items()}
        elif isinstance(value, list):
            this_level_kwargs[name] = [prov_dict_to_kwargs_dict(e) for e in value]
        else:
            this_level_kwargs[name] = value

    kwargs_dict[prov_dict["class"]] = this_level_kwargs
    return this_level_kwargs


def _get_all_kwargs_and_values(my_pipeline):
    """
    Get all keyword arguments and their values from a pipeline,
    including the default values.
    """

    all_kwargs = {}
    for preprocessor in my_pipeline.preprocessor_dict:

        preprocessor_name = preprocessor.split(".")[-1]
        pp_function = pp_names_to_functions[preprocessor.split(".")[-1]]
        signature = inspect.signature(pp_function)

        all_kwargs[preprocessor_name] = {}

        for _, value in signature.parameters.items():
            par_name = str(value).split("=")[0].split(":")[0]
            if par_name != "recording":
                try:
                    default_value = str(value).split("=")
                    if len(default_value) == 1:
                        default_value = None
                    else:
                        default_value = default_value[-1]
                except:
                    default_value = None

                pipeline_value = my_pipeline.preprocessor_dict[preprocessor].get(par_name)

                if pipeline_value is None:
                    if default_value != pipeline_value:
                        pipeline_value = default_value

                all_kwargs[preprocessor_name][par_name] = pipeline_value

    return all_kwargs
