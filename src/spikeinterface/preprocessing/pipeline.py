from pathlib import Path
import inspect
import warnings
from spikeinterface.core import BaseRecording
from spikeinterface.core.core_tools import is_dict_extractor, is_path_remote
from spikeinterface.core.zarrextractors import super_zarr_open
from spikeinterface.preprocessing.preprocessing_classes import preprocessor_dict, _all_preprocesser_dict

pp_names_to_functions = {preprocessor.__name__: preprocessor for preprocessor in preprocessor_dict.values()}


class BasePipeline:
    """
    Base processing pipeline to construct a processing pipeline from a list of processing steps and their params.

    Inherited classes should define the `function_names_to_functions` attributes,
    which map the names of processing steps to their corresponding functions, respectively.
    """

    function_names_to_functions = dict()

    def __init__(self, preprocessor_list_or_dict):
        non_supported_preprocessors = []
        # convert dicts to lists
        preprocessor_list = []
        if isinstance(preprocessor_list_or_dict, dict):
            for key, value in preprocessor_list_or_dict.items():
                step = dict(name=key, params=value)
                preprocessor_list.append(step)
        elif isinstance(preprocessor_list_or_dict, list):
            preprocessor_list = preprocessor_list_or_dict
            assert all(
                isinstance(step, dict) and "name" in step for step in preprocessor_list
            ), "Each step in the preprocessor list must be a dict with 'name' key."

        for step in preprocessor_list:
            if "params" not in step:
                step["params"] = {}

        for preprocessor in preprocessor_list:
            if preprocessor["name"] not in self.function_names_to_functions.keys():
                non_supported_preprocessors.append(preprocessor["name"])

        if len(non_supported_preprocessors) > 0:
            raise TypeError(
                f"The preprocessors '{non_supported_preprocessors}' are not supported by the pipeline. "
                f"Available preprocessors are: {list(self.function_names_to_functions.keys())}"
            )

        self.preprocessor_list = preprocessor_list

    def __repr__(self):
        txt = "Pipeline: \tinput \u2192 "
        for preprocessor in self.preprocessor_list:
            txt += str(preprocessor["name"]) + " \u2192 "
        txt += "preprocessed"
        return txt

    def _repr_html_(self):

        all_kwargs_list = _get_all_kwargs_and_values(self)

        html_text = "<div'>"
        html_text += "<strong>PreprocessingPipeline</strong>"
        html_text += "<div style='border:1px solid #ccc; padding:10px;'><strong>input</strong></div>"
        html_text += "<div style='margin: auto; text-indent: 30px;'>&#x2193;</div>"

        for all_kwargs in all_kwargs_list:
            preprocessor = all_kwargs["name"]
            kwargs = all_kwargs["kwargs"]
            html_text += "<details style='border:1px solid #ddd; padding:5px;'>"
            html_text += f"<summary><strong>{preprocessor}</strong></summary>"

            html_text += "<ul>"
            for kwarg, value in kwargs.items():
                html_text += f"<li><strong>{kwarg}</strong>: {value}</li>"
            html_text += "</ul>"
            html_text += "</details>"

        html_text += """<div style='margin: auto; text-indent: 30px;'>&#x2193;</div>"""
        html_text += "<div style='border:1px solid #ccc; padding:10px;'><strong>preprocessed</strong></div>"
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
        instantiated_recordings = {"raw": recording}
        for step in self.preprocessor_list:
            preprocessor_name = step["name"]
            params = step["params"].copy()
            dont_apply_kwargs = ["recording", "parent_recording"]

            for k, v in params.items():
                if isinstance(v, str) and "pipeline[" in v:
                    if "recording" not in k:
                        raise ValueError(
                            f"Cannot substitute recording for argument '{k}' of preprocessor '{preprocessor_name}' "
                            f"because this argument is not meant to be a recording object."
                        )
                    if k in dont_apply_kwargs:
                        raise ValueError(
                            f"Cannot substitute recording for argument '{k}' of preprocessor '{preprocessor_name}' "
                            f"because this argument is reserved for the recording to be preprocessed."
                        )
                    rec_name = v.split("pipeline[")[-1].split("]")[0]
                    substituted_recording = instantiated_recordings.get(rec_name)
                    if substituted_recording is None:
                        raise ValueError(f"Cannot find recording '{rec_name}' from previous steps in the pipeline.")
                    params[k] = substituted_recording

            if not apply_precomputed_kwargs:
                preprocessor_function = self.function_names_to_functions[preprocessor_name]
                if hasattr(preprocessor_function, "_precomputable_kwarg_names"):
                    precomputable_kwarg_names = preprocessor_function._precomputable_kwarg_names
                    dont_apply_kwargs += precomputable_kwarg_names

            non_rec_params = {key: value for key, value in params.items() if key not in dont_apply_kwargs}
            pp_output = self.function_names_to_functions[preprocessor_name](recording, **non_rec_params)
            recording = pp_output
            instantiated_recordings[preprocessor_name] = recording

        return recording


class PreprocessingPipeline(BasePipeline):
    """
    A preprocessing pipeline, containing ordered preprocessing steps.

    Parameters
    ----------
    preprocessor_list_or_dict : dict or list
        Dictionary or list containing preprocessing steps and their kwargs

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

    function_names_to_functions = pp_names_to_functions


def apply_preprocessing_pipeline(
    recording_or_dict: BaseRecording | dict,
    pipeline: PreprocessingPipeline | list | dict,
    apply_precomputed_kwargs=True,
):
    """
    Creates a preprocessed recording by applying the preprocessing steps in
    `pipeline` to `recording`.

    Parameters
    ----------
    recording_or_dict : BaseRecording | dict
        The initial recording or a dictionary of recordings
    pipeline : PreprocessingPipeline | list | dict
        A list of preprocessing steps, or a pipeline object.
        If None, the original recording is returned.
    apply_precomputed_kwargs : Bool, default: True
        Some preprocessing steps (e.g. Whitening) contain arguments which are computed
        during preprocessing. If True, we use the arguments which have already been
        computed. If False, we recompute them on application of the pipeline.

    Returns
    -------
    preprocessed_recording : BaseRecording
        Preprocessed recording

    Examples
    --------
    Create a preprocessed recording from a generated recording and a preprocessing pipeline

    >>> from spikeinterface.preprocessing import create_preprocessed
    >>> from spikeinterface.generation import generate_recording
    >>> recording = generate_recording()
    >>> pipeline = [{'name': 'bandpass_filter', 'kwargs': {'freq_max': 3000}}, {'name': 'common_reference', 'kwargs': {}}]
    >>> preprocessed_recording = apply_preprocessing_pipeline(recording, pipeline)
    """

    if isinstance(pipeline, PreprocessingPipeline):
        pipeline = pipeline
    elif isinstance(pipeline, list):
        pipeline = PreprocessingPipeline(pipeline)
    elif isinstance(pipeline, dict):
        warnings.warn(
            "Passing a dict to `apply_preprocessing_pipeline` is deprecated and will be removed in 0.106.0. "
            "Please pass a list of preprocessing steps instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        pipeline = PreprocessingPipeline(pipeline)
    else:
        raise TypeError("`pipeline` must be a `PreprocessingPipeline`, a list, or a dict")

    preprocessed_recording = pipeline._apply(recording_or_dict, apply_precomputed_kwargs)
    return preprocessed_recording


def get_preprocessing_list_from_analyzer(analyzer_folder, format="auto", backend_options=None):
    """
    Generates a preprocessing list from a saved analyzer. The list can be passed to the
    `PreprocessingPipeline` class to create a preprocessing pipeline from the list.

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
    preprocessing_list : list
        The preprocessing list extracted from the analyzer's recording.
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
            preprocessing_list = get_preprocessing_list_from_file(recording_file)

    elif format == "zarr":
        backend_options = {} if backend_options is None else backend_options
        storage_options = backend_options.get("storage_options", {})
        zarr_root = super_zarr_open(str(analyzer_folder), mode="r", storage_options=storage_options)

        rec_field = zarr_root.get("recording")
        if rec_field is not None:
            recording_dict = rec_field[0]
        else:
            recording_dict = {}

        preprocessing_list = _make_pipeline_list_from_recording_dict(recording_dict)

    return preprocessing_list


def get_preprocessing_list_from_file(recording_dictionary_path):
    """
    Generates a preprocessing list, passable to `apply_preprocessing_pipeline` function and
    `PreprocessPipeline` class, from a recording dictionary.

    Only extracts preprocessing steps which can be applied "globally" to any recording.
    Hence this does not extract `ChannelSlice` and `FrameSlice` steps.

    Parameters
    ----------
    recording_dictionary_path : str or Path
        Path to the `.json` or `.pkl` output from a saved recording.

    Returns
    -------
    preprocessing_list : list
        List containing preprocessing steps and their kwargs, each element is a dict with keys "name" and "kwargs".

    """

    if str(recording_dictionary_path).endswith(".json"):
        import json

        with open(recording_dictionary_path, "r") as f:
            recording_dict = json.load(f)
    elif str(recording_dictionary_path).endswith(".pkl") or str(recording_dictionary_path).endswith(".pickle"):
        import pickle

        with open(recording_dictionary_path, "rb") as f:
            recording_dict = pickle.load(f)

    preprocessing_list = _make_pipeline_list_from_recording_dict(recording_dict)
    return preprocessing_list


def _make_pipeline_list_from_recording_dict(recording_dict):
    """
    Transforms a recording dict (created by the `dump` method of `BaseRecording`)
    into a preprocessing pipeline list.
    """

    pipeline_dict_from_file = {}
    _ = _load_pp_from_dict(recording_dict, pipeline_dict_from_file)

    preprocessing_list = []
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

        preprocessing_list.append({"name": preprocessor_function.__name__, "params": pp_kwargs})

    return preprocessing_list


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
        elif isinstance(value, BaseRecording):
            extractor_as_dict = value.to_dict()
            if name in ["recording", "parent_recording"]:
                this_level_kwargs[name] = _load_pp_from_dict(extractor_as_dict, kwargs_dict)
            else:  # this branch takes care of other arguments being a recording, e.g., `recording_to_detect`
                this_level_kwargs[name] = value
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

    all_kwargs_list = []
    for preprocessor in my_pipeline.preprocessor_list:

        preprocessor_name = preprocessor["name"].split(".")[-1]
        pp_function = my_pipeline.function_names_to_functions[preprocessor["name"].split(".")[-1]]
        signature = inspect.signature(pp_function)

        all_kwargs = {"name": preprocessor_name, "kwargs": {}}

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

                preprocessor_index = my_pipeline.preprocessor_list.index(preprocessor)
                pipeline_value = my_pipeline.preprocessor_list[preprocessor_index]["params"].get(par_name)

                if pipeline_value is None:
                    if default_value != pipeline_value:
                        pipeline_value = default_value

                all_kwargs["kwargs"][par_name] = pipeline_value

        all_kwargs_list.append(all_kwargs)
    return all_kwargs_list
