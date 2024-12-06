import re
import json
import inspect
from copy import deepcopy
from spikeinterface.core.core_tools import is_dict_extractor
from spikeinterface.preprocessing.preprocessinglist import pp_function_to_class, preprocesser_dict, pp_name_to_function


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
    >>> my_pipeline.apply_to(recording)

    """

    def __init__(self, preprocessor_list):
        for preprocessor in preprocessor_list:
            assert _is_genuine_preprocessor(
                preprocessor
            ), f"'{preprocessor}' is not a preprocessing step in spikeinterface. To see the full list run:\n\t>>> from spikeinterface.preprocessing import pp_function_to_class\n\t>>> print(pp_function_to_class.keys())"

        self.preprocessor_list = preprocessor_list

    def __repr__(self):
        txt = "PreprocessingPipeline: \tRaw Recording \u2192 "
        for preprocessor in self.preprocessor_list:
            txt += str(preprocessor) + " \u2192 "
        txt += "Preprocessed Recording"
        return txt

    def _repr_html_(self):

        all_kwargs = _get_all_kwargs_and_values(self)

        num_titles = len(all_kwargs) + 3
        colors = [
            [255 - (a * 15) // num_titles, 255 - (a * 82) // num_titles, 255 - (a * 30) // num_titles]
            for a in range(0, num_titles + 1)
        ]

        html_text = "<div style='width: 260px; text-align: center'>"

        html_text += f"<div style='font-size: 20px; border: 1px solid; background-color: rgb({colors[1][0]}, {colors[1][1]}, {colors[1][2]});'><strong>Raw Recording</strong></div>"

        html_text += "<div style='margin: auto'>&#x2193;</div>"

        for a, (preprocessor, kwargs) in enumerate(all_kwargs.items()):
            html_text += "<details style='width: 240px; border: 1px solid; text-align: center; margin: auto;'>"
            html_text += f"<summary style='background-color: rgb({colors[a+2][0]}, {colors[a+2][1]}, {colors[a+2][2]});'><strong>{preprocessor}</strong></summary>"
            for kwarg, value in kwargs.items():
                html_text += f"{kwarg}: {value}<br />"
            html_text += "</details>"
            html_text += """<p style="margin: auto; text-align:center;">&#x2193;</p>"""

        html_text += f"<div style='font-size: 20px; border: 1px solid; background-color: rgb({colors[num_titles][0]}, {colors[num_titles][1]}, {colors[num_titles][2]});'><strong>Preprocessed Recording</strong></div>"

        html_text += "</div>"

        return html_text

    def apply_to(self, recording):
        """
        Creates a preprocessed recording by applying the PreprocessingPipeline to
        `recording`.

        Parameters
        ----------
        recording : RecordingExtractor
            The initial recording

        Returns
        -------
        preprocessed_recording : RecordingExtractor
            Preprocessed recording

        """

        preprocessor_list = self.preprocessor_list

        for preprocessor, kwargs in preprocessor_list.items():

            kwargs.pop("recording", kwargs)
            kwargs.pop("parent_recording", kwargs)

            using_class_name = bool(re.search("Recording", preprocessor))
            if using_class_name is True:
                pp_output = preprocesser_dict[preprocessor.split(".")[-1]](recording, **kwargs)
            else:
                pp_output = pp_name_to_function[preprocessor.split(".")[-1]](recording, **kwargs)

            if preprocessor == "motion_correct":
                pp_output = pp_output[0]

            recording = pp_output

        return recording


def create_preprocessed(recording=None, preprocessor_dict=None):
    """
    Creates a preprocessed recording by applying the preprocessing steps in
    `preprocessor_dict` to `recording`.

    Parameters
    ----------
    recording : RecordingExtractor
        The initial recording
    preprocessor_dict : dict
        Dictionary containing preprocessing steps and their kwargs

    Returns
    -------
    preprocessed_recording : RecordingExtractor
        Preprocessed recording

    Examples
    --------
    Create a preprocessed recording from a generated recording and a preprocessor_dict

    >>> from spikeinterface.preprocessing import create_preprocessed
    >>> from spikeinterface.generation import generate_recording
    >>> rec = generate_recording()
    >>> preprocessor_dict = {'bandpass_filter': {'freq_max': 3000}, 'common_reference': {}}
    >>> preprocessed_rec = create_preprocessed(rec, preprocessor_dict)


    """

    pipeline = PreprocessingPipeline(preprocessor_dict)
    preprocessed_recording = pipeline.apply_to(recording)
    return preprocessed_recording


def get_preprocessing_dict_from_json(recording_json_path):
    """
    Generates a preprocessing dict, passable to `create_preprocessed` function and
    `PreprocessPipline` class, from a `recording.json` provenance file.

    Only extracts preprocessing steps which can be applied "globally" to any recording.
    Hence this does not extract `ChannelSlice` and `FrameSlice` steps. To see the
    supported list of preprocessors run
    >>> from spikeinterface.preprocessing import pp_function_to_class
    >>> print(pp_function_to_class.keys()


    Parameters
    ----------
    recording_json_path : str or Path
        Path to the `recording.json` file

    Returns
    -------
    preprocessor_dict : dict
        Dictionary containing preprocessing steps and their kwargs

    """
    recording_json = json.load(open(recording_json_path))

    initial_preprocessor_dict = {}
    _load_pp_from_dict(recording_json, initial_preprocessor_dict)

    preprocessor_dict = deepcopy(initial_preprocessor_dict)
    for preprocessor in initial_preprocessor_dict:
        preprocessor_name = preprocessor.split(".")[-1]

        if not _is_genuine_preprocessor(preprocessor_name):
            preprocessor_dict.pop(preprocessor, preprocessor_dict)
            continue

        # remove recording details
        preprocessor_dict[preprocessor].pop("recording", preprocessor_dict[preprocessor])
        preprocessor_dict[preprocessor].pop("parent_recording", preprocessor_dict[preprocessor])

        # rename keys to be the class names
        preprocessor_dict[preprocessor_name] = preprocessor_dict[preprocessor]
        preprocessor_dict.pop(preprocessor)

    preprocessor_dict = dict(reversed(preprocessor_dict.items()))

    return preprocessor_dict


def _is_genuine_preprocessor(preprocessor):
    """
    Check is string 'preprocessor' is in the list of preprocessors from
    `pp_function_to_class`.
    """

    using_class_name = bool(re.search("Recording", preprocessor))
    if using_class_name:
        genuine_preprocessor = preprocessor in preprocesser_dict.keys()
    else:
        genuine_preprocessor = preprocessor in pp_name_to_function.keys()

    return genuine_preprocessor


def _load_pp_from_dict(prov_dict, kwargs_dict):
    """
    Recursive function used to iterate through recording provenance dictionary, and
    extract preprocessing steps and their kwargs.
    """
    new_kwargs = dict()
    transform_dict_to_extractor = lambda x: _load_pp_from_dict(x) if is_dict_extractor(x) else x
    for name, value in prov_dict["kwargs"].items():
        if is_dict_extractor(value):
            new_kwargs[name] = _load_pp_from_dict(value, kwargs_dict)
        elif isinstance(value, dict):
            new_kwargs[name] = {k: transform_dict_to_extractor(v) for k, v in value.items()}
        elif isinstance(value, list):
            new_kwargs[name] = [transform_dict_to_extractor(e) for e in value]
        else:
            new_kwargs[name] = value

    kwargs_dict[prov_dict["class"]] = new_kwargs
    return new_kwargs


def _get_all_kwargs_and_values(my_pipeline):

    all_kwargs = {}
    for preprocessor in my_pipeline.preprocessor_list:

        preprocessor_name = preprocessor.split(".")[-1]
        pp_function = pp_name_to_function[preprocessor.split(".")[-1]]
        signature = inspect.signature(pp_function)

        all_kwargs[preprocessor_name] = {}

        for parameter, value in signature.parameters.items():
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

                pipeline_value = my_pipeline.preprocessor_list[preprocessor].get(par_name)

                if pipeline_value is None:
                    if default_value != pipeline_value:
                        pipeline_value = default_value

                all_kwargs[preprocessor_name][par_name] = pipeline_value

    return all_kwargs
