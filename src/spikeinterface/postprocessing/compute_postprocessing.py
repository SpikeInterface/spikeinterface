from __future__ import annotations

from typing import Optional
from copy import deepcopy

from ..core import WaveformExtractor
from ..core.job_tools import _shared_job_kwargs_doc, fix_job_kwargs
from ..core.node_pipeline import run_node_pipeline

from .amplitude_scalings import AmplitudeScalingsCalculator
from .correlograms import CorrelogramsCalculator
from .isi import ISIHistogramsCalculator
from .noise_level import NoiseLevelsCalculator
from .principal_component import PrincipalComponentsCalculator
from .spike_amplitudes import SpikeAmplitudesCalculator
from .spike_locations import SpikeLocationsCalculator
from .template_metrics import TemplateMetricsCalculator
from .template_similarity import TemplateSimilarityCalculator
from .unit_locations import UnitLocationsCalculator


def get_default_postprocessing_params():
    return deepcopy(_default_postprocessing_params)


def get_postprocessing_list():
    return list(_postprocessing_names_to_calculators.keys())


def resolve_postprocessing_graph(pipelines):
    pass


def compute_postprocessing(
    waveform_extractor: WaveformExtractor,
    postprocessing_names: Optional[list] = None,
    postprocessing_params: Optional[dict] = None,
    verbose: bool = False,
    **job_kwargs,
):
    """
    Compute postprocessing for a given waveform extractor.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        A waveform extractor object.
    postprocessing_names: list, default=None
        List of postprocessing names.
        A list of postprocessing names can be obtained with `spikeinterface.postprocessing.get_postprocessing_list()`
        function. If None, all postprocessing steps are computed.
    postprocessing_kwargs: dict, default=None
        Dictionary of dictionary of postprocessing params of postprocessing arguments. Each en
        If None, default values are used, which can be retrieved with the
        `spikeinterface.postprocessing.get_default_postprocessing_params()` function.
    {}
    """
    job_kwargs = fix_job_kwargs(job_kwargs)
    if postprocessing_names is None:
        postprocessing_names = get_postprocessing_list()
    assert isinstance(postprocessing_names, (list, tuple)), "'postprocessing_names' must be a list or tuple"

    params = get_default_postprocessing_params()
    if postprocessing_params is not None:
        for name, param_values in postprocessing_params.items():
            assert name in postprocessing_names, f"'{name}' is not a key for postprocessing params"
            params[name].update(param_values)

    pipeline_postprocessing = []
    standard_postprocessing = []
    for name in postprocessing_names:
        assert name in get_postprocessing_list(), f"'{name}' is not a valid postprocessing name"
        if _postprocessing_names_to_calculators[name].pipeline_compatible:
            pipeline_postprocessing.append(name)
        else:
            standard_postprocessing.append(name)

    extensions = dict()
    for standard_step in standard_postprocessing:
        if verbose:
            print(f"Running {standard_step} postprocessing step")
        calculator = _postprocessing_names_to_calculators[standard_step](waveform_extractor)
        calculator.set_params(**params[standard_step])
        calculator.run()
        extensions[standard_step] = calculator

    pipeline_nodes = []
    for pipeline_step in pipeline_postprocessing:
        calculator = _postprocessing_names_to_calculators[pipeline_step](waveform_extractor)
        calculator.set_params(**params[pipeline_step])
        pipeline_nodes += calculator.get_pipeline_nodes()
        extensions[pipeline_step] = calculator

    if verbose:
        print(f"Running pipeline postprocessing steps: {pipeline_postprocessing}")

    recording = waveform_extractor.recording
    pipeline_data = run_node_pipeline(
        recording, nodes=pipeline_nodes, job_kwargs=job_kwargs, job_name="postprocessing pipeline", gather_mode="memory"
    )
    for name, data in zip(pipeline_postprocessing, pipeline_data):
        calculator = extensions[name]
        calculator._extension_data[calculator.extension_name] = data


compute_postprocessing.__doc__ = compute_postprocessing.__doc__.format(_shared_job_kwargs_doc)


_postprocessing_names_to_calculators = dict(
    amplitude_scalings=AmplitudeScalingsCalculator,
    correlograms=CorrelogramsCalculator,
    isi_histograms=ISIHistogramsCalculator,
    noise_levels=NoiseLevelsCalculator,
    principal_components=PrincipalComponentsCalculator,
    spike_amplitudes=SpikeAmplitudesCalculator,
    spike_locations=SpikeLocationsCalculator,
    template_metrics=TemplateMetricsCalculator,
    template_similarity=TemplateSimilarityCalculator,
    unit_locations=UnitLocationsCalculator,
)

# TODO: use typing / pydantic models
# TODO: centralize across module
_default_postprocessing_params = dict(
    amplitude_scalings=dict(
        sparsity=None,
        max_dense_channels=16,
        ms_before=None,
        ms_after=None,
        handle_collisions=True,
        delta_collision_ms=2,
    ),
    correlograms=dict(window_ms=50.0, bin_ms=1.0, method="auto"),
    isi_histograms=dict(window_ms=50.0, bin_ms=1.0, method="auto"),
    noise_levels=dict(),
    principal_components=dict(
        n_components=5,
        mode="by_channel_local",
        sparsity=None,
        whiten=True,
        dtype="float32",
        tmp_folder=None,
    ),
    spike_amplitudes=dict(peak_sign="neg", return_scaled=True),
    spike_locations=dict(
        ms_before=0.5,
        ms_after=0.5,
        method="center_of_mass",
        method_kwargs={},
    ),
    template_metrics=dict(
        metric_names=None,
        peak_sign="neg",
        upsampling_factor=10,
        sparsity=None,
        include_multi_channel_metrics=False,
        metrics_kwargs=None,
    ),
    template_similarity=dict(method="cosine_similarity"),
    unit_locations=dict(method="monopolar_triangulation"),
)
