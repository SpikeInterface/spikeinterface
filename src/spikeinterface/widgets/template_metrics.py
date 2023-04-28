from .metrics import MetricsBaseWidget
from ..core.waveform_extractor import WaveformExtractor


class TemplateMetricsWidget(MetricsBaseWidget):
    """
    Plots template metrics distributions.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The object to compute/get crosscorrelograms from
    unit_ids : list
        List of unit ids.
    include_metrics : list
        If given list of quality metrics to include, default None
    skip_metrics : list or None
        If given, a list of quality metrics to skip, default None
    compute_kwargs : dict or None
        If given, dictionary with keyword arguments for "compute_template_metrics" function, default None
    unit_colors :  dict or None
        If given, a dictionary with unit ids as keys and colors as values, default None
    hide_unit_selector : bool
        For sortingview backend, if True the unit selector is not displayed, default False
    backend : None or str
        Three possible options:
        * 'matplotlib': uses matplotlib backend
        * 'ipywidgets': can only be used in Jupyter notebooks/Jupyter lab
        * 'sortingview': for web-based GUIs
        Default is None which uses the matplotlib backend
    """

    possible_backends = {}

    def __init__(
        self,
        waveform_extractor: WaveformExtractor,
        unit_ids=None,
        include_metrics=None,
        skip_metrics=None,
        unit_colors=None,
        hide_unit_selector=False,
        backend=None,
        **backend_kwargs
    ):
        self.check_extensions(waveform_extractor, "template_metrics")
        tmc = waveform_extractor.load_extension("template_metrics")
        template_metrics = tmc.get_data()

        sorting = waveform_extractor.sorting

        MetricsBaseWidget.__init__(self, template_metrics, sorting,
                                   unit_ids=unit_ids, unit_colors=unit_colors,
                                   include_metrics=include_metrics, 
                                   skip_metrics=skip_metrics,
                                   hide_unit_selector=hide_unit_selector,
                                   backend=backend, **backend_kwargs)


