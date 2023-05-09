from .metrics import MetricsBaseWidget
from ..core.waveform_extractor import WaveformExtractor
from ..qualitymetrics import compute_quality_metrics


class QualityMetricsWidget(MetricsBaseWidget):
    """
    Plots quality metrics distributions.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        The object to compute/get crosscorrelograms from
    unit_ids: list
        List of unit ids.
    skip_metrics: list or None
        If given, a list of quality metrics to skip
    unit_colors :  dict or None
        If given, a dictionary with unit ids as keys and colors as values
    hide_unit_selector : bool
        For sortingview backend, if True the unit selector is not displayed
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
        self.check_extensions(waveform_extractor, "quality_metrics")
        qlc = waveform_extractor.load_extension("quality_metrics")
        quality_metrics = qlc.get_data()

        sorting = waveform_extractor.sorting

        MetricsBaseWidget.__init__(self, quality_metrics, sorting,
                                   unit_ids=unit_ids, unit_colors=unit_colors,
                                   include_metrics=include_metrics, 
                                   skip_metrics=skip_metrics,
                                   hide_unit_selector=hide_unit_selector,
                                   backend=backend, **backend_kwargs)


