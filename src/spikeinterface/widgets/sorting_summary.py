from __future__ import annotations

import numpy as np

import warnings

from .base import BaseWidget, to_attr

from .amplitudes import AmplitudesWidget
from .crosscorrelograms import CrossCorrelogramsWidget
from .template_similarity import TemplateSimilarityWidget
from .unit_locations import UnitLocationsWidget
from .unit_templates import UnitTemplatesWidget


from spikeinterface.core import SortingAnalyzer


_default_displayed_unit_properties = ["firing_rate", "num_spikes", "x", "y", "amplitude_median", "snr", "rp_violations"]


class SortingSummaryWidget(BaseWidget):
    """
    Plots spike sorting summary.
    This is the main viewer to visualize the final result with several sub view.
    This use sortingview (in a web browser) or spikeinterface-gui (with Qt).

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object
    unit_ids : list or None, default: None
        List of unit ids
    sparsity : ChannelSparsity or None, default: None
        Optional ChannelSparsity to apply
        If SortingAnalyzer is already sparse, the argument is ignored
    max_amplitudes_per_unit : int or None, default: None
        Maximum number of spikes per unit for plotting amplitudes.
        If None, all spikes are plotted
    min_similarity_for_correlograms : float, default: 0.2
        Threshold for computing pair-wise cross-correlograms. If template similarity between two units
        is below this threshold, the cross-correlogram is not computed
        (sortingview backend)
    curation : bool, default: False
        If True, manual curation is enabled
        (sortingview backend)
    label_choices : list or None, default: None
        List of labels to be added to the curation table
        (sortingview backend)
    displayed_unit_properties : list or None, default: None
        List of properties to be added to the unit table.
        These may be drawn from the sorting extractor, and, if available,
        the quality_metrics/template_metrics/unit_locations extensions of the SortingAnalyzer.
        See all properties available with sorting.get_property_keys(), and, if available,
        analyzer.get_extension("quality_metrics").get_data().columns and
        analyzer.get_extension("template_metrics").get_data().columns.
    extra_unit_properties : dict or None, default: None
        A dict with extra units properties to display.
        The key is the property name and the value must be a numpy.array.
    curation_dict : dict or None, default: None
        When curation is True, optionaly the viewer can get a previous 'curation_dict'
        to continue/check  previous curations on this analyzer.
        In this case label_definitions must be None beacuse it is already included in the curation_dict.
        (spikeinterface_gui backend)
    label_definitions : dict or None, default: None
        When curation is True, optionaly the user can provide a label_definitions dict.
        This replaces the label_choices in the curation_format.
        (spikeinterface_gui backend)
    """

    def __init__(
        self,
        sorting_analyzer: SortingAnalyzer,
        unit_ids=None,
        sparsity=None,
        max_amplitudes_per_unit=None,
        min_similarity_for_correlograms=0.2,
        curation=False,
        displayed_unit_properties=None,
        extra_unit_properties=None,
        label_choices=None,
        curation_dict=None,
        label_definitions=None,
        backend=None,
        unit_table_properties=None,
        **backend_kwargs,
    ):

        if unit_table_properties is not None:
            warnings.warn(
                "plot_sorting_summary() : unit_table_properties is deprecated, use displayed_unit_properties instead",
                category=DeprecationWarning,
                stacklevel=2,
            )
            displayed_unit_properties = unit_table_properties

        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)
        self.check_extensions(
            sorting_analyzer, ["correlograms", "spike_amplitudes", "unit_locations", "template_similarity"]
        )
        sorting = sorting_analyzer.sorting

        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()

        if curation_dict is not None and label_definitions is not None:
            raise ValueError("curation_dict and label_definitions are mutualy exclusive, they cannot be not None both")

        if displayed_unit_properties is None:
            displayed_unit_properties = list(_default_displayed_unit_properties)
        if extra_unit_properties is not None:
            displayed_unit_properties = displayed_unit_properties + list(extra_unit_properties.keys())

        data_plot = dict(
            sorting_analyzer=sorting_analyzer,
            unit_ids=unit_ids,
            sparsity=sparsity,
            min_similarity_for_correlograms=min_similarity_for_correlograms,
            displayed_unit_properties=displayed_unit_properties,
            extra_unit_properties=extra_unit_properties,
            curation=curation,
            label_choices=label_choices,
            max_amplitudes_per_unit=max_amplitudes_per_unit,
            curation_dict=curation_dict,
            label_definitions=label_definitions,
        )

        BaseWidget.__init__(self, data_plot, backend=backend, **backend_kwargs)

    def plot_sortingview(self, data_plot, **backend_kwargs):
        import sortingview.views as vv
        from .utils_sortingview import generate_unit_table_view, make_serializable, handle_display_and_url

        dp = to_attr(data_plot)
        sorting_analyzer = dp.sorting_analyzer
        unit_ids = dp.unit_ids
        sparsity = dp.sparsity
        min_similarity_for_correlograms = dp.min_similarity_for_correlograms

        unit_ids = make_serializable(dp.unit_ids)

        v_spike_amplitudes = AmplitudesWidget(
            sorting_analyzer,
            unit_ids=unit_ids,
            max_spikes_per_unit=dp.max_amplitudes_per_unit,
            hide_unit_selector=True,
            generate_url=False,
            display=False,
            backend="sortingview",
        ).view
        v_average_waveforms = UnitTemplatesWidget(
            sorting_analyzer,
            unit_ids=unit_ids,
            sparsity=sparsity,
            hide_unit_selector=True,
            generate_url=False,
            display=False,
            backend="sortingview",
        ).view
        v_cross_correlograms = CrossCorrelogramsWidget(
            sorting_analyzer,
            unit_ids=unit_ids,
            min_similarity_for_correlograms=min_similarity_for_correlograms,
            hide_unit_selector=True,
            generate_url=False,
            display=False,
            backend="sortingview",
        ).view

        v_unit_locations = UnitLocationsWidget(
            sorting_analyzer,
            unit_ids=unit_ids,
            hide_unit_selector=True,
            generate_url=False,
            display=False,
            backend="sortingview",
        ).view

        w = TemplateSimilarityWidget(
            sorting_analyzer,
            unit_ids=unit_ids,
            immediate_plot=False,
            generate_url=False,
            display=False,
            backend="sortingview",
        )
        similarity = w.data_plot["similarity"]

        # similarity
        similarity_scores = []
        for i1, u1 in enumerate(unit_ids):
            for i2, u2 in enumerate(unit_ids):
                similarity_scores.append(
                    vv.UnitSimilarityScore(unit_id1=u1, unit_id2=u2, similarity=similarity[i1, i2].astype("float32"))
                )

        # unit ids
        v_units_table = generate_unit_table_view(
            dp.sorting_analyzer,
            dp.displayed_unit_properties,
            similarity_scores=similarity_scores,
            extra_unit_properties=dp.extra_unit_properties,
        )

        if dp.curation:
            v_curation = vv.SortingCuration2(label_choices=dp.label_choices)
            v1 = vv.Splitter(direction="vertical", item1=vv.LayoutItem(v_units_table), item2=vv.LayoutItem(v_curation))
        else:
            v1 = v_units_table
        v2 = vv.Splitter(
            direction="horizontal",
            item1=vv.LayoutItem(v_unit_locations, stretch=0.2),
            item2=vv.LayoutItem(
                vv.Splitter(
                    direction="horizontal",
                    item1=vv.LayoutItem(v_average_waveforms),
                    item2=vv.LayoutItem(
                        vv.Splitter(
                            direction="vertical",
                            item1=vv.LayoutItem(v_spike_amplitudes),
                            item2=vv.LayoutItem(v_cross_correlograms),
                        )
                    ),
                )
            ),
        )

        # assemble layout
        self.view = vv.Splitter(direction="horizontal", item1=vv.LayoutItem(v1), item2=vv.LayoutItem(v2))

        self.url = handle_display_and_url(self, self.view, **backend_kwargs)

    def plot_spikeinterface_gui(self, data_plot, **backend_kwargs):
        sorting_analyzer = data_plot["sorting_analyzer"]

        from spikeinterface_gui import run_mainwindow

        run_mainwindow(
            sorting_analyzer,
            with_traces=True,
            curation=data_plot["curation"],
            curation_dict=data_plot["curation_dict"],
            label_definitions=data_plot["label_definitions"],
            extra_unit_properties=data_plot["extra_unit_properties"],
            displayed_unit_properties=data_plot["displayed_unit_properties"],
        )
