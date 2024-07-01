from __future__ import annotations

import numpy as np

from .base import BaseWidget, to_attr

from .amplitudes import AmplitudesWidget
from .crosscorrelograms import CrossCorrelogramsWidget
from .template_similarity import TemplateSimilarityWidget
from .unit_locations import UnitLocationsWidget
from .unit_templates import UnitTemplatesWidget


from ..core import SortingAnalyzer


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
    unit_table_properties : list or None, default: None
        List of properties to be added to the unit table
        (sortingview backend)
    """

    def __init__(
        self,
        sorting_analyzer: SortingAnalyzer,
        unit_ids=None,
        sparsity=None,
        max_amplitudes_per_unit=None,
        min_similarity_for_correlograms=0.2,
        curation=False,
        unit_table_properties=None,
        label_choices=None,
        backend=None,
        **backend_kwargs,
    ):
        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)
        self.check_extensions(
            sorting_analyzer, ["correlograms", "spike_amplitudes", "unit_locations", "template_similarity"]
        )
        sorting = sorting_analyzer.sorting

        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()

        plot_data = dict(
            sorting_analyzer=sorting_analyzer,
            unit_ids=unit_ids,
            sparsity=sparsity,
            min_similarity_for_correlograms=min_similarity_for_correlograms,
            unit_table_properties=unit_table_properties,
            curation=curation,
            label_choices=label_choices,
            max_amplitudes_per_unit=max_amplitudes_per_unit,
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

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
            dp.sorting_analyzer.sorting, dp.unit_table_properties, similarity_scores=similarity_scores
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

        import spikeinterface_gui

        app = spikeinterface_gui.mkQApp()
        win = spikeinterface_gui.MainWindow(sorting_analyzer, curation=data_plot["curation"])
        win.show()
        app.exec_()
