from ..base import to_attr
from ..sorting_summary import SortingSummaryWidget
from .base_sortingview import SortingviewPlotter

from .amplitudes import AmplitudeTimeseriesPlotter
from .autocorrelograms import AutoCorrelogramsPlotter
from .crosscorrelograms import CrossCorrelogramsPlotter
from .unit_waveforms import UnitWaveformPlotter


class SortingSummaryPlotter(SortingviewPlotter):
    def do_plot(self, data_plot, **backend_kwargs):
        import sortingview.views as vv
        dp = to_attr(data_plot)
        
        unit_ids = self.make_serializable(dp.unit_ids)

        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        
        amplitudes_plotter = AmplitudeTimeseriesPlotter()
        v_spike_amplitudes = amplitudes_plotter.do_plot(dp.amplitudes, hide_unit_selector=True, 
                                                        generate_url=False, backend="sortingview")
        waveforms_plotter = UnitWaveformPlotter()
        v_average_waveforms = waveforms_plotter.do_plot(dp.waveforms, 
                                                        generate_url=False, backend="sortingview")
        xcorrelograms_plotter = CrossCorrelogramsPlotter()
        v_cross_correlograms = xcorrelograms_plotter.do_plot(dp.correlograms, hide_unit_selector=True, 
                                                             generate_url=False, backend="sortingview")
        
        acorrelograms_plotter = AutoCorrelogramsPlotter()
        v_auto_correlograms = acorrelograms_plotter.do_plot(dp.correlograms, hide_unit_selector=True, 
                                                            generate_url=False, backend="sortingview")
        
        # unit ids
        ut_rows = [
            vv.UnitsTableRow(unit_id=u, values={})
            for u in unit_ids
        ]
        ut_columns = []
        v_units_table = vv.UnitsTable(rows=ut_rows, columns=ut_columns)
        
        # similarity
        ss_items = []
        for i1, u1 in enumerate(unit_ids):
            for i2, u2 in enumerate(unit_ids):
                ss_items.append(vv.UnitSimilarityScore(
                    unit_id1=u1,
                    unit_id2=u2,
                    similarity=dp.similarity["similarity"][i1, i2].astype("float32")
                ))

        v_unit_similarity_matrix = vv.UnitSimilarityMatrix(
            unit_ids=list(unit_ids),
            similarity_scores=ss_items
        )
        
        # raise Exception

        # assemble layout
        v_summary = vv.Box(
            direction='horizontal',
            items=[
                vv.LayoutItem(v_units_table, max_size=150),
                vv.LayoutItem(
                    vv.Splitter(
                        direction='horizontal',
                        item1=vv.LayoutItem(
                            vv.Box(
                                direction='vertical',
                                items=[
                                    vv.LayoutItem(v_spike_amplitudes),
                                    vv.LayoutItem(
                                        vv.Splitter(
                                            direction='horizontal',
                                            item1=vv.LayoutItem(v_cross_correlograms, stretch=3),
                                            item2=vv.LayoutItem(v_unit_similarity_matrix, stretch=1)
                                        )
                                    )
                                ]
                            )
                        ),
                        item2=vv.LayoutItem(
                            vv.Box(
                                direction='vertical',
                                items=[
                                    vv.LayoutItem(v_average_waveforms),
                                    vv.LayoutItem(v_auto_correlograms)
                                    ]
                                )
                            )
                        )
                    )
                ]
            )

        if backend_kwargs["generate_url"]:
            if backend_kwargs.get("figlabel") is None:
                label = "SpikeInterface - SortingSummary"
            url = v_summary.url(label=label)
            print(url)
        return v_summary


SortingSummaryPlotter.register(SortingSummaryWidget)
