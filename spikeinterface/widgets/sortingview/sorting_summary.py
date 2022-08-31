from ..base import to_attr
from ..sorting_summary import SortingSummaryWidget
from .base_sortingview import SortingviewPlotter, generate_unit_table_view

from .amplitudes import AmplitudesPlotter
from .autocorrelograms import AutoCorrelogramsPlotter
from .crosscorrelograms import CrossCorrelogramsPlotter
from .template_similarity import TemplateSimilarityPlotter
from .unit_locations import UnitLocationsPlotter
from .unit_templates import UnitTemplatesPlotter


class SortingSummaryPlotter(SortingviewPlotter):
    def do_plot(self, data_plot, **backend_kwargs):
        import sortingview.views as vv
        dp = to_attr(data_plot)

        unit_ids = self.make_serializable(dp.unit_ids)

        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)

        amplitudes_plotter = AmplitudesPlotter()
        v_spike_amplitudes = amplitudes_plotter.do_plot(dp.amplitudes, generate_url=False, backend="sortingview")
        template_plotter = UnitTemplatesPlotter()
        v_average_waveforms = template_plotter.do_plot(dp.templates, generate_url=False, backend="sortingview")
        xcorrelograms_plotter = CrossCorrelogramsPlotter()
        v_cross_correlograms = xcorrelograms_plotter.do_plot(dp.correlograms, generate_url=False, backend="sortingview")
        unitlocation_plotter = UnitLocationsPlotter()
        v_unit_locations = unitlocation_plotter.do_plot(dp.unit_locations, generate_url=False, backend="sortingview")
        template_sim_plotter = TemplateSimilarityPlotter()
        v_unit_similarity = template_sim_plotter.do_plot(dp.similarity, generate_url=False, backend="sortingview")

        # unit ids
        v_units_table = generate_unit_table_view(unit_ids)

        # assemble layout
        v_summary = vv.Box(
            direction='horizontal',
            items=[
                vv.LayoutItem(v_units_table, max_size=150),
                vv.LayoutItem(vv.Splitter(
                    direction='horizontal',
                    item1=vv.LayoutItem(v_unit_locations, stretch=0.2),
                    item2=vv.LayoutItem(
                        vv.Splitter(
                            direction='horizontal',
                            item1=vv.LayoutItem(v_average_waveforms),
                            item2=vv.LayoutItem(
                                vv.Splitter(
                                    direction='vertical',
                                    item1=vv.LayoutItem(v_spike_amplitudes),
                                    item2=vv.LayoutItem(
                                        vv.Splitter(
                                            direction='horizontal',
                                            item1=vv.LayoutItem(v_cross_correlograms, stretch=2),
                                            item2=vv.LayoutItem(v_unit_similarity, stretch=2)
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                ]
            )

        self.set_view(v_summary)

        if backend_kwargs["generate_url"]:
            if backend_kwargs.get("figlabel") is None:
                label = "SpikeInterface - SortingSummary"
            url = v_summary.url(label=label)
            print(url)
        return v_summary


SortingSummaryPlotter.register(SortingSummaryWidget)
