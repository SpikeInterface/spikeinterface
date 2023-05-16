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
    default_label = "SpikeInterface - Sorting Summary"

    def do_plot(self, data_plot, **backend_kwargs):
        import sortingview.views as vv
        dp = to_attr(data_plot)

        unit_ids = self.make_serializable(dp.unit_ids)

        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)

        amplitudes_plotter = AmplitudesPlotter()
        v_spike_amplitudes = amplitudes_plotter.do_plot(dp.amplitudes, generate_url=False, 
                                                        display=False, backend="sortingview")
        template_plotter = UnitTemplatesPlotter()
        v_average_waveforms = template_plotter.do_plot(dp.templates, generate_url=False, 
                                                       display=False, backend="sortingview")
        xcorrelograms_plotter = CrossCorrelogramsPlotter()
        v_cross_correlograms = xcorrelograms_plotter.do_plot(dp.correlograms, generate_url=False, 
                                                             display=False, backend="sortingview")
        unitlocation_plotter = UnitLocationsPlotter()
        v_unit_locations = unitlocation_plotter.do_plot(dp.unit_locations, generate_url=False, 
                                                        display=False, backend="sortingview")
        # similarity
        similarity_scores = []
        for i1, u1 in enumerate(unit_ids):
            for i2, u2 in enumerate(unit_ids):
                similarity_scores.append(vv.UnitSimilarityScore(
                    unit_id1=u1,
                    unit_id2=u2,
                    similarity=dp.similarity['similarity'][i1, i2].astype("float32")
                    ))

        # unit ids
        v_units_table = generate_unit_table_view(dp.waveform_extractor.sorting, 
                                                 dp.unit_table_properties,
                                                 similarity_scores=similarity_scores)

        if dp.curation:
            v_curation = vv.SortingCuration2(label_choices=dp.label_choices)
            v1 = vv.Splitter(
                direction='vertical',
                item1=vv.LayoutItem(v_units_table),
                item2=vv.LayoutItem(v_curation)
            )
        else:
            v1 = v_units_table
        v2 = vv.Splitter(
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
                                    item2=vv.LayoutItem(v_cross_correlograms),
                                        )
                                    )
                                )
                            )
                        )

        # assemble layout
        v_summary = vv.Splitter(
            direction='horizontal',
            item1=vv.LayoutItem(v1),
            item2=vv.LayoutItem(v2)
        )

        self.handle_display_and_url(v_summary, **backend_kwargs)
        return v_summary


SortingSummaryPlotter.register(SortingSummaryWidget)
