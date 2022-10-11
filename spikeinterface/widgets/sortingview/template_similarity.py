from ..base import to_attr
from ..template_similarity import TemplateSimilarityWidget
from .base_sortingview import SortingviewPlotter


class TemplateSimilarityPlotter(SortingviewPlotter):
    default_label = "SpikeInterface - Template Similarity"

    def do_plot(self, data_plot, **backend_kwargs):
        import sortingview.views as vv

        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        dp = to_attr(data_plot)

        # ensure serializable for sortingview
        unit_ids = self.make_serializable(dp.unit_ids)

        # similarity
        ss_items = []
        for i1, u1 in enumerate(unit_ids):
            for i2, u2 in enumerate(unit_ids):
                ss_items.append(vv.UnitSimilarityScore(
                    unit_id1=u1,
                    unit_id2=u2,
                    similarity=dp.similarity[i1, i2].astype("float32")
                ))

        view = vv.UnitSimilarityMatrix(
            unit_ids=list(unit_ids),
            similarity_scores=ss_items
        )

        self.handle_display_and_url(view, **backend_kwargs)
        return view


TemplateSimilarityPlotter.register(TemplateSimilarityWidget)
