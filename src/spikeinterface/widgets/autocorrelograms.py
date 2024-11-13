from __future__ import annotations

from .base import BaseWidget, to_attr

from .crosscorrelograms import CrossCorrelogramsWidget


class AutoCorrelogramsWidget(CrossCorrelogramsWidget):
    # the doc is copied form CrossCorrelogramsWidget

    def __init__(self, *args, **kargs):
        _ = kargs.pop("min_similarity_for_correlograms", 0.0)
        CrossCorrelogramsWidget.__init__(
            self,
            *args,
            **kargs,
            min_similarity_for_correlograms=None,
        )

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure

        dp = to_attr(data_plot)
        backend_kwargs["num_axes"] = len(dp.unit_ids)
        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        bins = dp.bins
        unit_ids = dp.unit_ids
        correlograms = dp.correlograms
        bin_width = bins[1] - bins[0]

        for i, unit_id in enumerate(unit_ids):
            ccg = correlograms[i, i]
            ax = self.axes.flatten()[i]
            if dp.unit_colors is None:
                color = "g"
            else:
                color = dp.unit_colors[unit_id]
            ax.bar(x=bins[:-1], height=ccg, width=bin_width, color=color, align="edge")
            ax.set_title(str(unit_id))

    def plot_sortingview(self, data_plot, **backend_kwargs):
        import sortingview.views as vv
        from .utils_sortingview import make_serializable, handle_display_and_url

        dp = to_attr(data_plot)
        unit_ids = make_serializable(dp.unit_ids)

        ac_items = []
        for i in range(len(unit_ids)):
            for j in range(i, len(unit_ids)):
                if i == j:
                    ac_items.append(
                        vv.AutocorrelogramItem(
                            unit_id=unit_ids[i],
                            bin_edges_sec=(dp.bins / 1000.0).astype("float32"),
                            bin_counts=dp.correlograms[i, j].astype("int32"),
                        )
                    )

        self.view = vv.Autocorrelograms(autocorrelograms=ac_items)

        self.url = handle_display_and_url(self, self.view, **backend_kwargs)


AutoCorrelogramsWidget.__doc__ = CrossCorrelogramsWidget.__doc__
