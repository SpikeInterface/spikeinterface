from __future__ import annotations

import numpy as np
from typing import Union

# from probeinterface import ProbeGroup

from .base import BaseWidget, to_attr

# from .utils import get_unit_colors
from ..core.sortinganalyzer import SortingAnalyzer
from ..core.template_tools import get_dense_templates_array


class UnitProbeMapWidget(BaseWidget):
    """
    Plots unit map. Amplitude is color coded on probe contact.

    Can be static (animated=False) or animated (animated=True)

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
    unit_ids : list
        List of unit ids.
    channel_ids : list
        The channel ids to display
    animated : bool, default: False
        Animation for amplitude on time
    with_channel_ids : bool, default: False
        add channel ids text on the probe
    """

    def __init__(
        self,
        sorting_analyzer,
        unit_ids=None,
        channel_ids=None,
        animated=None,
        with_channel_ids=False,
        colorbar=True,
        backend=None,
        **backend_kwargs,
    ):
        sorting_analyzer = self.ensure_sorting_analyzer(sorting_analyzer)

        if unit_ids is None:
            unit_ids = sorting_analyzer.unit_ids
        self.unit_ids = unit_ids
        if channel_ids is None:
            channel_ids = sorting_analyzer.channel_ids
        self.channel_ids = channel_ids

        data_plot = dict(
            sorting_analyzer=sorting_analyzer,
            unit_ids=unit_ids,
            channel_ids=channel_ids,
            animated=animated,
            with_channel_ids=with_channel_ids,
            colorbar=colorbar,
        )

        BaseWidget.__init__(self, data_plot, backend=backend, **backend_kwargs)

    def plot_matplotlib(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        from .utils_matplotlib import make_mpl_figure
        from probeinterface.plotting import plot_probe

        dp = to_attr(data_plot)
        # backend_kwargs = self.update_backend_kwargs(**backend_kwargs)

        # self.make_mpl_figure(**backend_kwargs)
        if backend_kwargs.get("axes", None) is None:
            backend_kwargs["num_axes"] = len(dp.unit_ids)

        self.figure, self.axes, self.ax = make_mpl_figure(**backend_kwargs)

        sorting_analyzer = dp.sorting_analyzer
        probe = sorting_analyzer.get_probe()

        probe_shape_kwargs = dict(facecolor="w", edgecolor="k", lw=0.5, alpha=1.0)

        templates = get_dense_templates_array(sorting_analyzer, return_scaled=True)
        templates = templates[sorting_analyzer.sorting.ids_to_indices(dp.unit_ids), :, :]

        all_poly_contact = []
        for i, unit_id in enumerate(dp.unit_ids):
            ax = self.axes.flatten()[i]
            # template = we.get_template(unit_id)
            template = templates[i, :, :]
            # static
            if dp.animated:
                contacts_values = np.zeros(template.shape[1])
            else:
                contacts_values = np.max(np.abs(template), axis=0)
            text_on_contact = None
            if dp.with_channel_ids:
                text_on_contact = dp.channel_ids

            poly_contact, poly_contour = plot_probe(
                probe,
                contacts_values=contacts_values,
                ax=ax,
                probe_shape_kwargs=probe_shape_kwargs,
                text_on_contact=text_on_contact,
            )

            poly_contact.set_zorder(2)
            if poly_contour is not None:
                poly_contour.set_zorder(1)

            if dp.colorbar:
                self.figure.colorbar(poly_contact, ax=ax)

            poly_contact.set_clim(0, np.max(np.abs(template)))
            all_poly_contact.append(poly_contact)

            ax.set_title(str(unit_id))

        if dp.animated:
            num_frames = template.shape[0]

            def animate_func(frame):
                for i, unit_id in enumerate(self.unit_ids):
                    # template = we.get_template(unit_id)
                    template = templates[i, :, :]
                    contacts_values = np.abs(template[frame, :])
                    poly_contact = all_poly_contact[i]
                    poly_contact.set_array(contacts_values)
                return all_poly_contact

            from matplotlib.animation import FuncAnimation

            self.animation = FuncAnimation(self.figure, animate_func, frames=num_frames, interval=20, blit=True)
