from __future__ import annotations

from probeinterface import ProbeGroup

from .base import BaseWidget, to_attr
from .utils import get_unit_colors
from spikeinterface.core.sortinganalyzer import SortingAnalyzer

from .unit_templates import UnitTemplatesWidget
from ..core import Templates


class DriftingTemplatesWidget(BaseWidget):
    """
    Plot a drifting templates object to explore motion

    Parameters
    ----------
    drifting_templates :
        A drifting templates object
    scale : float, default: 1
        Scale factor for the waveforms/templates (matplotlib backend)
    """

    def __init__(
        self,
        drifting_templates: SortingAnalyzer,
        scale=1,
        backend=None,
        **backend_kwargs,
    ):
        self.drifting_templates = drifting_templates

        data_plot = dict(
            drifting_templates=drifting_templates,
        )

        BaseWidget.__init__(self, data_plot, backend=backend, **backend_kwargs)

    def plot_ipywidgets(self, data_plot, **backend_kwargs):
        import matplotlib.pyplot as plt
        import ipywidgets.widgets as widgets
        from IPython.display import display
        from .utils_ipywidgets import check_ipywidget_backend, UnitSelector

        check_ipywidget_backend()

        # self.next_data_plot = data_plot.copy()
        self.drifting_templates = data_plot["drifting_templates"]

        cm = 1 / 2.54

        width_cm = backend_kwargs["width_cm"]
        height_cm = backend_kwargs["height_cm"]

        ratios = [0.15, 0.85]

        with plt.ioff():
            output = widgets.Output()
            with output:
                fig, self.ax = plt.subplots(figsize=((ratios[1] * width_cm) * cm, height_cm * cm))
                plt.show()

        unit_ids = self.drifting_templates.unit_ids
        self.unit_selector = UnitSelector(unit_ids)
        self.unit_selector.value = list(unit_ids)[:1]

        arr = self.drifting_templates.templates_array_moved

        self.slider = widgets.IntSlider(
            orientation="horizontal",
            value=arr.shape[0] // 2,
            min=0,
            max=arr.shape[0] - 1,
            readout=False,
            continuous_update=True,
            layout=widgets.Layout(width=f"100%"),
        )

        self.widget = widgets.AppLayout(
            center=fig.canvas,
            left_sidebar=self.unit_selector,
            pane_widths=ratios + [0],
            footer=self.slider,
        )

        self._update_ipywidget()

        self.unit_selector.observe(self._change_unit, names="value", type="change")
        self.slider.observe(self._change_displacement, names="value", type="change")

        if backend_kwargs["display"]:
            display(self.widget)

    def _change_unit(self, change=None):
        self._update_ipywidget(keep_lims=False)

    def _change_displacement(self, change=None):
        self._update_ipywidget(keep_lims=True)

    def _update_ipywidget(self, keep_lims=False):
        if keep_lims:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

        self.ax.clear()
        unit_ids = self.unit_selector.value

        displacement_index = self.slider.value

        templates_array = self.drifting_templates.templates_array_moved[displacement_index, :, :, :]
        templates = Templates(
            templates_array,
            self.drifting_templates.sampling_frequency,
            self.drifting_templates.nbefore,
            is_in_uV=self.drifting_templates.is_in_uV,
            sparsity_mask=None,
            channel_ids=self.drifting_templates.channel_ids,
            unit_ids=self.drifting_templates.unit_ids,
            probe=self.drifting_templates.probe,
        )

        UnitTemplatesWidget(
            templates, unit_ids=unit_ids, scale=5, plot_legend=False, backend="matplotlib", ax=self.ax, same_axis=True
        )

        displacement = self.drifting_templates.displacements[displacement_index]
        self.ax.set_title(f"{displacement_index}:{displacement} - untis:{unit_ids}")

        if keep_lims:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        fig = self.ax.get_figure()
        fig.canvas.draw()
        fig.canvas.flush_events()
