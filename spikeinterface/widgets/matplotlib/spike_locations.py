from probeinterface import ProbeGroup
from probeinterface.plotting import plot_probe

import numpy as np

from ..base import to_attr
from ..spike_locations import SpikeLocationsWidget, estimate_axis_lims
from .base_mpl import MplPlotter

from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D


class SpikeLocationsPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        dp = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)

        self.make_mpl_figure(**backend_kwargs)
        
        spike_locations = dp.spike_locations
        
        probegroup = ProbeGroup.from_dict(dp.probegroup_dict)
        probe_shape_kwargs = dict(facecolor='w', edgecolor='k', lw=0.5, alpha=1.)
        contacts_kargs = dict(alpha=1., edgecolor='k', lw=0.5)
        
        for probe in probegroup.probes:

            text_on_contact = None
            if dp.with_channel_ids:
                text_on_contact = dp.channel_ids
            
            poly_contact, poly_contour = plot_probe(probe, ax=self.ax,
                                                    contacts_colors='w', contacts_kargs=contacts_kargs,
                                                    probe_shape_kwargs=probe_shape_kwargs,
                                                    text_on_contact=text_on_contact)
            poly_contact.set_zorder(2)
            if poly_contour is not None:
                poly_contour.set_zorder(1)

        self.ax.set_title('')
        
        if dp.plot_all_units:
            unit_colors = {}
            unit_ids = dp.all_unit_ids
            for unit in dp.all_unit_ids:
                if unit not in dp.unit_ids:
                    unit_colors[unit] = "gray"
                else:
                    unit_colors[unit] = dp.unit_colors[unit]
        else:
            unit_ids = dp.unit_ids
            unit_colors = dp.unit_colors
        labels = unit_ids

        for i, unit in enumerate(unit_ids):
            locs = spike_locations[unit]
            
            zorder = 5 if unit in dp.unit_ids else 3
            self.ax.scatter(locs["x"], locs["y"], s=2, alpha=0.3, color=unit_colors[unit],
                            label=labels[i], zorder=zorder)
            
        handles = [Line2D([0], [0], ls="", marker='o', markersize=5, markeredgewidth=2, 
                          color=unit_colors[unit]) for unit in unit_ids]
            
        self.figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.),
                           ncol=5, fancybox=True, shadow=True)

        # set proper axis limits
        xlims, ylims = estimate_axis_lims(spike_locations)
        
        ax_xlims = list(self.ax.get_xlim())
        ax_ylims = list(self.ax.get_ylim())
        
        ax_xlims[0] = xlims[0] if xlims[0] < ax_xlims[0] else ax_xlims[0]
        ax_xlims[1] = xlims[1] if xlims[1] > ax_xlims[1] else ax_xlims[1]
        ax_ylims[0] = ylims[0] if ylims[0] < ax_ylims[0] else ax_ylims[0]
        ax_ylims[1] = ylims[1] if ylims[1] > ax_ylims[1] else ax_ylims[1]
        
        self.ax.set_xlim(ax_xlims)
        self.ax.set_ylim(ax_ylims)



SpikeLocationsPlotter.register(SpikeLocationsWidget)
