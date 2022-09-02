from probeinterface import ProbeGroup
from probeinterface.plotting import plot_probe

import numpy as np
from spikeinterface.core import waveform_extractor

from ..base import to_attr
from ..unit_locations import UnitLocationsWidget
from .base_mpl import MplPlotter

from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

class UnitLocationsPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        dp = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)

        self.make_mpl_figure(**backend_kwargs)
        
        unit_locations = dp.unit_locations
        
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

        # color = np.array([dp.unit_colors[unit_id] for unit_id in dp.unit_ids])
        width = height = 10
        ellipse_kwargs = dict(width=width, height=height, lw=2)
        
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

        patches = [Ellipse((unit_locations[unit]), color=unit_colors[unit], 
                           zorder=5 if unit in dp.unit_ids else 3, 
                           alpha=0.9 if unit in dp.unit_ids else 0.5, label=labels[i],
                            **ellipse_kwargs) for i, unit in enumerate(unit_ids)]
        for p in patches:
            self.ax.add_patch(p)
            
        handles = [Line2D([0], [0], ls="", marker='o', markersize=5, markeredgewidth=2, 
                          color=unit_colors[unit]) for unit in unit_ids]
            
        self.figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.),
                           ncol=5, fancybox=True, shadow=True)



UnitLocationsPlotter.register(UnitLocationsWidget)
