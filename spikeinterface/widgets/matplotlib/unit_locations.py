from probeinterface import ProbeGroup
from probeinterface.plotting import plot_probe

import numpy as np

from ..base import to_attr
from ..unit_locations import UnitLocationsWidget
from .base_mpl import MplPlotter


class UnitLocationsPlotter(MplPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        dp = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)
        backend_kwargs["num_axes"] = 1

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

        color = np.array([dp.unit_colors[unit_id] for unit_id in dp.unit_ids])
        loc = self.ax.scatter(unit_locations[:, 0], unit_locations[:, 1], marker='1', color=color, s=80, lw=3)
        loc.set_zorder(3)



UnitLocationsPlotter.register(UnitLocationsWidget)
