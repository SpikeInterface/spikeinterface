import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget
from matplotlib.animation import FuncAnimation

from probeinterface.plotting import plot_probe


class UnitProbeMapWidget(BaseWidget):
    """
    Plots unit map. Amplitude is color coded on probe contact.
    
    Can be static (animated=False) or animated (animated=True)

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
    unit_ids: list
        List of unit ids.
    channel_ids: list
        The channel ids to display
    animated: True/False
        animation for amplitude on time
        
    """

    def __init__(self, waveform_extractor, unit_ids=None, channel_ids=None,
                 animated=None, colorbar=True,
                 ncols=5,  axes=None):

        self.waveform_extractor = waveform_extractor
        if unit_ids is None:
            unit_ids = waveform_extractor.sorting.unit_ids
        self.unit_ids = unit_ids
        if channel_ids is None:
            channel_ids = waveform_extractor.recording.channel_ids
        self.channel_ids = channel_ids

        self.animated = animated
        self.colorbar = colorbar

        # layout
        n = len(unit_ids)
        if n < ncols:
            ncols = n
        nrows = int(np.ceil(n / ncols))
        if axes is None:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
        BaseWidget.__init__(self, None, None, axes)

    def plot(self):
        we = self.waveform_extractor
        probe = we.recording.get_probe()

        probe_shape_kwargs = dict(facecolor='w', edgecolor='k', lw=0.5, alpha=1.)

        all_poly_contact = []
        for i, unit_id in enumerate(self.unit_ids):
            ax = self.axes.flatten()[i]
            template = we.get_template(unit_id)
            # static
            if self.animated:
                contacts_values = np.zeros(template.shape[1])
            else:
                contacts_values = np.max(np.abs(template), axis=0)
            poly_contact, poly_contour = plot_probe(probe, contacts_values=contacts_values,
                                                    ax=ax, probe_shape_kwargs=probe_shape_kwargs)

            poly_contact.set_zorder(2)
            if poly_contour is not None:
                poly_contour.set_zorder(1)

            if self.colorbar:
                self.figure.colorbar(poly_contact, ax=ax)

            poly_contact.set_clim(0, np.max(np.abs(template)))
            all_poly_contact.append(poly_contact)

            ax.set_title(str(unit_id))

        if self.animated:
            num_frames = template.shape[0]

            def animate_func(frame):
                for i, unit_id in enumerate(self.unit_ids):
                    template = we.get_template(unit_id)
                    contacts_values = np.abs(template[frame, :])
                    poly_contact = all_poly_contact[i]
                    poly_contact.set_array(contacts_values)
                return all_poly_contact

            self.animation = FuncAnimation(self.figure, animate_func, frames=num_frames,
                                           interval=20, blit=True)


def plot_unit_probe_map(*args, **kwargs):
    W = UnitProbeMapWidget(*args, **kwargs)
    W.plot()
    return W


plot_unit_probe_map.__doc__ = UnitProbeMapWidget.__doc__
