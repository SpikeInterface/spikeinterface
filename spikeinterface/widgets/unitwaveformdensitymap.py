import numpy as np
from matplotlib import pyplot as plt

from .basewidget import BaseWidget, BaseMultiWidget
from .utils import get_unit_colors
from ..toolkit import get_template_channel_sparsity


class UnitWaveformDensityMapWidget(BaseMultiWidget):
    """
    Plots unit waveforms using heat map density.
    
    Parameters
    ----------
    waveform_extractor: WaveformExtractor
    channel_ids: list
        The channel ids to display
    unit_ids: list
        List of unit ids.
    plot_templates: bool
        If True, templates are plotted over the waveforms
    radius_um: None or float
        If not None, all channels within a circle around the peak waveform will be displayed
        Incompatible with with `max_channels`
    max_channels : None or int
        If not None only max_channels are displayed per units.
        Incompatible with with `radius_um`
        
    
    set_title: bool
        Create a plot title with the unit number if True.
    plot_channels: bool
        Plot channel locations below traces, only used if channel_locs is True
    """
    def __init__(self, waveform_extractor, channel_ids=None, unit_ids=None,
            max_channels=None, radius_um=None,
            unit_colors=None,
           ncols=5, figure=None, ax=None, axes=None, 
            set_title=True):

        

        self.waveform_extractor = waveform_extractor
        self.recording = waveform_extractor.recording
        self.sorting = waveform_extractor.sorting

        if unit_ids is None:
            unit_ids = self.sorting.get_unit_ids()
        self.unit_ids = unit_ids

        if channel_ids is None:
            channel_ids = self.recording.get_channel_ids()
        self.channel_ids = channel_ids

        if unit_colors is None :
            unit_colors = get_unit_colors(self.sorting)
        self.unit_colors = unit_colors

        self.ncols = ncols
        
        if radius_um is not None:
            assert max_channels is None, 'radius_um and max_channels are mutually exclussive'
        if max_channels is not None:
            assert radius_um is None, 'radius_um and max_channels are mutually exclussive'
        
        self.radius_um = radius_um
        self.max_channels = max_channels
        
        fig, axes = plt.subplots(nrows=len(unit_ids), )
        BaseMultiWidget.__init__(self, figure=None, ax=None, axes=axes)

    def plot(self):
        we = self.waveform_extractor
        
        # channel sparsity
        if self.radius_um is not None:
            channel_inds = get_template_channel_sparsity(we, method='radius', outputs='index', radius_um=self.radius_um)
        elif self.max_channels is not None:
            channel_inds = get_template_channel_sparsity(we, method='best_channels', outputs='index', num_channels=self.max_channels)
        else:
            # all channels
            channel_inds = {unit_id: slice(None) for unit_id in self.unit_ids}
        
        # bins
        templates = we.get_all_templates(unit_ids=self.unit_ids, mode='median')
        bin_min = np.min(templates) * 1.3
        bin_max = np.max(templates) *  1.3
        bin_size = (bin_max - bin_min) / 100
        bins = np.arange(bin_min, bin_max, bin_size)
        
        for unit_index, unit_id in enumerate(self.unit_ids):
            ax = self.axes[unit_index]
            
            chan_inds = channel_inds[unit_id]
            
            wfs = we.get_waveforms(unit_id)
            wfs = wfs[:, :, chan_inds]
            
            # make histogram density
            wfs_flat = wfs.swapaxes(1,2).reshape(wfs.shape[0], -1)
            hist2d = np.zeros((wfs_flat.shape[1], bins.size))
            indexes0 = np.arange(wfs_flat.shape[1])
            
            wf_bined = np.floor((wfs_flat-bin_min)/bin_size).astype('int32')
            wf_bined = wf_bined.clip(0, bins.size-1)
            for d in wf_bined:
                hist2d[indexes0, d] += 1

            im = ax.imshow(hist2d.T, interpolation='nearest', 
                            origin='lower', aspect='auto', extent=(0, hist2d.shape[0], bin_min, bin_max), cmap='hot')
            
            # plot median
            template = templates[unit_index,:, chan_inds]
            template_flat = template.flatten()
            color = self.unit_colors[unit_id]
            ax.plot(template_flat, color=color, lw=1)
            
            # final cosmetics
            for i, chan_ind in enumerate(chan_inds):
                if i != 0:
                    ax.axvline(i * wfs.shape[1], color='w', lw=3)
                channel_id = self.recording.channel_ids[chan_ind]
                x = i * wfs.shape[1] + wfs.shape[1] //2
                y = (bin_max + bin_min) /2.
                ax.text(x, y, f'chan_id {channel_id}', color='w', ha='center', va='center')
            
            ax.set_xticks([])
            ax.set_ylabel(f'unit_id {unit_id}')


def plot_unit_waveform_density_map(*args, **kwargs):
    W = UnitWaveformDensityMapWidget(*args, **kwargs)
    W.plot()
    return W
plot_unit_waveform_density_map.__doc__ = UnitWaveformDensityMapWidget.__doc__
