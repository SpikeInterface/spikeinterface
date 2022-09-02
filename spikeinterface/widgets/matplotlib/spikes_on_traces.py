import numpy as np

from ..base import to_attr
from ..spikes_on_traces import SpikesOnTracesWidget
from .base_mpl import MplPlotter
from .timeseries import TimeseriesPlotter

from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D


class SpikesOnTracesPlotter(MplPlotter):
    
    def do_plot(self, data_plot, **backend_kwargs):
        dp = to_attr(data_plot)

        # first plot time series
        tsplotter = TimeseriesPlotter()
        data_plot["timeseries"]["add_legend"] = False
        tsplotter.do_plot(dp.timeseries, **backend_kwargs)
        self.ax = tsplotter.ax
        self.axes = tsplotter.axes
        self.figure = tsplotter.figure
        
        ax = self.ax
        
        we = dp.waveform_extractor
        sorting = dp.waveform_extractor.sorting
        frame_range = dp.timeseries["frame_range"]
        segment_index = dp.timeseries["segment_index"]
        min_y = np.min(dp.timeseries["channel_locations"][:, 1])
        max_y = np.max(dp.timeseries["channel_locations"][:, 1])
        
        n = len(dp.timeseries["channel_ids"])
        order = dp.timeseries["order"]
        if order is None:
            order = np.arange(n)
        
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        
        # loop through units and plot a scatter of spikes at estimated location
        handles = []
        labels = []

        for unit in dp.unit_ids:
            spike_frames = sorting.get_unit_spike_train(unit, segment_index=segment_index)
            spike_start, spike_end = np.searchsorted(spike_frames, frame_range)
            
            spike_frames_to_plot = spike_frames[spike_start:spike_end]
            
            if dp.timeseries["mode"] == "map":
                spike_times_to_plot = sorting.get_unit_spike_train(unit, segment_index=segment_index, 
                                                                   return_times=True)[spike_start:spike_end]
                unit_y_loc = min_y + max_y - dp.unit_locations[unit][1]
                # markers = np.ones_like(spike_frames_to_plot) * (min_y + max_y - dp.unit_locations[unit][1])
                width = 2 * 1e-3
                ellipse_kwargs = dict(width=width, height=10, fc='none', ec=dp.unit_colors[unit], lw=2)
                patches = [Ellipse((s, unit_y_loc), **ellipse_kwargs) for s in spike_times_to_plot]
                for p in patches:
                    ax.add_patch(p)
                handles.append(Line2D([0], [0], ls="", marker='o', markersize=5, markeredgewidth=2, 
                                      markeredgecolor=dp.unit_colors[unit], markerfacecolor='none'))
                labels.append(unit)
            else:
                # construct waveforms
                label_set = False
                if len(spike_frames_to_plot) > 0:
                    vspacing = dp.timeseries["vspacing"]
                    traces = dp.timeseries["list_traces"][0]
                    waveform_idxs = spike_frames_to_plot[:, None] + np.arange(-we.nbefore, we.nafter) - frame_range[0]
                    
                    times = dp.timeseries["times"][waveform_idxs]
                    # discontinuity
                    times[:, -1] = np.nan
                    times_r = times.reshape(times.shape[0] * times.shape[1])
                    waveforms = traces[waveform_idxs] #[:, :, order]
                    waveforms_r = waveforms.reshape((waveforms.shape[0] * waveforms.shape[1], waveforms.shape[2]))
                    
                    for i, chan_id in enumerate(dp.timeseries["channel_ids"]):
                        offset = vspacing * (n - 1 - i)
                        if chan_id in dp.sparsity[unit]:
                            l = ax.plot(times_r, offset + waveforms_r[:, i], color=dp.unit_colors[unit])
                            if not label_set:
                                handles.append(l[0])
                                labels.append(unit)
                                label_set = True
        ax.legend(handles, labels)  

SpikesOnTracesPlotter.register(SpikesOnTracesWidget)
