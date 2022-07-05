# import numpy as np

# from .base import BaseWidget, define_widget_function_from_class
# from ..core.waveform_extractor import WaveformExtractor
# from ..core.baserecording import BaseRecording
# from ..core.basesorting import BaseSorting
# from .utils import get_unit_colors
# from ..postprocessing import get_template_channel_sparsity


# class UnitAmplitudesWidget(BaseWidget):
#     possible_backends = {}
    
#     def __init__(self, waveform_extractor: WaveformExtractor, channel_ids=None, unit_ids=None,
#                  plot_waveforms=True, plot_templates=True, plot_channels=False,
#                  unit_colors=None, max_channels=None, radius_um=None,
#                  ncols=5, lw=2, axis_equal=False, unit_selected_waveforms=None,
#                  set_title=True,
                 
#                  backend=None, **backend_kwargs):
#         """
#         Plots unit waveforms.

#         Parameters
#         ----------
#         waveform_extractor: WaveformExtractor
#         channel_ids: list
#             The channel ids to display
#         unit_ids: list
#             List of unit ids.
#         plot_templates: bool
#             If True, templates are plotted over the waveforms
#         radius_um: None or float
#             If not None, all channels within a circle around the peak waveform will be displayed
#             Incompatible with with `max_channels`
#         max_channels : None or int
#             If not None only max_channels are displayed per units.
#             Incompatible with with `radius_um`
#         set_title: bool
#             Create a plot title with the unit number if True.
#         plot_channels: bool
#             Plot channel locations below traces.
#         axis_equal: bool
#             Equal aspect ratio for x and y axis, to visualize the array geometry to scale.
#         lw: float
#             Line width for the traces.
#         unit_colors: None or dict
#             A dict key is unit_id and value is any color format handled by matplotlib.
#             If None, then the get_unit_colors() is internally used.
#         unit_selected_waveforms: None or dict
#             A dict key is unit_id and value is the subset of waveforms indices that should be 
#             be displayed
#         show_all_channels: bool
#             Show the whole probe if True, or only selected channels if False
#             The axis to be used. If not given an axis is created
#         """

#         # self.waveform_extractor = waveform_extractor
#         # self._recording = we.recording
#         # self._sorting = we.sorting

#         we = waveform_extractor
#         recording: BaseRecording = we.recording
#         sorting: BaseSorting = we.sorting

#         if unit_ids is None:
#             unit_ids = sorting.get_unit_ids()
#         unit_ids = unit_ids
#         if channel_ids is None:
#             channel_ids = recording.get_channel_ids()

#         if unit_colors is None:
#             unit_colors = get_unit_colors(sorting)

#         # self.ncols = ncols
#         # self._plot_waveforms = plot_waveforms
#         # self._plot_templates = plot_templates
#         # self._plot_channels = plot_channels

#         if radius_um is not None:
#             assert max_channels is None, 'radius_um and max_channels are mutually exclusive'
#         if max_channels is not None:
#             assert radius_um is None, 'radius_um and max_channels are mutually exclusive'

#         num_axes = len(unit_ids)
        
#         templates = we.get_all_templates(unit_ids=unit_ids)
#         channel_locations = recording.get_channel_locations(channel_ids=channel_ids)

#         xvectors, y_scale, y_offset = get_waveforms_scales(waveform_extractor, templates, channel_locations)

#         if radius_um is not None:
#             channel_inds = get_template_channel_sparsity(we, method='radius', outputs='index', radius_um=radius_um)
#         elif max_channels is not None:
#             channel_inds = get_template_channel_sparsity(we, method='best_channels', outputs='index', num_channels=max_channels)
#         else:
#             # all channels
#             channel_inds = {unit_id: np.arange(recording.get_num_channels()) for unit_id in unit_ids}
        
#         wfs_by_ids = {unit_id: we.get_waveforms(unit_id) for unit_id in unit_ids}

#         plot_data = dict(
#             sampling_frequency=recording.get_sampling_frequency(),
#             unit_ids=unit_ids,
#             channel_ids=channel_ids,
#             unit_colors=unit_colors,
#             channel_locations=channel_locations,
#             templates=templates,
#             plot_waveforms=plot_waveforms,
#             plot_templates=plot_templates,
#             plot_channels=plot_channels,
#             ncols=ncols,
#             radius_um=radius_um,
#             max_channels=max_channels,
#             unit_selected_waveforms=unit_selected_waveforms,
#             axis_equal=axis_equal,
#             lw=lw,
#             xvectors=xvectors,
#             y_scale=y_scale,
#             y_offset=y_offset,
#             channel_inds=channel_inds,
#             num_axes=num_axes,
#             wfs_by_ids=wfs_by_ids,
#             set_title=set_title,
#         )

#         BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)


# import numpy as np
# from matplotlib import pyplot as plt

# from .base import BaseWidget

# from ..postprocessing import compute_spike_amplitudes
# from .utils import get_unit_colors


# class AmplitudeBaseWidget(BaseWidget):
#     def __init__(self, waveform_extractor, unit_ids=None, 
#                  compute_kwargs={}, unit_colors=None, figure=None, ax=None):
#         BaseWidget.__init__(self, figure, ax)

#         self.we = waveform_extractor
        
#         if self.we.is_extension('spike_amplitudes'):
#             sac = self.we.load_extension('spike_amplitudes')
#             self.amplitudes = sac.get_amplitudes(outputs='by_unit')
#         else:
#             self.amplitudes = compute_spike_amplitudes(self.we, outputs='by_unit', **compute_kwargs)
        
#         if unit_ids is None:
#             unit_ids = waveform_extractor.sorting.unit_ids
#         self.unit_ids = unit_ids

#         if unit_colors is None:
#             unit_colors = get_unit_colors(self.we.sorting)
#         self.unit_colors = unit_colors

#     def plot(self):
#         self._do_plot()


# class AmplitudeTimeseriesWidget(AmplitudeBaseWidget):
#     possible_backends = {}
#     possible_backends_kwargs = {}

#     """
#     Plots waveform amplitudes distribution.

#     Parameters
#     ----------
#     waveform_extractor: WaveformExtractor

#     amplitudes: None or pre computed amplitudes
#         If None then amplitudes are recomputed
#     peak_sign: 'neg', 'pos', 'both'
#         In case of recomputing amplitudes.

#     Returns
#     -------
#     W: AmplitudeDistributionWidget
#         The output widget
#     """

#     def _do_plot(self):
#         sorting = self.we.sorting
#         # ~ unit_ids = sorting.unit_ids
#         num_seg = sorting.get_num_segments()
#         fs = sorting.get_sampling_frequency()

#         # TODO handle segment
#         ax = self.ax
#         for i, unit_id in enumerate(self.unit_ids):
#             for segment_index in range(num_seg):
#                 times = sorting.get_unit_spike_train(unit_id, segment_index=segment_index)
#                 times = times / fs
#                 amps = self.amplitudes[segment_index][unit_id]
#                 ax.scatter(times, amps, color=self.unit_colors[unit_id], s=3, alpha=1)

#                 if i == 0:
#                     ax.set_title(f'segment {segment_index}')
#                 if i == len(self.unit_ids) - 1:
#                     ax.set_xlabel('Times [s]')
#                 if segment_index == 0:
#                     ax.set_ylabel(f'Amplitude')

#         ylims = ax.get_ylim()
#         if np.max(ylims) < 0:
#             ax.set_ylim(min(ylims), 0)
#         if np.min(ylims) > 0:
#             ax.set_ylim(0, max(ylims))


#~ class AmplitudeDistributionWidget(AmplitudeBaseWidget):
    #~ """
    #~ Plots waveform amplitudes distribution.

    #~ Parameters
    #~ ----------
    #~ waveform_extractor: WaveformExtractor

    #~ amplitudes: None or pre computed amplitudes
        #~ If None then amplitudes are recomputed
    #~ peak_sign: 'neg', 'pos', 'both'
        #~ In case of recomputing amplitudes.

    #~ Returns
    #~ -------
    #~ W: AmplitudeDistributionWidget
        #~ The output widget
    #~ """

    #~ def _do_plot(self):
        #~ sorting = self.we.sorting
        #~ unit_ids = sorting.unit_ids
        #~ num_seg = sorting.get_num_segments()

        #~ ax = self.ax
        #~ unit_amps = []
        #~ for i, unit_id in enumerate(unit_ids):
            #~ amps = []
            #~ for segment_index in range(num_seg):
                #~ amps.append(self.amplitudes[segment_index][unit_id])
            #~ amps = np.concatenate(amps)
            #~ unit_amps.append(amps)
        #~ parts = ax.violinplot(unit_amps, showmeans=False, showmedians=False, showextrema=False)

        #~ for i, pc in enumerate(parts['bodies']):
            #~ color = self.unit_colors[unit_ids[i]]
            #~ pc.set_facecolor(color)
            #~ pc.set_edgecolor('black')
            #~ pc.set_alpha(1)

        #~ ax.set_xticks(np.arange(len(unit_ids)) + 1)
        #~ ax.set_xticklabels([str(unit_id) for unit_id in unit_ids])

        #~ ylims = ax.get_ylim()
        #~ if np.max(ylims) < 0:
            #~ ax.set_ylim(min(ylims), 0)
        #~ if np.min(ylims) > 0:
            #~ ax.set_ylim(0, max(ylims))


# def plot_amplitudes_timeseries(*args, **kwargs):
#     W = AmplitudeTimeseriesWidget(*args, **kwargs)
#     W.plot()
#     return W


# plot_amplitudes_timeseries.__doc__ = AmplitudeTimeseriesWidget.__doc__


#~ def plot_amplitudes_distribution(*args, **kwargs):
    #~ W = AmplitudeDistributionWidget(*args, **kwargs)
    #~ W.plot()
    #~ return W


#~ plot_amplitudes_distribution.__doc__ = AmplitudeDistributionWidget.__doc__
