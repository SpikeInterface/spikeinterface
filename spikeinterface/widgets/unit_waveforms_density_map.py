import numpy as np

from .base import BaseWidget
from .utils import get_unit_colors

from ..core import ChannelSparsity, get_template_extremum_channel


class UnitWaveformDensityMapWidget(BaseWidget):
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
    sparsity : ChannelSparsity or None
        Optional ChannelSparsity to apply.
        If WaveformExtractor is already sparse, the argument is ignored
    use_max_channel: bool default False
        Use only the max channel
    peak_sign: str "neg"
        Used to detect max channel only when use_max_channel=True 
    unit_colors: None or dict
        A dict key is unit_id and value is any color format handled by matplotlib.
        If None, then the get_unit_colors() is internally used.
    same_axis: bool
        If True then all density are plot on the same axis and then channels is the union
        all channel per units.
    set_title: bool
        Create a plot title with the unit number if True.
    plot_channels: bool
        Plot channel locations below traces, only used if channel_locs is True
    """
    possible_backends = {}

    
    def __init__(self, waveform_extractor, channel_ids=None, unit_ids=None,
                 sparsity=None, same_axis=False, use_max_channel=False, peak_sign="neg", unit_colors=None,
                 backend=None, **backend_kwargs):
        we = waveform_extractor

        if channel_ids is None:
            channel_ids = we.channel_ids

        if unit_ids is None:
            unit_ids = we.unit_ids

        if unit_colors is None:
            unit_colors = get_unit_colors(we.sorting)

        if use_max_channel:
            assert len(unit_ids) == 1, ' UnitWaveformDensity : use_max_channel=True works only with one unit'
            max_channels = get_template_extremum_channel(we,  mode="extremum", peak_sign=peak_sign, outputs="index")


        # sparsity is done on all the units even if unit_ids is a few ones because some backend need then all
        if waveform_extractor.is_sparse():
            assert sparsity is None, 'UnitWaveformDensity WaveformExtractor is already sparse'
            used_sparsity = waveform_extractor.sparsity
        elif sparsity is not None:
            assert isinstance(sparsity, ChannelSparsity), "'sparsity' should be a ChannelSparsity object!"
            used_sparsity = sparsity
        else:
            # in this case, we construct a dense sparsity
            used_sparsity = ChannelSparsity.create_dense(we)

        channel_inds = used_sparsity.unit_id_to_channel_indices

        # bins
        templates = we.get_all_templates(unit_ids=unit_ids)
        bin_min = np.min(templates) * 1.3
        bin_max = np.max(templates) * 1.3
        bin_size = (bin_max - bin_min) / 100
        bins = np.arange(bin_min, bin_max, bin_size)

        # 2d histograms
        if same_axis:
            all_hist2d = None
            # channel union across units
            unit_inds = we.sorting.ids_to_indices(unit_ids)
            shared_chan_inds, = np.nonzero(np.sum(used_sparsity.mask[unit_inds, :], axis=0))
        else:
            all_hist2d = {}

        for unit_index, unit_id in enumerate(unit_ids):
            chan_inds = channel_inds[unit_id]
            
            # this have already the sparsity
            wfs = we.get_waveforms(unit_id, sparsity=sparsity)

            if use_max_channel:
                chan_ind = max_channels[unit_id]
                wfs = wfs[:, :, chan_inds == chan_ind]
            
            if same_axis and not np.array_equal(chan_inds, shared_chan_inds):
                # add more channels if necessary
                wfs_ = np.zeros((wfs.shape[0], wfs.shape[1], shared_chan_inds.size), dtype=float)
                mask = np.in1d(shared_chan_inds, chan_inds)
                wfs_[:, :, mask] = wfs
                wfs_[:, :, ~mask] = np.nan
                wfs = wfs_

            # make histogram density
            wfs_flat = wfs.swapaxes(1, 2).reshape(wfs.shape[0], -1)
            hist2d = np.zeros((wfs_flat.shape[1], bins.size))
            indexes0 = np.arange(wfs_flat.shape[1])

            wf_bined = np.floor((wfs_flat - bin_min) / bin_size).astype('int32')
            wf_bined = wf_bined.clip(0, bins.size - 1)
            for d in wf_bined:
                hist2d[indexes0, d] += 1

            if same_axis:
                if all_hist2d is None:
                    all_hist2d = hist2d
                else:
                    all_hist2d += hist2d
            else:
                all_hist2d[unit_id] = hist2d

        # update final channel_inds
        if same_axis:
            channel_inds = {unit_id: shared_chan_inds for unit_id in unit_ids}
        if use_max_channel:
            channel_inds = {unit_id: [max_channels[unit_id]] for unit_id in unit_ids}


        # plot median
        templates_flat = {}
        for unit_index, unit_id in enumerate(unit_ids):
            chan_inds = channel_inds[unit_id]
            template = templates[unit_index, :, chan_inds]
            template_flat = template.flatten()
            templates_flat[unit_id] = template_flat


        plot_data = dict(
            unit_ids=unit_ids,
            unit_colors=unit_colors,
            channel_ids=we.channel_ids,
            channel_inds=channel_inds,
            same_axis=same_axis,
            bin_min=bin_min,
            bin_max=bin_max,
            all_hist2d=all_hist2d,
            templates_flat=templates_flat,
            template_width=wfs.shape[1]
        )

        BaseWidget.__init__(self, plot_data, backend=backend, **backend_kwargs)

