from pathlib import Path

from spikeinterface.core import (BaseSorting, BaseSortingSegment)
from spikeinterface.core.core_tools import define_function_from_class

try:
    import tridesclous as tdc
    HAVE_TDC = True
except ImportError:
    HAVE_TDC = False


class TridesclousSortingExtractor(BaseSorting):
    """Load Tridesclous format data as a sorting extractor.

    Parameters
    ----------
    folder_path : str or Path
        Path to the Tridesclous folder.
    chan_grp : list, optional
        The channel group(s) to load.

    Returns
    -------
    extractor : TridesclousSortingExtractor
        Loaded data.
    """

    extractor_name = 'TridesclousSortingExtractor'
    installed = HAVE_TDC
    mode = 'folder'
    installation_mesg = "To use the TridesclousSortingExtractor install tridesclous: \n\n pip install tridesclous\n\n"  # error message when not installed
    name = "tridesclous"

    def __init__(self, folder_path, chan_grp=None):
        assert self.installed, self.installation_mesg
        tdc_folder = Path(folder_path)

        dataio = tdc.DataIO(str(tdc_folder))
        if chan_grp is None:
            # if chan_grp is not provided, take the first one if unique
            chan_grps = list(dataio.channel_groups.keys())
            assert len(chan_grps) == 1, 'There are several groups in the folder, specify chan_grp=...'
            chan_grp = chan_grps[0]

        catalogue = dataio.load_catalogue(name='initial', chan_grp=chan_grp)

        labels = catalogue['clusters']['cluster_label']
        labels = labels[labels >= 0]
        unit_ids = list(labels)

        sampling_frequency = dataio.sample_rate

        BaseSorting.__init__(self, sampling_frequency, unit_ids)
        for seg_num in range(dataio.nb_segment):
            # load all spike in memory (this avoid to lock the folder with memmap throug dataio
            all_spikes = dataio.get_spikes(seg_num=seg_num, chan_grp=chan_grp, i_start=None, i_stop=None).copy()
            self.add_sorting_segment(TridesclousSortingSegment(all_spikes))

        self._kwargs = {'folder_path': str(Path(folder_path).absolute()), 'chan_grp': chan_grp}
        self.extra_requirements.append('tridesclous')


class TridesclousSortingSegment(BaseSortingSegment):
    def __init__(self, all_spikes):
        BaseSortingSegment.__init__(self)
        self._all_spikes = all_spikes

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        spikes = self._all_spikes
        spikes = spikes[spikes['cluster_label'] == unit_id]
        spike_times = spikes['index']
        if start_frame is not None:
            spike_times = spike_times[spike_times >= start_frame]
        if end_frame is not None:
            spike_times = spike_times[spike_times < end_frame]
        return spike_times.copy()


read_tridesclous = define_function_from_class(source_class=TridesclousSortingExtractor, name="read_tridesclous")
