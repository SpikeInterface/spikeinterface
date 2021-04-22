import numpy as np
from pathlib import Path

from spikeinterface.core import (BaseRecording, BaseSorting,
                                 BaseRecordingSegment, BaseSortingSegment)

try:
    import h5py

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


class CombinatoSortingExtractor(BaseSorting):
    extractor_name = 'CombinatoSortingExtractor'
    installation_mesg = ""
    installed = HAVE_H5PY
    is_writable = False
    installation_mesg = "To use the CombinatoSortingExtractor install h5py: \n\n pip install h5py\n\n"

    def __init__(self, folder_path, sampling_frequency=None, user='simple', det_sign='both'):

        folder_path = Path(folder_path)
        assert folder_path.is_dir(), 'Folder {} doesn\'t exist'.format(folder_path)
        if sampling_frequency is None:
            h5_path = str(folder_path) + '.h5'
            if Path(h5_path).exists():
                with h5py.File(h5_path, mode='r') as f:
                    sampling_frequency = f['sr'][0]

        # ~ self.set_sampling_frequency(sampling_frequency)
        det_file = str(folder_path / Path('data_' + folder_path.stem + '.h5'))
        sort_cat_files = []
        for sign in ['neg', 'pos']:
            if det_sign in ['both', sign]:
                sort_cat_file = folder_path / Path('sort_{}_{}/sort_cat.h5'.format(sign, user))
                if sort_cat_file.exists():
                    sort_cat_files.append((sign, str(sort_cat_file)))

        unit_counter = 0
        spiketrains = {}
        metadata = {}
        unsorted = []
        with h5py.File(det_file, mode='r') as fdet:
            for sign, sfile in sort_cat_files:
                with h5py.File(sfile, mode='r') as f:
                    sp_class = f['classes'][()]
                    gaux = f['groups'][()]
                    groups = {g: gaux[gaux[:, 1] == g, 0] for g in np.unique(gaux[:, 1])}  # array of classes per group
                    group_type = {group: g_type for group, g_type in f['types'][()]}
                    sp_index = f['index'][()]

                times_css = fdet[sign]['times'][()]
                for gr, cls in groups.items():
                    if group_type[gr] == -1:  # artifacts
                        continue
                    elif group_type[gr] == 0:  # unsorted
                        unsorted.append(
                            np.rint(times_css[sp_index[np.isin(sp_class, cls)]] * (sampling_frequency / 1000)))
                        continue

                    unit_counter = unit_counter + 1
                    spiketrains[unit_counter] = np.rint(
                        times_css[sp_index[np.isin(sp_class, cls)]] * (sampling_frequency / 1000))
                    metadata[unit_counter] = {'det_sign': sign,
                                              'group_type': 'single-unit' if group_type[gr] else 'multi-unit'}

        self._unsorted_train = np.array([])
        if len(unsorted) == 1:
            self._unsorted_train = unsorted[0]
        elif len(unsorted) == 2:  # unsorted in both signs
            self._unsorted_train = np.sort(np.concatenate(unsorted), kind='mergesort')

        unit_ids = np.arange(unit_counter, dtype='int64') + 1
        BaseSorting.__init__(self, sampling_frequency, unit_ids)
        self.add_sorting_segment(CombinatoSortingSegment(spiketrains))

        # TODO metadata as property 'det_sign'/ 'group_type' / 
        # ~ for u in unit_ids:
        # ~ for prop,value in metadata[u].items():
        # ~ self.set_unit_property(u, prop, value)

        self._kwargs = {'folder_path': str(folder_path), 'user': user, 'det_sign': det_sign}

    """
    def get_unsorted_spike_train(self, start_frame=None, end_frame=None):
        start_frame, end_frame = self._cast_start_end_frame(start_frame, end_frame)

        start_frame = start_frame or 0
        end_frame = end_frame or np.infty
        u = self._unsorted_train
        return u[(u >= start_frame) & (u < end_frame)]
    """


class CombinatoSortingSegment(BaseSortingSegment):
    def __init__(self, spiketrains):
        BaseSortingSegment.__init__(self)
        # spiketrains is dict
        self._spiketrains = spiketrains

    def get_unit_spike_train(self, unit_id, start_frame, end_frame):
        times = self._spiketrains[unit_id]
        if start_frame is not None:
            times = times[times >= start_frame]
        if end_frame is not None:
            times = times[times < end_frame]
        return times
