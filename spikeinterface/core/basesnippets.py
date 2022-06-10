from typing import List, Union
from pathlib import Path
from .base import BaseExtractor, BaseSegment
import numpy as np
from warnings import warn
from probeinterface import Probe, ProbeGroup, write_probeinterface, read_probeinterface, select_axes

#snippets segments? 

class BaseSnippets(BaseExtractor):
    """
    Abstract class representing several multichannel snippets.
    """
    _main_annotations = ['is_filtered','is_alinged']
    _main_properties = ['group', 'location', 'gain_to_uV', 'offset_to_uV']
    _main_features = [] 

    def __init__(self,sampling_frequency: float, nafter: Union[int, None], snippet_len: int, channel_ids: List):

        BaseExtractor.__init__(self, channel_ids)
        self._nafter = nafter
        self._snippet_len = snippet_len
        self.is_dumpable = True

        self._sampling_frequency = sampling_frequency
        self._snippets_segments: List[BaseSnippetsSegment] = []
        # initialize main annotation and properties
        self.annotate(is_filtered=True)

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def __repr__(self):
        clsname = self.__class__.__name__
        nchan = self.get_num_channels()
        nseg = self.get_num_segments()
        sf_khz = self.get_sampling_frequency() / 1000.
        txt = f'{clsname}: {nchan} channels - {nseg} segments -  {sf_khz:0.1f}kHz \n snippet_len:{self._snippet_len} after peak:{self._nbefore}'
        return txt

    def get_num_segments(self):
        return len(self._snippets_segments)

    def add_snippets_segment(self, snippets_segment):
        # todo: check channel count and sampling frequency
        self._snippets_segments.append(snippets_segment)
        snippets_segment.set_parent_extractor(self)

    @property
    def channel_ids(self):
        return self._main_ids

    def get_channel_ids(self):
        return self._main_ids

    def get_num_channels(self):
        return len(self.get_channel_ids())

    def get_dtype(self):
        return self._dtype


    def get_num_snippets(self, segment_index=None):
        segment_index = self._check_segment_index(segment_index)
        return self._snippets_segments[segment_index].get_num_snippets()

    def get_total_snippets(self):
        s = 0
        for segment_index in range(self.get_num_segments()):
            s += self.get_num_snippets(segment_index)
        return s

    def has_scaled_snippets(self):
        if self.get_property('gain_to_uV') is None or self.get_property('offset_to_uV') is None:
            return False
        else:
            return True

    def is_filtered(self):
        # the is_filtered is handle with annotation
        return self._annotations.get('is_filtered', False)

    def set_probe(self, probe, group_mode='by_probe', in_place=False):
        """
        Wrapper on top on set_probes when there one unique probe.
        """
        assert isinstance(probe, Probe), 'must give Probe'
        probegroup = ProbeGroup()
        probegroup.add_probe(probe)
        return self.set_probes(probegroup, group_mode=group_mode, in_place=in_place)

    def set_probegroup(self, probegroup, group_mode='by_probe', in_place=False):
        return self.set_probes(probegroup, group_mode=group_mode, in_place=in_place)

    def set_probes(self, probe_or_probegroup, group_mode='by_probe', in_place=False):
        """
        Attach a Probe to the snippets.
        For this Probe.device_channel_indices is used to link contacts to snippets channels.
        If some contacts of the Probe are not connected (device_channel_indices=-1)
        then the snippets are "sliced" and only connected channel are kept.

        The probe order is not kept. Channel ids are re-ordered to match the channel_ids of the snippets.


        Parameters
        ----------
        probe_or_probegroup: Probe, list of Probe, or ProbeGroup
            The probe(s) to be attached to the snippets
        group_mode: str
            'by_probe' or 'by_shank'. Adds grouping property to the snippets based on the probes ('by_probe')
            or  shanks ('by_shanks')
        in_place: bool
            False by default.
            Useful internally when extractor do self.set_probegroup(probe)

        Returns
        -------
        sub_snippets: BaseSnippets
            A view of the snippets
        """
        raise NotImplementedError
      

    def get_probe(self):
        probes = self.get_probes()
        assert len(probes) == 1, 'there are several probe use .get_probes() or get_probegroup()'
        return probes[0]

    def get_probes(self):
        probegroup = self.get_probegroup()
        return probegroup.probes

    def get_probegroup(self):
        arr = self.get_property('contact_vector')
        if arr is None:
            positions = self.get_property('location')
            if positions is None:
                raise ValueError('There is not Probe attached to snippets. use set_probe(...)')
            else:
                warn('There is no Probe attached to these snippets. Creating a dummy one with contact positions')
                ndim = positions.shape[1]
                probe = Probe(ndim=ndim)
                probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 5})
                probe.set_device_channel_indices(np.arange(self.get_num_channels(), dtype='int64'))
                #  probe.create_auto_shape()
                probegroup = ProbeGroup()
                probegroup.add_probe(probe)
        else:
            probegroup = ProbeGroup.from_numpy(arr)
            for probe_index, probe in enumerate(probegroup.probes):
                contour = self.get_annotation(f'probe_{probe_index}_planar_contour')
                if contour is not None:
                    probe.set_planar_contour(contour)
        return probegroup

    def set_dummy_probe_from_locations(self, locations, shape="circle", shape_params={"radius": 1},
                                       axes="xy"):
        """
        Sets a 'dummy' probe based on locations.

        Parameters
        ----------
        locations : np.array
            Array with channel locations (num_channels, ndim) [ndim can be 2 or 3]
        shape : str, optional
            Electrode shapes, by default "circle"
        shape_params : dict, optional
            Shape parameters, by default {"radius": 1}
        axes : str, optional
            If ndim is 3, indicates the axes that define the plane of the electrodes, by default "xy"
        """
        ndim = locations.shape[1]
        probe = Probe(ndim=2)
        if ndim == 3:
            locations_2d = select_axes(locations, axes)
        else:
            locations_2d = locations
        probe.set_contacts(locations_2d, shapes=shape, shape_params=shape_params)
        probe.set_device_channel_indices(np.arange(self.get_num_channels()))

        if ndim == 3:
            probe = probe.to_3d(axes=axes)

        self.set_probe(probe, in_place=True)

    def set_channel_locations(self, locations, channel_ids=None):
        if self.get_property('contact_vector') is not None:
            raise ValueError('set_channel_locations(..) destroy the probe description, prefer set_probes(..)')
        self.set_property('location', locations, ids=channel_ids)

    def get_channel_locations(self, channel_ids=None, axes: str = 'xy'):
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        channel_indices = self.ids_to_indices(channel_ids)
        if self.get_property('contact_vector') is not None:
            if len(self.get_probes()) == 1:
                probe = self.get_probe()
                positions = probe.contact_positions[channel_indices]
            else:
                # check that multiple probes are non-overlapping
                all_probes = self.get_probes()
                all_positions = []
                for i in range(len(all_probes)):
                    probe_i = all_probes[i]
                    # check that all positions in probe_j are outside probe_i boundaries
                    x_bounds_i = [np.min(probe_i.contact_positions[:, 0]),
                                  np.max(probe_i.contact_positions[:, 0])]
                    y_bounds_i = [np.min(probe_i.contact_positions[:, 1]),
                                  np.max(probe_i.contact_positions[:, 1])]

                    for j in range(i + 1, len(all_probes)):
                        probe_j = all_probes[j]

                        if np.any(np.array([x_bounds_i[0] < cp[0] < x_bounds_i[1] and 
                                            y_bounds_i[0] < cp[1] < y_bounds_i[1]
                                            for cp in probe_j.contact_positions])):
                            raise Exception("Probes are overlapping! Retrieve locations of single probes separately")
                all_positions = np.vstack([probe.contact_positions for probe in all_probes])
                positions = all_positions[channel_indices]
            return select_axes(positions, axes)
        else:
            locations = self.get_property('location')
            if locations is None:
                raise Exception('There are no channel locations')
            locations = np.asarray(locations)[channel_indices]
            return select_axes(locations, axes)

    def has_3d_locations(self):
        return self.get_property('location').shape[1] == 3

    def clear_channel_locations(self, channel_ids=None):
        if channel_ids is None:
            n = self.get_num_channel()
        else:
            n = len(channel_ids)
        locations = np.zeros((n, 2)) * np.nan
        self.set_property('location', locations, ids=channel_ids)

    def set_channel_groups(self, groups, channel_ids=None):
        if 'probes' in self._annotations:
            warn('set_channel_groups() destroys the probe description. Using set_probe() is preferable')
            self._annotations.pop('probes')
        self.set_property('group', groups, ids=channel_ids)

    def get_channel_groups(self, channel_ids=None):
        groups = self.get_property('group', ids=channel_ids)
        return groups

    def clear_channel_groups(self, channel_ids=None):
        if channel_ids is None:
            n = self.get_num_channels()
        else:
            n = len(channel_ids)
        groups = np.zeros(n, dtype='int64')
        self.set_property('group', groups, ids=channel_ids)

    def set_channel_gains(self, gains, channel_ids=None):
        if np.isscalar(gains):
            gains = [gains] * self.get_num_channels()
        self.set_property('gain_to_uV', gains, ids=channel_ids)

    def get_channel_gains(self, channel_ids=None):
        return self.get_property('gain_to_uV', ids=channel_ids)

    def set_channel_offsets(self, offsets, channel_ids=None):
        if np.isscalar(offsets):
            offsets = [offsets] * self.get_num_channels()
        self.set_property('offset_to_uV', offsets, ids=channel_ids)

    def get_channel_offsets(self, channel_ids=None):
        return self.get_property('offset_to_uV', ids=channel_ids)

    def get_channel_property(self, channel_id, key):
        values = self.get_property(key)
        v = values[self.id_to_index(channel_id)]
        return v

    def get_frames(self,
                   indeces = None,
                   segment_index: Union[int, None] = None
                   ):
        segment_index = self._check_segment_index(segment_index)
        spts = self._snippets_segments[segment_index]
        return spts.get_frames(indeces)

    def get_snippets(self,
                   indeces,
                   segment_index: Union[int, None] = None,
                   channel_ids: Union[List, None] = None,
                   return_scaled=False,
                   ):

        segment_index = self._check_segment_index(segment_index)
        spts = self._snippets_segments[segment_index]
        channel_indices = self.ids_to_indices(channel_ids, prefer_slice=True)
        wfs = spts.get_snippets(indeces, channel_indices=channel_indices)
        
        
        if return_scaled:
            if not self.has_scaled_snippets():
                raise ValueError('These snippets do not support return_scaled=True (need gain_to_uV and offset_'
                                 'to_uV properties)')
            else:
                gains = self.get_property('gain_to_uV')
                offsets = self.get_property('offset_to_uV')
                gains = gains[channel_indices].astype('float32')
                offsets = offsets[channel_indices].astype('float32')
                wfs = wfs.astype('float32') * gains + offsets
        return wfs


    def get_snippets_from_frames(self,
                   segment_index: Union[int, None] = None,
                   start_frame: Union[int, None] = None,
                   end_frame: Union[int, None] = None,
                   channel_ids: Union[List, None] = None,
                   return_scaled=False,
                   ):

        segment_index = self._check_segment_index(segment_index)
        spts = self._snippets_segments[segment_index]
        indeces = spts.frames_to_indices(start_frame, end_frame)

        return self.get_snippets(indeces, channel_ids=channel_ids, return_scaled=return_scaled)

    def planarize(self, axes: str = "xy"):
        """
        Returns snippets with a 2D probe from one with a 3D probe

        Parameters
        ----------
        axes : str, optional
            The axes to keep, by default "xy"

        Returns
        -------
        BaseSnippets
            The snippets with 2D positions
        """
        assert self.has_3d_locations, "The 'planarize' function needs snippets with 3d locations"
        assert len(axes) == 2, "You need to specify 2 dimensions (e.g. 'xy', 'zy')"

        probe2d = self.get_probe().to_2d(axes=axes)
        snippets2d = self.clone()
        snippets2d.set_probe(probe2d, in_place=True)

        return snippets2d

    def _save(self, format='binary', **save_kwargs):
        raise NotImplementedError

    def get_num_segments(self):
        return len(self._snippets_segments)

    def _extra_metadata_from_folder(self, folder):
        # load probe
        folder = Path(folder)
        if (folder / 'probe.json').is_file():
            probegroup = read_probeinterface(folder / 'probe.json')
            self.set_probegroup(probegroup, in_place=True)

        # load time vector if any
        for segment_index, rs in enumerate(self._snippets_segments):
            time_file = folder / f'times_cached_seg{segment_index}.npy'
            if time_file.is_file():
                time_vector = np.load(time_file)
                rs.time_vector = time_vector

    def _extra_metadata_to_folder(self, folder):
        # save probe
        if self.get_property('contact_vector') is not None:
            probegroup = self.get_probegroup()
            write_probeinterface(folder / 'probe.json', probegroup)

        # save time vector if any
        for segment_index, rs in enumerate(self._snippets_segments):
            d = rs.get_times_kwargs()
            time_vector = d['time_vector']
            if time_vector is not None:
                np.save(folder / f'times_cached_seg{segment_index}.npy', time_vector)


class BaseSnippetsSegment(BaseSegment):
    """
    Abstract class representing multichannel snippets
    """

    def __init__(self):
        BaseSegment.__init__(self)

    def get_snippets(self,
                    indices,
                    end_frame: Union[int, None] = None,
                    channel_indices: Union[List, None] = None,
                    ) -> np.ndarray:
        """
        Return the snippets, optionally for a subset of samples and/or channels

        Parameters
        ----------
        indexes: (Union[int, None], optional)
            start sample index, or zero if None. Defaults to None.
        end_frame: (Union[int, None], optional)
            end_sample, or number of samples if None. Defaults to None.
        channel_indices: (Union[List, None], optional)
            Indices of channels to return, or all channels if None. Defaults to None.

        Returns
        -------
        snippets: np.ndarray
            Array of snippets, num_snippets x num_samples x num_channels
        """
        raise NotImplementedError

    def get_num_snippets(self):
        """Returns the number of snippets in this segment

        Returns:
            SampleIndex: Number of snippets in the segment
        """
        raise NotImplementedError

    def get_frames(self, indeces):
        """Returns the frames of the snippets in this  segment

        Returns:
            SampleIndex: Number of samples in the  segment
        """
        raise NotImplementedError

    def frames_to_indices(start_frame: Union[int, None] = None,
                            end_frame: Union[int, None] = None):

        """
        Return the slice of snippets

        Parameters
        ----------
        start_frame: (Union[int, None], optional)
            start sample index, or zero if None. Defaults to None.
        end_frame: (Union[int, None], optional)
            end_sample, or number of samples if None. Defaults to None.

        Returns
        -------
        snippets: slice
            slice of selected snippets
        """
        raise NotImplementedError