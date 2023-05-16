import numpy as np
from pathlib import Path
from typing import Union, Optional

from ..core import BaseSorting, BaseSortingSegment
from ..core.core_tools import define_function_from_class


try:
    import scipy.io 
    import hdf5storage
    HAVE_SCIPY_AND_HDF5STORAGE = True
except ImportError:
    HAVE_SCIPY_AND_HDF5STORAGE = False


PathType = Union[str, Path]
OptionalPathType = Optional[PathType]  


class CellExplorerSortingExtractor(BaseSorting):
    """
    Extracts spiking information from .mat files stored in the CellExplorer format.
    Spike times are stored in units of seconds.

    Parameters
    ----------
    spikes_matfile_path : PathType
        Path to the sorting_id.spikes.cellinfo.mat file.
    """

    extractor_name = "CellExplorerSortingExtractor"
    installed = HAVE_SCIPY_AND_HDF5STORAGE
    is_writable = True
    mode = "file"
    installation_mesg = "To use the CellExplorerSortingExtractor install scipy and hdf5storage: \n\n pip install scipy  hdf5storage"

    def __init__(self, spikes_matfile_path: PathType,
                 session_info_matfile_path: OptionalPathType=None,
                 sampling_frequency: Optional[float] = None):
        assert self.installed, self.installation_mesg

        spikes_matfile_path = Path(spikes_matfile_path)
        assert (
            spikes_matfile_path.is_file()
        ), f"The spikes_matfile_path ({spikes_matfile_path}) must exist!"
        
        if sampling_frequency is None:
            folder_path = spikes_matfile_path.parent
            sorting_id = spikes_matfile_path.name.split(".")[0]
            if session_info_matfile_path is None:
                session_info_matfile_path = folder_path / f"{sorting_id}.sessionInfo.mat"
            session_info_matfile_path = Path(session_info_matfile_path)
            assert (
                (session_info_matfile_path).is_file()
            ), f"No {sorting_id}.sessionInfo.mat file found in the folder!" 

            try:
                session_info_mat = scipy.io.loadmat(file_name=str(session_info_matfile_path))
                self.read_session_info_with_scipy = True
            except NotImplementedError:
                session_info_mat = hdf5storage.loadmat(file_name=str(session_info_matfile_path))
                self.read_session_info_with_scipy = False
            
            assert session_info_mat["sessionInfo"]["rates"][0][0]["wideband"], (
                "The sesssionInfo.mat file must contain "
                "a 'sessionInfo' struct with field 'rates' containing field 'wideband' to extract the sampling frequency!"
            )
            if self.read_session_info_with_scipy:
                sampling_frequency = float(
                    session_info_mat["sessionInfo"]["rates"][0][0]["wideband"][0][0][0][0]
                )  # careful not to confuse it with the lfpsamplingrate; reported in units Hz
            else:
                sampling_frequency = float(
                    session_info_mat["sessionInfo"]["rates"][0][0]["wideband"][0][0]
                )  # careful not to confuse it with the lfpsamplingrate; reported in units Hz

        try:
            spikes_mat = scipy.io.loadmat(file_name=str(spikes_matfile_path))
            self.read_spikes_info_with_scipy = True
        except NotImplementedError: 
            spikes_mat = hdf5storage.loadmat(file_name=str(spikes_matfile_path))
            self.read_spikes_info_with_scipy = False

        assert np.all(
            np.isin(["UID", "times"], spikes_mat["spikes"].dtype.names)
        ), "The spikes.cellinfo.mat file must contain a 'spikes' struct with fields 'UID' and 'times'!"

        # CellExplorer reports spike times in units seconds; SpikeExtractors uses time units of sampling frames
        # Rounding is necessary to prevent data loss from int-casting floating point errors
        if self.read_spikes_info_with_scipy:
            unit_ids = np.asarray(spikes_mat["spikes"]["UID"][0][0][0])
            spiketrains = [
                (np.array([y[0] for y in x]) * sampling_frequency).round().astype(np.int64)
                for x in spikes_mat["spikes"]["times"][0][0][0]
            ]
        else:
            unit_ids = np.asarray(spikes_mat["spikes"]["UID"][0][0])
            spiketrains = [
                (np.array([y[0] for y in x]) * sampling_frequency).round().astype(np.int64)
                for x in spikes_mat["spikes"]["times"][0][0]            
            ]
        
        BaseSorting.__init__(self, unit_ids=unit_ids, sampling_frequency=sampling_frequency)
        sorting_segment = CellExplorerSortingSegment(spiketrains, unit_ids)
        self.add_sorting_segment(sorting_segment)

        self.extra_requirements.append('scipy')
        self.extra_requirements.append('hdf5storage')

        self._kwargs = dict(spikes_matfile_path=str(spikes_matfile_path.absolute()))


class CellExplorerSortingSegment(BaseSortingSegment):
    def __init__(self, spiketrains, unit_ids):
        self._spiketrains = spiketrains
        self._unit_ids = list(unit_ids)
        BaseSortingSegment.__init__(self)

    def get_unit_spike_train(self,
                             unit_id,
                             start_frame,
                             end_frame,
                             ) -> np.ndarray:
        # must be implemented in subclass
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.inf
        spike_frames = self._spiketrains[self._unit_ids.index(unit_id)]
        inds = np.where((start_frame <= spike_frames) & (spike_frames < end_frame))
        return spike_frames[inds]


read_cellexplorer = define_function_from_class(source_class=CellExplorerSortingExtractor,
                                               name="read_cellexplorer")
