from __future__ import annotations

import numpy as np
from pathlib import Path

from ..core import BaseSorting, BaseSortingSegment
from ..core.core_tools import define_function_from_class


class CellExplorerSortingExtractor(BaseSorting):
    """
    Extracts spiking information from .mat files stored in the CellExplorer format.
    Spike times are stored in units of seconds so we transform them to units of samples.
    
    The newer version of the format is described here:
    https://cellexplorer.org/data-structure/
    
    Whereas the old format is described here:
    https://github.com/buzsakilab/buzcode/wiki/Data-Formatting-Standards
    
    Parameters
    ----------
    folder_path: str | Path
        Path to the folder containing the .mat files from the session.
    sampling_frequency: float | None, optional
        The sampling frequency of the data. If None, it will be extracted from the files.
    session_info_matfile_path: str | Path | None, optional
        Path to the session info .mat file. If None, it will be inferred from the folder path.
    """

    extractor_name = "CellExplorerSortingExtractor"
    is_writable = True
    mode = "folder"
    installation_mesg = "To use the CellExplorerSortingExtractor install scipy and h5py"

    def __init__(self, folder_path: str | Path,
                 sampling_frequency: float | None = None,
                 session_info_matfile_path: str | Path | None = None):
        
        try:
            import h5py
            import scipy.io 
        except ImportError:
            raise ImportError(self.installation_mesg)

        folder_path = Path(folder_path)
        session_id = next((path.stem.split(".")[0] for path in folder_path.iterdir() if path.suffix == ".mat"), None)
        assert session_id is not None, f"No .mat files found in the {folder_path}!"
        
        spikes_cellinfo_path = folder_path / f"{session_id}.spikes.cellinfo.mat"
        assert spikes_cellinfo_path.is_file(), f"The spikes.cellinfo.mat file must exist in the {folder_path}!"

        try:
            spikes_mat = scipy.io.loadmat(file_name=str(spikes_cellinfo_path), simplify_cells=True)
            self.read_spikes_info_with_scipy = True
        except NotImplementedError: 
            spikes_mat = h5py.File(name=spikes_cellinfo_path, mode='r')
            self.read_spikes_info_with_scipy = False      
        
        if sampling_frequency is None:
            # This is for the new format of spikes.cellinfo.mat files
            sampling_frequency = spikes_mat["spikes"].get("sr", None) 
        
        if sampling_frequency is None:
            if session_info_matfile_path is None:
                session_info_matfile_path = folder_path / f"{session_id}.sessionInfo.mat"
            session_info_matfile_path = Path(session_info_matfile_path)
            assert session_info_matfile_path.is_file(), f"No {session_id}.sessionInfo.mat file found in the {folder_path}!" 

            try:
                session_info_mat = scipy.io.loadmat(file_name=str(session_info_matfile_path), simplify_cells=True)
                self.read_session_info_with_scipy = True
            except NotImplementedError:
                session_info_mat = h5py.File(name=str(session_info_matfile_path), mode="r")
                self.read_session_info_with_scipy = False
            
            rates = session_info_mat["sessionInfo"]["rates"]
            wideband_in_rates = "wideband" in rates.keys()
            assert wideband_in_rates, "a 'sessionInfo' should contain a  'wideband' to extract the sampling frequency!"
    
            # careful not to confuse it with the lfpsamplingrate; reported in units Hz
            if self.read_session_info_with_scipy:
                sampling_frequency = float(rates["wideband"])
            else:
                sampling_frequency = float(rates["wideband"][()])
            
        unit_ids_available = "UID" in spikes_mat["spikes"].keys()
        assert unit_ids_available, "The spikes.cellinfo.mat file must contain field 'UID'!"
        
        spike_times_available =  "times" in spikes_mat["spikes"].keys()
        assert spike_times_available, "The spikes.cellinfo.mat file must contain field 'times'!"
        
        if self.read_spikes_info_with_scipy:
            unit_ids = spikes_mat["spikes"]["UID"].tolist()
            spike_times = spikes_mat["spikes"]["times"]            
        else:
            unit_ids = spikes_mat["spikes"]["UID"][:].squeeze().astype("int").tolist()
            spike_references = spikes_mat["spikes"]["times"][:]  # These are HDF5 references
            spike_times = [spikes_mat[reference[0]][()].squeeze() for reference in spike_references]
            
        # CellExplorer reports spike times in units seconds; SpikeExtractors uses time units of sampling frames
        # Rounding is necessary to prevent data loss from int-casting floating point errors
        spiketrains_dict = {unit_id: spike_times[index] for index, unit_id in enumerate(unit_ids)}
        for unit_id in unit_ids:
            spiketrains_dict[unit_id] = (sampling_frequency * spiketrains_dict[unit_id]).round().astype(np.int64)
        
        BaseSorting.__init__(self, unit_ids=unit_ids, sampling_frequency=sampling_frequency)
        sorting_segment = CellExplorerSortingSegment(spiketrains_dict, unit_ids)
        self.add_sorting_segment(sorting_segment)

        self.extra_requirements.append('scipy')

        self._kwargs = dict(folder_path=str(folder_path.absolute()))

class CellExplorerSortingSegment(BaseSortingSegment):
    def __init__(self, spiketrains_dict, unit_ids):
        self.spiketrains_dict = spiketrains_dict
        self._unit_ids = list(unit_ids)
        BaseSortingSegment.__init__(self)

    def get_unit_spike_train(self,
                             unit_id,
                             start_frame,
                             end_frame,
                             ) -> np.ndarray:

        spike_frames = self.spiketrains_dict[unit_id]
        # clip
        if start_frame is not None:
            spike_frames = spike_frames[spike_frames >= start_frame]

        if end_frame is not None:
            spike_frames = spike_frames[spike_frames <= end_frame]

        return spike_frames

read_cellexplorer = define_function_from_class(source_class=CellExplorerSortingExtractor,
                                               name="read_cellexplorer")
