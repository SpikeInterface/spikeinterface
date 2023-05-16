from __future__ import annotations

import numpy as np
from pathlib import Path
import warnings
import datetime

from ..core import BaseSorting, BaseSortingSegment
from ..core.core_tools import define_function_from_class


class CellExplorerSortingExtractor(BaseSorting):
    """
    Extracts spiking information from `.mat` file stored in the CellExplorer format.
    Spike times are stored in units of seconds so we transform them to units of samples.

    The newer version of the format is described here:
    https://cellexplorer.org/data-structure/

    Whereas the old format is described here:
    https://github.com/buzsakilab/buzcode/wiki/Data-Formatting-Standards

    Parameters
    ----------
    file_path: str | Path
        Path to `.mat` file containing spikes. Usually named `session_id.spikes.cellinfo.mat`
    sampling_frequency: float | None, optional
        The sampling frequency of the data. If None, it will be extracted from the files.
    session_info_file_path: str | Path | None, optional
        Path to the `sessionInfo.mat` file. If None, it will be inferred from the file_path.
    """

    extractor_name = "CellExplorerSortingExtractor"
    is_writable = True
    mode = "file"
    installation_mesg = "To use the CellExplorerSortingExtractor install scipy and h5py"

    def __init__(
        self,
        file_path: str | Path | None = None,
        sampling_frequency: float | None = None,
        session_info_file_path: str | Path | None = None,
        spikes_matfile_path: str | Path | None = None,
        session_info_matfile_path: str | Path | None = None,
    ):
        try:
            import h5py
            import scipy.io
        except ImportError:
            raise ImportError(self.installation_mesg)

        assert (
            file_path is not None or spikes_matfile_path is not None
        ), "Either file_path or spikes_matfile_path must be provided!"

        if spikes_matfile_path is not None:
            # Raise an error if the warning period has expired
            deprecation_issued = datetime.datetime(2023, 4, 1)
            deprecation_deadline = deprecation_issued + datetime.timedelta(days=180)
            if datetime.datetime.now() > deprecation_deadline:
                raise ValueError("The spikes_matfile_path argument is no longer supported in. Use file_path instead.")

            # Otherwise, issue a DeprecationWarning
            else:
                warnings.warn(
                    "The spikes_matfile_path argument is deprecated and will be removed in six months. "
                    "Use file_path instead.",
                    DeprecationWarning,
                )
            file_path = spikes_matfile_path if file_path is None else file_path

        if session_info_matfile_path is not None:
            # Raise an error if the warning period has expired
            deprecation_issued = datetime.datetime(2023, 4, 1)
            deprecation_deadline = deprecation_issued + datetime.timedelta(days=180)
            if datetime.datetime.now() > deprecation_deadline:
                raise ValueError(
                    "The session_info_matfile_path argument is no longer supported in. Use session_info_file_path instead."
                )

            # Otherwise, issue a DeprecationWarning
            else:
                warnings.warn(
                    "The session_info_matfile_path argument is deprecated and will be removed in six months. "
                    "Use session_info_file_path instead.",
                    DeprecationWarning,
                )
            session_info_file_path = (
                session_info_matfile_path if session_info_file_path is None else session_info_file_path
            )

        self.spikes_cellinfo_path = Path(file_path).absolute()
        assert self.spikes_cellinfo_path.is_file(), f"The spikes.cellinfo.mat file must exist in {self.folder_path}!"

        self.folder_path = self.spikes_cellinfo_path.parent
        self.session_info_file_path = session_info_file_path

        self.session_id = self.spikes_cellinfo_path.stem.split(".")[0]

        read_as_hdf5 = False
        try:
            matlab_file = scipy.io.loadmat(file_name=str(self.spikes_cellinfo_path), simplify_cells=True)
            spikes_mat = matlab_file["spikes"]
            assert isinstance(spikes_mat, dict), f"field `spikes` must be a dict, not {type(spikes_mat)}!"

        except NotImplementedError:
            matlab_file = h5py.File(name=self.spikes_cellinfo_path, mode="r")
            spikes_mat = matlab_file["spikes"]
            assert isinstance(spikes_mat, h5py.Group), f"field `spikes` must be a Group, not {type(spikes_mat)}!"
            read_as_hdf5 = True

        if sampling_frequency is None:
            # First try the new format of spikes.cellinfo.mat files where sampling rate is included in the file
            sr_data = spikes_mat.get("sr", None)
            sampling_frequency = sr_data[()] if isinstance(sr_data, h5py.Dataset) else None

        if sampling_frequency is None:
            sampling_frequency = self._retrieve_sampling_frequency_from_session_info()

        sampling_frequency = float(sampling_frequency)

        unit_ids_available = "UID" in spikes_mat.keys()
        assert unit_ids_available, f"The `spikes struct` must contain field 'UID'! fields: {spikes_mat.keys()}"

        spike_times_available = "times" in spikes_mat.keys()
        assert spike_times_available, f"The `spike struct` must contain field 'times'! fields: {spikes_mat.keys()}"

        unit_ids = spikes_mat["UID"]
        spike_times = spikes_mat["times"]

        if read_as_hdf5:
            assert isinstance(unit_ids, h5py.Dataset), f"`unit_ids` must be a Dataset, not {type(unit_ids)}!"
            assert isinstance(spike_times, h5py.Dataset), f"`spike_times` must be a Dataset, not {type(spike_times)}!"

            unit_ids = unit_ids[:].squeeze().astype("int")
            references = (ref[0] for ref in spike_times[:])  # These are HDF5 references
            spike_times_data = (matlab_file[ref] for ref in references if isinstance(matlab_file[ref], h5py.Dataset))
            # Format as a list of numpy arrays
            spike_times = [data[()].squeeze() for data in spike_times_data]

        # CellExplorer reports spike times in units seconds; SpikeExtractors uses time units of sampling frames
        unit_ids = unit_ids[:].tolist()
        spiketrains_dict = {unit_id: spike_times[index] for index, unit_id in enumerate(unit_ids)}
        for unit_id in unit_ids:
            spiketrains_dict[unit_id] = (sampling_frequency * spiketrains_dict[unit_id]).round().astype(np.int64)
            # Rounding is necessary to prevent data loss from int-casting floating point errors

        BaseSorting.__init__(self, unit_ids=unit_ids, sampling_frequency=sampling_frequency)
        sorting_segment = CellExplorerSortingSegment(spiketrains_dict, unit_ids)
        self.add_sorting_segment(sorting_segment)

        self.extra_requirements.append("scipy")

        self._kwargs = dict(
            file_path=str(self.spikes_cellinfo_path),
            sampling_frequency=sampling_frequency,
            session_info_file_path=str(session_info_file_path),
        )

    def _retrieve_sampling_frequency_from_session_info(self) -> float:
        """
        Retrieve the sampling frequency from the `sessionInfo.mat` file when available.

        This function tries to locate a .sessionInfo.mat file corresponding to the current session. It then loads this
        file (either as a standard .mat file or as an HDF5 file if the former is not possible) and extracts the wideband
        sampling frequency from the 'rates' field of the 'sessionInfo' struct.

        Returns
        -------
        float
            The wideband sampling frequency for the current session.
        """
        import h5py
        import scipy.io

        if self.session_info_file_path is None:
            self.session_info_file_path = self.folder_path / f"{self.session_id}.sessionInfo.mat"

        self.session_info_file_path = Path(self.session_info_file_path).absolute()
        assert (
            self.session_info_file_path.is_file()
        ), f"No {self.session_id}.sessionInfo.mat file found in the {self.folder_path}!, can't inferr sampling rate"

        read_as_hdf5 = False
        try:
            session_info_mat = scipy.io.loadmat(file_name=str(self.session_info_file_path), simplify_cells=True)
        except NotImplementedError:
            session_info_mat = h5py.File(name=str(self.session_info_file_path), mode="r")
            read_as_hdf5 = True

        rates = session_info_mat["sessionInfo"]["rates"]
        wideband_in_rates = "wideband" in rates.keys()
        assert wideband_in_rates, "a 'sessionInfo' should contain a  'wideband' to extract the sampling frequency!"

        # Not to be connfused with the lfpsamplingrate; reported in units Hz also present in rates
        sampling_frequency = rates["wideband"]

        if read_as_hdf5:
            sampling_frequency = sampling_frequency[()]

        return sampling_frequency


class CellExplorerSortingSegment(BaseSortingSegment):
    def __init__(self, spiketrains_dict, unit_ids):
        self.spiketrains_dict = spiketrains_dict
        self._unit_ids = list(unit_ids)
        BaseSortingSegment.__init__(self)

    def get_unit_spike_train(
        self,
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


read_cellexplorer = define_function_from_class(source_class=CellExplorerSortingExtractor, name="read_cellexplorer")
