from __future__ import annotations

import numpy as np
from pathlib import Path


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
    sampling_frequency: float | None, default: None
        The sampling frequency of the data. If None, it will be extracted from the files.
    session_info_file_path: str | Path | None, default: None
        Path to the `sessionInfo.mat` file. If None, it will be inferred from the file_path.
    """

    installation_mesg = "To use the CellExplorerSortingExtractor install pymatreader"

    def __init__(
        self,
        file_path: str | Path,
        sampling_frequency: float | None = None,
        session_info_file_path: str | Path | None = None,
    ):
        try:
            from pymatreader import read_mat
        except ImportError:
            raise ImportError(self.installation_mesg)

        self.spikes_cellinfo_path = Path(file_path)
        self.session_path = self.spikes_cellinfo_path.parent
        self.session_id = self.spikes_cellinfo_path.stem.split(".")[0]
        assert self.spikes_cellinfo_path.is_file(), f"The spikes.cellinfo.mat file must exist in {self.session_path}!"

        self.session_info_file_path = session_info_file_path

        ignore_fields = [  # This is useful in large files to avoid loading waveform data
            "maxWaveformCh",
            "maxWaveformCh1",
            "peakVoltage",
            "peakVoltage_expFitLengthConstant",
            "peakVoltage_sorted",
            "amplitudes",
            "filtWaveform",
            "filtWaveform_std",
            "rawWaveform",
            "rawWaveform_std",
            "timeWaveform",
            "maxWaveform_all",
            "rawWaveform_all",
            "filtWaveform_all",
            "timeWaveform_all",
            "channels_all",
        ]  # Note that the ignore_fields only works in matlab files that were saved on hdf5 files
        matlab_data = read_mat(self.spikes_cellinfo_path, ignore_fields=ignore_fields)
        spikes_data = matlab_data["spikes"]

        # First try to fetch it from the spikes.cellinfo file
        if sampling_frequency is None:
            sampling_frequency = spikes_data.get("sr", None)

        # If sampling rate is not available in the spikes cellinfo file, try to get it from the session file
        if sampling_frequency is None:
            sampling_frequency = self._retrieve_sampling_frequency_from_session_file()

        # Finally, fetch it from the sessionInfo file
        if sampling_frequency is None:
            sampling_frequency = self._retrieve_sampling_frequency_from_session_info_file()

        sampling_frequency = float(sampling_frequency)

        unit_ids_available = "UID" in spikes_data.keys()
        assert unit_ids_available, f"The `spikes struct` must contain field 'UID'! fields: {spikes_data.keys()}"

        spike_times_available = "times" in spikes_data.keys()
        assert spike_times_available, f"The `spike struct` must contain field 'times'! fields: {spikes_data.keys()}"

        unit_ids = spikes_data["UID"]
        spike_times = spikes_data["times"]

        # CellExplorer reports spike times in units seconds; SpikeExtractors uses time units of sampling frames
        unit_ids = [str(unit_id) for unit_id in unit_ids]
        spiketrains_dict = {unit_id: spike_times[index] for index, unit_id in enumerate(unit_ids)}
        for unit_id in unit_ids:
            spiketrains_dict[unit_id] = (sampling_frequency * spiketrains_dict[unit_id]).round().astype(np.int64)
            # Rounding is necessary to prevent data loss from int-casting floating point errors

        BaseSorting.__init__(self, unit_ids=unit_ids, sampling_frequency=sampling_frequency)
        sorting_segment = CellExplorerSortingSegment(spiketrains_dict, unit_ids)
        self.add_sorting_segment(sorting_segment)

        self.extra_requirements.append(["pymatreader"])

        self._kwargs = dict(
            file_path=str(self.spikes_cellinfo_path),
            sampling_frequency=sampling_frequency,
            session_info_file_path=str(session_info_file_path),
        )

    def _retrieve_sampling_frequency_from_session_file(self) -> float | None:
        """
        Retrieve the sampling frequency from the `.session.mat` file if available.

        This function tries to locate a .session.mat file corresponding to the current session. If found, it loads the
        data from the file, ignoring certain fields. It then tries to find 'extracellular' in the 'session' data,
        and if found, retrieves the sampling frequency ('sr') from the 'extracellular' data.

        Returns
        -------
        float | None
            The sampling frequency for the current session, or None if not found.

        """
        from pymatreader import read_mat

        sampling_frequency = None
        session_file_path = self.session_path / f"{self.session_id}.session.mat"
        if session_file_path.is_file():
            ignore_fields = ["animal", "behavioralTracking", "timeSeries", "spikeSorting", "epochs"]
            matlab_data = read_mat(session_file_path, ignore_fields=ignore_fields)
            session_data = matlab_data["session"]
            if "extracellular" in session_data.keys():
                sampling_frequency = session_data["extracellular"].get("sr", None)

        return sampling_frequency

    def _retrieve_sampling_frequency_from_session_info_file(self) -> float:
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

        from pymatreader import read_mat

        if self.session_info_file_path is None:
            self.session_info_file_path = self.session_path / f"{self.session_id}.sessionInfo.mat"

        self.session_info_file_path = Path(self.session_info_file_path).absolute()
        assert (
            self.session_info_file_path.is_file()
        ), f"No {self.session_id}.sessionInfo.mat file found in the {self.session_path}!, can't inferr sampling rate, please pass the sampling rate at initialization"

        session_info_mat = read_mat(self.session_info_file_path)
        rates = session_info_mat["sessionInfo"]["rates"]
        wideband_in_rates = "wideband" in rates.keys()
        assert wideband_in_rates, "a 'sessionInfo' should contain a  'wideband' to extract the sampling frequency!"

        # Not to be connfused with the lfpsamplingrate; reported in units Hz also present in rates
        sampling_frequency = float(rates["wideband"])

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
