from __future__ import annotations

from io import StringIO
from typing import List, Optional, Union
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
import probeinterface

from spikeinterface.core import BaseRecording, BaseRecordingSegment, BaseSorting
from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.extractors.alfsortingextractor import ALFSortingSegment


class IblRecordingExtractor(BaseRecording):
    """
    Stream IBL data as an extractor object.

    Parameters
    ----------
    session : str
        The session ID to extract recordings for.
        In ONE, this is sometimes referred to as the "eid".
        When doing a session lookup such as

        >>> from one.api import ONE
        >>> one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international", silent=True)
        >>> sessions = one.alyx.rest("sessions", "list", tag="2022_Q2_IBL_et_al_RepeatedSite")

        each returned value in `sessions` refers to it as the "id".
    stream_name : str
        The name of the stream to load for the session.
        These can be retrieved from calling `StreamingIblExtractor.get_stream_names(session="<your session ID>")`.
    load_sync_channels : bool, default: false
        Load or not the last channel (sync).
        If not then the probe is loaded.
    cache_folder : str or None, default: None
        The location to temporarily store chunks of data during streaming.
        The default uses the folder designated by ONE.alyx._par.CACHE_DIR / "cache", which is typically the designated
        "Downloads" folder on your operating system. As long as `remove_cached` is set to True, the only files that will
        persist in this folder are the metadata header files and the chunk of data being actively streamed and used in RAM.
    remove_cached : bool, default: True
        Whether or not to remove streamed data from the cache immediately after it is read.
        If you expect to reuse fetched data many times, and have the disk space available, it is recommended to set this to False.

    Returns
    -------
    recording : IblStreamingRecordingExtractor
        The recording extractor which allows access to the traces.
    """

    extractor_name = "IblRecording"
    mode = "folder"
    installation_mesg = "To use the IblRecordingSegment, install ibllib: \n\n pip install ONE-api\npip install ibllib\n"
    name = "ibl_recording"

    @staticmethod
    def _get_default_one(cache_folder: Optional[Union[Path, str]] = None):
        try:
            from one.api import ONE
            from brainbox.io.one import EphysSessionLoader
        except ImportError:
            raise ImportError(IblRecordingExtractor.installation_mesg)
        one = ONE(
            base_url="https://openalyx.internationalbrainlab.org",
            password="international",
            silent=True,
            cache_dir=cache_folder,
        )
        return one

    @staticmethod
    def get_stream_names(session: str, cache_folder: Optional[Union[Path, str]] = None, one=None) -> List[str]:
        """
        Convenient retrieval of available stream names.

        Parameters
        ----------
        session : str
            The session ID to extract recordings for.
            In ONE, this is sometimes referred to as the "eid".
            When doing a session lookup such as

            >>> from one.api import ONE
            >>> one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international", silent=True)
            >>> sessions = one.alyx.rest("sessions", "list", tag="2022_Q2_IBL_et_al_RepeatedSite")

            each returned value in `sessions` refers to it as the "id".

        Returns
        -------
        stream_names : list of str
            List of stream names as expected by the `stream_name` argument for the class initialization.
        """
        try:
            from one.api import ONE
            from brainbox.io.one import EphysSessionLoader
        except ImportError:
            raise ImportError(IblRecordingExtractor.installation_mesg)

        cache_folder = Path(cache_folder) if cache_folder is not None else cache_folder

        if one is None:
            one = IblRecordingExtractor._get_default_one(cache_folder=cache_folder)
        esl = EphysSessionLoader(one=one, eid=session)
        stream_names = []
        for probe in esl.probes:
            if any(filter(lambda x: ".ap." in x, esl.ephys[probe]["ssl"].datasets)):
                stream_names.append(f"{probe}.ap")
            if any(filter(lambda x: ".lf." in x, esl.ephys[probe]["ssl"].datasets)):
                stream_names.append(f"{probe}.lf")
        return stream_names  # ['probe00.ap', 'probe00.lf', 'probe01.ap', 'probe01.lf']

    def __init__(
        self,
        session: str,
        stream_name: str,
        load_sync_channel: bool = False,
        cache_folder: Optional[Union[Path, str]] = None,
        remove_cached: bool = True,
        stream: bool = True,
        one: "one.api.OneAlyx" = None,
    ):
        try:
            from one.api import ONE
            from brainbox.io.one import SpikeSortingLoader
        except ImportError:
            raise ImportError(self.installation_mesg)

        from neo.rawio.spikeglxrawio import read_meta_file, extract_stream_info

        if one is None:
            one = IblRecordingExtractor._get_default_one(cache_folder=cache_folder)

        session_names = IblRecordingExtractor.get_stream_names(session=session, cache_folder=cache_folder, one=one)
        assert stream_name in session_names, (
            f"The `stream_name` '{stream_name}' was not found in the available listing for session '{session}'! "
            f"Please choose one of {session_names}."
        )
        pname, stream_type = stream_name.split(".")

        self.ssl = SpikeSortingLoader(one=one, eid=session, pname=pname)
        self.ssl.pid = one.alyx.rest("insertions", "list", session=session, name=pname)[0]["id"]

        self._file_streamer = self.ssl.raw_electrophysiology(band=stream_type, stream=stream)

        # get basic metadata
        meta_file = self._file_streamer.file_meta_data  # streamer downloads uncompressed metadata files on init
        meta = read_meta_file(meta_file)
        info = extract_stream_info(meta_file, meta)
        channel_ids = info["channel_names"]
        channel_gains = info["channel_gains"]
        channel_offsets = info["channel_offsets"]
        if not load_sync_channel:
            channel_ids = channel_ids[:-1]
            channel_gains = channel_gains[:-1]
            channel_offsets = channel_offsets[:-1]

        # initialize main extractor
        sampling_frequency = self._file_streamer.fs
        dtype = self._file_streamer.dtype
        BaseRecording.__init__(self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=dtype)
        self.set_channel_gains(channel_gains)
        self.set_channel_offsets(channel_offsets)
        self.extra_requirements.append("ONE-api")
        self.extra_requirements.append("ibllib")

        # set probe
        if not load_sync_channel:
            probe = probeinterface.read_spikeglx(meta_file)

            if probe.shank_ids is not None:
                self.set_probe(probe, in_place=True, group_mode="by_shank")
            else:
                self.set_probe(probe, in_place=True)

        # set channel properties
        # sometimes there are missing metadata files on the IBL side
        # when this happens a statement is printed to stderr saying these are using default metadata configurations
        with redirect_stderr(StringIO()):
            electrodes_geometry = self._file_streamer.geometry

        if not load_sync_channel:
            shank = electrodes_geometry["shank"]
            shank_row = electrodes_geometry["row"]
            shank_col = electrodes_geometry["col"]
            inter_sample_shift = electrodes_geometry["sample_shift"]
            adc = electrodes_geometry["adc"]
            index_on_probe = electrodes_geometry["ind"]
            good_channel = electrodes_geometry["flag"]
        else:
            shank = np.concatenate((electrodes_geometry["shank"], [np.nan]))
            shank_row = np.concatenate((electrodes_geometry["shank"], [np.nan]))
            shank_col = np.concatenate((electrodes_geometry["shank"], [np.nan]))
            inter_sample_shift = np.concatenate((electrodes_geometry["sample_shift"], [np.nan]))
            adc = np.concatenate((electrodes_geometry["adc"], [np.nan]))
            index_on_probe = np.concatenate((electrodes_geometry["ind"], [np.nan]))
            good_channel = np.concatenate((electrodes_geometry["shank"], [1.0]))

        self.set_property("shank", shank)
        self.set_property("shank_row", shank_row)
        self.set_property("shank_col", shank_col)
        self.set_property("inter_sample_shift", inter_sample_shift)
        self.set_property("adc", adc)
        self.set_property("index_on_probe", index_on_probe)
        if not all(good_channel):
            self.set_property("good_channel", good_channel)

        # init recording segment
        recording_segment = IblRecordingSegment(file_streamer=self._file_streamer, load_sync_channel=load_sync_channel)
        self.add_recording_segment(recording_segment)

        self._kwargs = {
            "session": session,
            "stream_name": stream_name,
            "load_sync_channel": load_sync_channel,
            "cache_folder": cache_folder,
            "remove_cached": remove_cached,
            "stream": stream,
        }


class IblRecordingSegment(BaseRecordingSegment):
    def __init__(self, file_streamer, load_sync_channel: bool = False):
        BaseRecordingSegment.__init__(self, sampling_frequency=file_streamer.fs)
        self._file_streamer = file_streamer
        self._load_sync_channel = load_sync_channel

    def get_num_samples(self):
        return self._file_streamer.ns

    def get_traces(self, start_frame: int, end_frame: int, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()
        if channel_indices is None:
            channel_indices = slice(None)
        traces = self._file_streamer.read(nsel=slice(start_frame, end_frame), volts=False)
        if not self._load_sync_channel:
            traces = traces[:, :-1]

        return traces[:, channel_indices]


class IblSortingExtractor(BaseSorting):
    """Load IBL data as a sorting extractor.

    Parameters
    ----------
    one: One = None
        instance of ONE.api to use for data loading
        for multi-processing applications, this can also be a dictionary of ONE.api arguments
        for example: one={} or one=dict(base_url='https://alyx.internationalbrainlab.org', mode='remote')
    pid: str = None
        probe insertion UUID in Alyx
    eid: str = ''
        session UUID in Alyx (optional if pid is provided)
    pname: str = ''
        probe name in Alyx (optional if pid is provided)
    kwargs:
        additional keyword arguments for brainbox.io.one.SpikeSortingLoader
    Returns
    -------
    extractor : IBLSortingExtractor
        The loaded data.
    """

    extractor_name = "IBLSorting"
    name = "ibl"
    installation_mesg = "IBL extractors require ibllib as a dependency." " To install, run: \n\n pip install ibllib\n\n"

    def __init__(self, one=None, pid=None, eid="", pname="", **kwargs):
        self._kwargs = dict(one=one, pid=pid, eid=eid, pname=pname, **kwargs)
        try:
            from one.api import ONE
            from brainbox.io.one import SpikeSortingLoader

            if isinstance(one, dict):
                one = ONE(**one)
            elif one is None:
                raise ValueError("one must be either an instance of ONE or a dictionary of ONE arguments")
        except ImportError:
            raise ImportError(self.installation_mesg)
        self.ssl = SpikeSortingLoader(one=one, pid=pid, eid=eid, pname=pname, **kwargs)
        sr = self.ssl.raw_electrophysiology(band="ap", stream=True)
        self._folder_path = self.ssl.session_path
        spikes, clusters, channels = self.ssl.load_spike_sorting(dataset_types=["spikes.samples"])
        total_units = clusters[next(iter(clusters))].size
        unit_ids = np.arange(total_units)  # in alf format, spikes.clusters index directly into clusters
        BaseSorting.__init__(self, unit_ids=unit_ids, sampling_frequency=sr.fs)
        sorting_segment = ALFSortingSegment(spikes["clusters"], spikes["samples"], sampling_frequency=sr.fs)
        self.add_sorting_segment(sorting_segment)
        self.extra_requirements.append("pandas")
        self.extra_requirements.append("ibllib")


read_ibl_recording = define_function_from_class(source_class=IblRecordingExtractor, name="read_ibl_streaming_recording")
read_ibl_sorting = define_function_from_class(source_class=IblSortingExtractor, name="read_ibl_sorting")
