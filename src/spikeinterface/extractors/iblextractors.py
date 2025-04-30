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
    eid : str or None, default: None
        The session ID to extract recordings for.
        In ONE, this is sometimes referred to as the "eid".
        When doing a session lookup such as

        >>> from one.api import ONE
        >>> one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international", silent=True)
        >>> sessions = one.alyx.rest("sessions", "list", tag="2022_Q2_IBL_et_al_RepeatedSite")

        each returned value in `sessions` refers to it as the "id".
    pid : str or None, default: None
        Probe insertion UUID in Alyx. To retrieve the PID from a session (or eid), use the following code:

        >>> from one.api import ONE
        >>> one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international", silent=True)
        >>> pids, _ = one.eid2pid("session_eid")
        >>> pid = pids[0]

        Either `eid` or `pid` must be provided.
    stream_name : str
        The name of the stream to load for the session.
        These can be retrieved from calling `StreamingIblExtractor.get_stream_names(session="<your session ID>")`.
    load_sync_channel : bool, default: false
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
    stream : bool, default: True
        Whether or not to stream the data.
    one : one.api.OneAlyx, default: None
        An instance of the ONE API to use for data loading.
        If not provided, a default instance is created using the default parameters.
        If you need to use a specific instance, you can create it using the ONE API and pass it here.

    Returns
    -------
    recording : IblStreamingRecordingExtractor
        The recording extractor which allows access to the traces.
    """

    installation_mesg = "To use the IblRecordingSegment, install ibllib: \n\n pip install ONE-api\npip install ibllib\n"

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
    def get_stream_names(eid: str, cache_folder: Optional[Union[Path, str]] = None, one=None) -> List[str]:
        """
        Convenient retrieval of available stream names.

        Parameters
        ----------
        eid : str
            The experiment ID to extract recordings for.
            In ONE, this is sometimes referred to as the "eid".
            When doing a session lookup such as

            >>> from one.api import ONE
            >>> one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international", silent=True)
            >>> eids = one.alyx.rest("sessions", "list", tag="2022_Q2_IBL_et_al_RepeatedSite")

            each returned value in `eids` refers to it as the experiment "id".
        cache_folder : str or None, default: None
            The location to temporarily store chunks of data during streaming.
        one : one.api.OneAlyx, default: None
            An instance of the ONE API to use for data loading.
            If not provided, a default instance is created using the default parameters.
            If you need to use a specific instance, you can create it using the ONE API and pass it here.
        stream_type : "ap" | "lf" | None, default: None
            The stream type to load, required when pid is provided and stream_name is not.

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
        esl = EphysSessionLoader(one=one, eid=eid)
        stream_names = []
        for probe in esl.probes:
            if any(filter(lambda x: ".ap." in x, esl.ephys[probe]["ssl"].datasets)):
                stream_names.append(f"{probe}.ap")
            if any(filter(lambda x: ".lf." in x, esl.ephys[probe]["ssl"].datasets)):
                stream_names.append(f"{probe}.lf")
        return stream_names

    def __init__(
        self,
        eid: str | None = None,
        pid: str | None = None,
        stream_name: str | None = None,
        load_sync_channel: bool = False,
        cache_folder: Optional[Path | str] = None,
        remove_cached: bool = True,
        stream: bool = True,
        one: "one.api.OneAlyx" = None,
        stream_type: str | None = None,
    ):
        try:
            from brainbox.io.one import SpikeSortingLoader
        except ImportError:
            raise ImportError(self.installation_mesg)

        from neo.rawio.spikeglxrawio import read_meta_file, extract_stream_info

        assert eid or pid, "Either `eid` or `pid` must be provided."

        if one is None:
            one = IblRecordingExtractor._get_default_one(cache_folder=cache_folder)

        if pid is not None:
            assert stream_type is not None, "When providing a PID, you must also provide a stream type."
            eid, _ = one.pid2eid(pid)
            eid = str(eid)
            pids, probes = one.eid2pid(eid)
            pids = [str(p) for p in pids]
            pname = probes[pids.index(pid)]
            stream_name = f"{pname}.{stream_type}"
        else:
            stream_names = IblRecordingExtractor.get_stream_names(eid=eid, cache_folder=cache_folder, one=one)
            if len(stream_names) > 1:
                assert (
                    stream_name is not None
                ), f"Multiple streams found for session. Please specify a stream name from {stream_names}."
                assert stream_name in stream_names, (
                    f"The `stream_name` '{stream_name}' is not available for this experiment {eid}! "
                    f"Please choose one of {stream_names}."
                )
            else:
                stream_name = stream_names[0]
            pname, stream_type = stream_name.split(".")

        self.ssl = SpikeSortingLoader(one=one, eid=eid, pid=pid, pname=pname)
        if pid is None:
            self.ssl.pid = one.alyx.rest("insertions", "list", session=eid, name=pname)[0]["id"]

        self._file_streamer = self.ssl.raw_electrophysiology(
            band=stream_type, stream=stream, remove_cached=remove_cached
        )

        # get basic metadata
        meta_file = str(self._file_streamer.file_meta_data)  # streamer downloads uncompressed metadata files on init
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
            if "lf" in stream_name:
                # same trick as in:
                # https://github.com/SpikeInterface/spikeinterface/blob/e990d53ea1024f6352596bcc237f3d60ae12e73a/src/spikeinterface/extractors/neoextractors/spikeglx.py#L63-L64
                meta_file = meta_file.replace(".lf.", ".ap.")
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
            "eid": eid,
            "pid": pid,
            "stream_name": stream_name,
            "load_sync_channel": load_sync_channel,
            "cache_folder": cache_folder,
            "remove_cached": remove_cached,
            "stream": stream,
            "stream_type": stream_type,
        }


class IblRecordingSegment(BaseRecordingSegment):
    def __init__(self, file_streamer, load_sync_channel: bool = False):
        BaseRecordingSegment.__init__(self, sampling_frequency=file_streamer.fs)
        self._file_streamer = file_streamer
        self._load_sync_channel = load_sync_channel

    def get_num_samples(self):
        return self._file_streamer.ns

    def get_traces(self, start_frame: int, end_frame: int, channel_indices):
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
    pid: str
        Probe insertion UUID in Alyx. To retrieve the PID from a session (or eid), use the following code:

        >>> from one.api import ONE
        >>> one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international", silent=True)
        >>> pids, _ = one.eid2pid("session_eid")
        >>> pid = pids[0]
    one: One | dict, required
        Instance of ONE.api or dict to use for data loading.
        For multi-processing applications, this can also be a dictionary of ONE.api arguments
        For example: one=dict(base_url='https://alyx.internationalbrainlab.org', mode='remote')
    good_clusters_only: bool, default: False
        If True, only load the good clusters
    load_unit_properties: bool, default: True
        If True, load the unit properties from the IBL database
    kwargs: dict, optional
        Additional keyword arguments to pass to the IBL SpikeSortingLoader constructor, such as `revision`.
    Returns
    -------
    extractor : IBLSortingExtractor
        The loaded data.
    """

    installation_mesg = "IBL extractors require ibllib as a dependency." " To install, run: \n\n pip install ibllib\n\n"

    def __init__(
        self, pid: str, good_clusters_only: bool = False, load_unit_properties: bool = True, one=None, **kwargs
    ):
        try:
            from one.api import ONE
            from brainbox.io.one import SpikeSortingLoader

            if isinstance(one, dict):
                one = ONE(**one)
            elif one is None:
                one = IblRecordingExtractor._get_default_one()
        except ImportError:
            raise ImportError(self.installation_mesg)
        self.ssl = SpikeSortingLoader(one=one, pid=pid)
        sr = self.ssl.raw_electrophysiology(band="ap", stream=True)
        self._folder_path = self.ssl.session_path
        spikes, clusters, channels = self.ssl.load_spike_sorting(dataset_types=["spikes.samples"], **kwargs)
        clusters = self.ssl.merge_clusters(spikes, clusters, channels)

        if good_clusters_only:
            good_cluster_slice = clusters["cluster_id"][clusters["label"] == 1]
        else:
            good_cluster_slice = slice(None)
        unit_ids = clusters["cluster_id"][good_cluster_slice]
        BaseSorting.__init__(self, unit_ids=unit_ids, sampling_frequency=sr.fs)
        sorting_segment = ALFSortingSegment(spikes["clusters"], spikes["samples"])
        self.add_sorting_segment(sorting_segment)

        if load_unit_properties:
            for key, val in clusters.items():
                # let's convert acronym to brain_area
                if key == "acronym":
                    property_name = "brain_area"
                else:
                    property_name = key
                self.set_property(property_name, val[good_cluster_slice])

        self.extra_requirements.append("ibllib")

        self._kwargs = dict(pid=pid, good_clusters_only=good_clusters_only, load_unit_properties=load_unit_properties)


read_ibl_recording = define_function_from_class(source_class=IblRecordingExtractor, name="read_ibl_streaming_recording")
read_ibl_sorting = define_function_from_class(source_class=IblSortingExtractor, name="read_ibl_sorting")
