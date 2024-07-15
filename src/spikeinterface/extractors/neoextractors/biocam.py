from __future__ import annotations

from pathlib import Path

import probeinterface

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor


class BiocamRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a Biocam file from 3Brain.

    Based on :py:class:`neo.rawio.BiocamRawIO`

    Parameters
    ----------
    file_path : str
        The file path to load the recordings from.
    mea_pitch : float, default: None
        The inter-electrode distance (pitch) between electrodes.
    electrode_width : float, default: None
        Width of the electrodes in um.
    stream_id : str, default: None
        If there are several streams, specify the stream id you want to load.
    stream_name : str, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations : bool, default: False
        Load exhaustively all annotations from neo.
    use_names_as_ids : bool, default: False
        Determines the format of the channel IDs used by the extractor. If set to True, the channel IDs will be the
        names from NeoRawIO. If set to False, the channel IDs will be the ids provided by NeoRawIO.
    """

    NeoRawIOClass = "BiocamRawIO"

    def __init__(
        self,
        file_path,
        mea_pitch=None,
        electrode_width=None,
        stream_id=None,
        stream_name=None,
        all_annotations: bool = False,
        use_names_as_ids: bool = False,
    ):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        NeoBaseRecordingExtractor.__init__(
            self,
            stream_id=stream_id,
            stream_name=stream_name,
            all_annotations=all_annotations,
            use_names_as_ids=use_names_as_ids,
            **neo_kwargs,
        )

        # load probe from probeinterface
        probe_kwargs = {}
        if mea_pitch is not None:
            probe_kwargs["mea_pitch"] = mea_pitch
        if electrode_width is not None:
            probe_kwargs["electrode_width"] = electrode_width
        probe = probeinterface.read_3brain(file_path, **probe_kwargs)
        self.set_probe(probe, in_place=True)
        self.set_property("row", self.get_property("contact_vector")["row"])
        self.set_property("col", self.get_property("contact_vector")["col"])

        self._kwargs.update(
            {
                "file_path": str(Path(file_path).absolute()),
                "mea_pitch": mea_pitch,
                "electrode_width": electrode_width,
            }
        )

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


read_biocam = define_function_from_class(source_class=BiocamRecordingExtractor, name="read_biocam")
