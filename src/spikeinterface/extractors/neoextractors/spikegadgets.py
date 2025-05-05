from __future__ import annotations
from pathlib import Path

import packaging

import packaging.version
import probeinterface
from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor


class SpikeGadgetsRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading rec files from spikegadgets.

    Based on :py:class:`neo.rawio.SpikeGadgetsRawIO`

    Parameters
    ----------
    file_path : str
        The file path to load the recordings from.
    stream_id : str or None, default: None
        If there are several streams, specify the stream id you want to load.
    stream_name : str or None, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations : bool, default: False
        Load exhaustively all annotations from neo.
    use_names_as_ids : bool, default: False
        Determines the format of the channel IDs used by the extractor. If set to True, the channel IDs will be the
        names from NeoRawIO. If set to False, the channel IDs will be the ids provided by NeoRawIO.

    Examples
    --------
    >>> from spikeinterface.extractors import read_spikegadgets
    >>> recording = read_spikegadgets(file_path=r'my_data.rec')
    """

    NeoRawIOClass = "SpikeGadgetsRawIO"

    def __init__(
        self,
        file_path,
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
        self._kwargs.update(dict(file_path=str(Path(file_path).absolute()), stream_id=stream_id))

        probegroup = None  # TODO remove once probeinterface is updated to 0.2.22 in the pyproject.toml
        if packaging.version.parse(probeinterface.__version__) > packaging.version.parse("0.2.21"):
            probegroup = probeinterface.read_spikegadgets(file_path, raise_error=False)

        if probegroup is not None:
            self.set_probegroup(probegroup, in_place=True)

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


read_spikegadgets = define_function_from_class(source_class=SpikeGadgetsRecordingExtractor, name="read_spikegadgets")
