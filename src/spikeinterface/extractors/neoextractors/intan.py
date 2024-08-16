from __future__ import annotations

from pathlib import Path

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor


class IntanRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a intan board. Supports rhd and rhs format.

    Based on :py:class:`neo.rawio.IntanRawIO`

    Parameters
    ----------
    file_path : str
        The file path to load the recordings from.
    stream_id : str, default: None
        If there are several streams, specify the stream id you want to load.
    stream_name : str, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations : bool, default: False
        Load exhaustively all annotations from neo.
    ignore_integrity_checks : bool, default: False.
        If True, data that violates integrity assumptions will be loaded. At the moment the only integrity
        check we perform is that timestamps are continuous. Setting this to True will ignore this check and set
        the attribute `discontinuous_timestamps` to True in the underlying neo object.
    use_names_as_ids : bool, default: False
        Determines the format of the channel IDs used by the extractor. If set to True, the channel IDs will be the
        names from NeoRawIO. If set to False, the channel IDs will be the ids provided by NeoRawIO.

        In Intan the ids provided by NeoRawIO are the hardware channel ids while the names are custom names given by
        the user


    """

    NeoRawIOClass = "IntanRawIO"

    def __init__(
        self,
        file_path,
        stream_id=None,
        stream_name=None,
        all_annotations=False,
        use_names_as_ids=False,
        ignore_integrity_checks: bool = False,
    ):
        neo_kwargs = self.map_to_neo_kwargs(file_path, ignore_integrity_checks=ignore_integrity_checks)
        NeoBaseRecordingExtractor.__init__(
            self,
            stream_id=stream_id,
            stream_name=stream_name,
            all_annotations=all_annotations,
            use_names_as_ids=use_names_as_ids,
            **neo_kwargs,
        )

        self._kwargs.update(dict(file_path=str(Path(file_path).absolute())))
        if "ignore_integrity_checks" in neo_kwargs:
            self._kwargs["ignore_integrity_checks"] = neo_kwargs["ignore_integrity_checks"]

    @classmethod
    def map_to_neo_kwargs(cls, file_path, ignore_integrity_checks: bool = False):

        # Only propagate the argument if the version is greater than 0.13.1
        import packaging
        import neo

        neo_version = packaging.version.parse(neo.__version__)
        if neo_version > packaging.version.parse("0.13.1"):
            neo_kwargs = {"filename": str(file_path), "ignore_integrity_checks": ignore_integrity_checks}
        else:
            neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


read_intan = define_function_from_class(source_class=IntanRecordingExtractor, name="read_intan")
