from __future__ import annotations

from pathlib import Path

import probeinterface

from spikeinterface.extractors.neuropixels_utils import get_neuropixels_sample_shifts


from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.extractors.neuropixels_utils import get_neuropixels_sample_shifts

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseEventExtractor


class SpikeGLXRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data saved by SpikeGLX software.
    See https://billkarsh.github.io/SpikeGLX/

    Based on :py:class:`neo.rawio.SpikeGLXRawIO`

    Contrary to older verions, this reader is folder-based.
    If the folder contains several streams (e.g., "imec0.ap", "nidq" ,"imec0.lf"),
    then the stream has to be specified with "stream_id" or "stream_name".

    Parameters
    ----------
    folder_path : str
        The folder path to load the recordings from.
    load_sync_channel : bool default: False
        Whether or not to load the last channel in the stream, which is typically used for synchronization.
        If True, then the probe is not loaded.
    stream_id : str or None, default: None
        If there are several streams, specify the stream id you want to load.
        For example, "imec0.ap", "nidq", or "imec0.lf".
    stream_name : str or None, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations : bool, default: False
        Load exhaustively all annotations from neo.
    use_names_as_ids : bool, default: False
        Determines the format of the channel IDs used by the extractor. If set to True, the channel IDs will be the
        names from NeoRawIO. If set to False, the channel IDs will be the ids provided by NeoRawIO.

    Examples
    --------
    >>> from spikeinterface.extractors import read_spikeglx
    >>> recording = read_spikeglx(folder_path=r'path_to_folder_with_data', load_sync_channel=False)
    # we can load the sync channel, but then the probe is not loaded
    >>> recording = read_spikeglx(folder_path=r'pat_to_folder_with_data', load_sync_channel=True)
    """

    NeoRawIOClass = "SpikeGLXRawIO"

    def __init__(
        self,
        folder_path,
        load_sync_channel=False,
        stream_id=None,
        stream_name=None,
        all_annotations: bool = False,
        use_names_as_ids: bool = False,
    ):
        neo_kwargs = self.map_to_neo_kwargs(folder_path, load_sync_channel=load_sync_channel)
        NeoBaseRecordingExtractor.__init__(
            self,
            stream_id=stream_id,
            stream_name=stream_name,
            all_annotations=all_annotations,
            use_names_as_ids=use_names_as_ids,
            **neo_kwargs,
        )

        # open the corresponding stream probe for LF and AP
        # if load_sync_channel=False
        if "nidq" not in self.stream_id and not load_sync_channel:
            signals_info_dict = {e["stream_name"]: e for e in self.neo_reader.signals_info_list}
            meta_filename = signals_info_dict[self.stream_id]["meta_file"]
            # Load probe geometry if available
            if "lf" in self.stream_id:
                meta_filename = meta_filename.replace(".lf", ".ap")
            probe = probeinterface.read_spikeglx(meta_filename)

            if probe.shank_ids is not None:
                self.set_probe(probe, in_place=True, group_mode="by_shank")
            else:
                self.set_probe(probe, in_place=True)

            # load num_channels_per_adc depending on probe type
            model_name = probe.annotations.get("model_name", None)
            if model_name is None:
                model_name = probe.annotations["probe_name"]

            if "2.0" in model_name:  # Neuropixels 2.0
                num_channels_per_adc = 16
                num_cycles_in_adc = 16
                total_channels = 384
            else:  # NP1.0
                num_channels_per_adc = 12
                num_cycles_in_adc = 13 if "ap" in self.stream_id else 12
                total_channels = 384

            # sample_shifts is generated from total channels (384) channels
            # when only some channels are saved we need to slice this vector (like we do for the probe)
            sample_shifts = get_neuropixels_sample_shifts(total_channels, num_channels_per_adc, num_cycles_in_adc)
            if self.get_num_channels() != total_channels:
                # need slice because not all channel are saved
                chans = probeinterface.get_saved_channel_indices_from_spikeglx_meta(meta_filename)
                # lets clip to 384 because this contains also the synchro channel
                chans = chans[chans < total_channels]
                sample_shifts = sample_shifts[chans]

            self.set_property("inter_sample_shift", sample_shifts)

        self._kwargs.update(dict(folder_path=str(Path(folder_path).absolute()), load_sync_channel=load_sync_channel))

    @classmethod
    def map_to_neo_kwargs(cls, folder_path, load_sync_channel=False):
        neo_kwargs = {"dirname": str(folder_path), "load_sync_channel": load_sync_channel}
        return neo_kwargs


read_spikeglx = define_function_from_class(source_class=SpikeGLXRecordingExtractor, name="read_spikeglx")


class SpikeGLXEventExtractor(NeoBaseEventExtractor):
    """
    Class for reading events saved on the event channel by SpikeGLX software.

    Parameters
    ----------
    folder_path: str

    """

    NeoRawIOClass = "SpikeGLXRawIO"

    def __init__(self, folder_path, block_index=None):
        neo_kwargs = self.map_to_neo_kwargs(folder_path)
        NeoBaseEventExtractor.__init__(self, block_index=block_index, **neo_kwargs)

    @classmethod
    def map_to_neo_kwargs(cls, folder_path):
        neo_kwargs = {"dirname": str(folder_path)}
        return neo_kwargs


def read_spikeglx_event(folder_path, block_index=None):
    """
    Read SpikeGLX events

    Parameters
    ----------
    folder_path: str or Path
        Path to openephys folder
    block_index: int, default: None
        If there are several blocks (experiments), specify the block index you want to load.

    Returns
    -------
    event: SpikeGLXEventExtractor
    """

    event = SpikeGLXEventExtractor(folder_path, block_index=block_index)
    return event
