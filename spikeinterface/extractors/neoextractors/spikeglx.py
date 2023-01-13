from packaging import version

import numpy as np

import neo
import probeinterface as pi

from spikeinterface.extractors.neuropixels_utils import get_neuropixels_sample_shifts


from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.extractors.neuropixels_utils import get_neuropixels_sample_shifts

from .neobaseextractor import NeoBaseRecordingExtractor



class SpikeGLXRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data saved by SpikeGLX software.
    See https://billkarsh.github.io/SpikeGLX/

    Based on :py:class:`neo.rawio.SpikeGLXRawIO`

    Contrary to older verion this reader is folder based.
    So if the folder contain several streams ('imec0.ap' 'nidq' 'imec0.lf')
    then it has to be specified with `stream_id`.

    Parameters
    ----------
    folder_path: str
        The folder path to load the recordings from.
    load_sync_channel: bool dafult False
        Load or not the last channel used for synchronization.
        If True, then the probe is not loaded because one more channel
    stream_id: str, optional
        If there are several streams, specify the stream id you want to load.
        For example, 'imec0.ap' 'nidq' or 'imec0.lf'.
    stream_name: str, optional
        If there are several streams, specify the stream name you want to load.
    all_annotations: bool, optional, default: False
        Load exhaustively all annotations from neo.
    """
    mode = "folder"
    NeoRawIOClass = "SpikeGLXRawIO"
    name = "spikeglx"
    has_default_locations = True

    def __init__(self, folder_path, load_sync_channel=False, stream_id=None, stream_name=None, all_annotations=False):
        neo_kwargs = self.map_to_neo_kwargs(folder_path, load_sync_channel=load_sync_channel)
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, 
                                           stream_name=stream_name,
                                           all_annotations=all_annotations,
                                           **neo_kwargs)


        # open the corresponding stream probe for LF and AP
        # if load_sync_channel=False
        if "nidq" not in self.stream_id and not load_sync_channel:
            signals_info_dict = {
                e["stream_name"]: e for e in self.neo_reader.signals_info_list
            }
            meta_filename = signals_info_dict[self.stream_id]["meta_file"]
            # Load probe geometry if available
            if "lf" in self.stream_id:
                meta_filename = meta_filename.replace(".lf", ".ap")
            probe = pi.read_spikeglx(meta_filename)

            if probe.shank_ids is not None:
                self.set_probe(probe, in_place=True, group_mode="by_shank")
            else:
                self.set_probe(probe, in_place=True)
            self.set_probe(probe, in_place=True)

            # load num_channels_per_adc depending on probe type
            ptype = probe.annotations["probe_type"]

            if ptype in [21, 24]: # NP2.0
                num_channels_per_adc = 16
            else: # NP1.0
                num_channels_per_adc = 12

            sample_shifts = get_neuropixels_sample_shifts(self.get_num_channels(), num_channels_per_adc)
            self.set_property("inter_sample_shift", sample_shifts)

        self._kwargs.update(dict(folder_path=str(folder_path), load_sync_channel=load_sync_channel))

    @classmethod
    def map_to_neo_kwargs(cls, folder_path, load_sync_channel=False):
        neo_kwargs = {'dirname': str(folder_path), 'load_sync_channel': load_sync_channel}
        return neo_kwargs


read_spikeglx = define_function_from_class(source_class=SpikeGLXRecordingExtractor, name="read_spikeglx")
