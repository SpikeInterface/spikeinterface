from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

import numpy as np
import probeinterface as pi

import neo

from packaging import version

from spikeinterface.extractors.neuropixels_utils import get_neuropixels_sample_shifts

HAS_NEO_10_2 = version.parse(neo.__version__) >= version.parse("0.10.2")


class SpikeGLXRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data saved by SpikeGLX software.
    See https://billkarsh.github.io/SpikeGLX/

    Based on :py:class:`neo.rawio.SpikeGLXRawIO`

    Contrary to older verion this reader is folder based.
    So if the folder contain several streams ('imec0.ap' 'nidq' 'imec0.lf')
    then it has to be specified with stream_id=

    Parameters
    ----------
    folder_path: str

    stream_id: str or None
        stream for instance : 'imec0.ap' 'nidq' or 'imec0.lf'
    stream_id: str or None
        If several stream, specify the one you want.
    all_annotations: bool  (default False)
        Load exhaustively all annotation from neo.

        
    """

    mode = "folder"
    NeoRawIOClass = "SpikeGLXRawIO"


    def __init__(self, folder_path, stream_id=None, all_annotations=False):
        neo_kwargs = {'dirname': str(folder_path)}
        if HAS_NEO_10_2:
            neo_kwargs['load_sync_channel'] = False
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, all_annotations=all_annotations, **neo_kwargs)

        # ~ # open the corresponding stream probe
        if HAS_NEO_10_2 and "nidq" not in self.stream_id:
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
            
            # load sample shifts
            imDatPrb_type = probe.annotations["imDatPrb_type"]

            if imDatPrb_type == 2:
                num_adcs = 16
            else:
                num_adcs = 12

            sample_shifts = get_neuropixels_sample_shifts(self.get_num_channels(), num_adcs)
            self.set_property("inter_sample_shift", sample_shifts)

        self._kwargs.update(dict(folder_path=str(folder_path)))


read_spikeglx = define_function_from_class(source_class=SpikeGLXRecordingExtractor, name="read_spikeglx")
