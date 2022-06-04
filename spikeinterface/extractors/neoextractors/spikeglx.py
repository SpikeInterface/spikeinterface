from .neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

import numpy as np
import probeinterface as pi

import neo

from packaging import version

HAS_NEO_10_2 = version.parse(neo.__version__) >= version.parse("0.10.2")


class SpikeGLXRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading data from a SpikeGLX system  (NI-DAQ for neuropixel probe)
    See https://billkarsh.github.io/SpikeGLX/

    Based on neo.rawio.SpikeGLXRawIO

    Contrary to older verion this reader is folder based.
    So if the folder contain several streams ('imec0.ap' 'nidq' 'imec0.lf')
    then it has to be specified xwith stream_id=

    Parameters
    ----------
    folder_path: str

    stream_id: str or None
        stream for instance : 'imec0.ap' 'nidq' or 'imec0.lf'
    """
    mode = "folder"
    NeoRawIOClass = "SpikeGLXRawIO"

    def __init__(self, folder_path, stream_id=None):
        neo_kwargs = {"dirname": str(folder_path)}
        if HAS_NEO_10_2:
            neo_kwargs["load_sync_channel"] = False
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)

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
            sample_shifts = _get_sample_shifts(self.get_num_channels(), imDatPrb_type)
            self.set_property("inter_sample_shift", sample_shifts)

        self._kwargs = dict(folder_path=str(folder_path), stream_id=stream_id)


def read_spikeglx(*args, **kwargs):
    recording = SpikeGLXRecordingExtractor(*args, **kwargs)
    return recording


read_spikeglx.__doc__ = SpikeGLXRecordingExtractor.__doc__


# TODO check sample shifts for different configurations!!!
def _get_sample_shifts(num_contact, imDatPrb_type):
    # calculate sample_shift
    # See adc_shift: https://github.com/int-brain-lab/ibllib/blob/master/ibllib/ephys/neuropixel.py
    if imDatPrb_type == 0:
        adc_channels = 12
    elif imDatPrb_type >= 2:
        adc_channels = 16

    adc = np.floor(np.arange(num_contact) / (adc_channels * 2)) * 2 + np.mod(np.arange(num_contact), 2)
    sample_shift = np.zeros_like(adc)
    for a in adc:
        sample_shift[adc == a] = np.arange(adc_channels) / adc_channels
    return sample_shift