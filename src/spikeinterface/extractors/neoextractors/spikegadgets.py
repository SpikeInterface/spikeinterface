from pathlib import Path
import warnings
import numpy as np

import probeinterface
from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.extractors.neuropixels_utils import (
    get_neuropixels_sample_shifts_from_probe,
    compute_saturation_threshold_from_probe,
)

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

        if probeinterface.has_spikegadgets_neuropixels_probes(file_path):
            probegroup = probeinterface.read_spikegadgets_neuropixels(file_path)

            # get inter-sample shifts based on the probe information and mux channels
            # SpikeGadgets writes multiple probes in the same file and the contacts are
            # interleaved. The `device_channel_indices` of the probe is used to map the
            # sample shifts to the correct channels.
            # We instantiate the sample_shifts array with -1 to indicate channels for
            # which we don't have a sample shift (sample shifts are [0-1] by definition)
            sample_shifts = -1 * np.ones(self.get_num_channels())
            saturation_thresholds_uV = []
            for probe in probegroup.probes:
                sample_shifts_probe = get_neuropixels_sample_shifts_from_probe(probe)
                if sample_shifts_probe is not None:
                    sample_shifts[probe.device_channel_indices] = sample_shifts_probe
                # add saturation levels if available
                saturation_threshold_uV_probe = compute_saturation_threshold_from_probe(probe, self.stream_id)
                if saturation_threshold_uV_probe is not None:
                    saturation_thresholds_uV.append(saturation_threshold_uV_probe)

            self.set_probegroup(probegroup)

            if np.all(sample_shifts != -1):
                self.set_property("inter_sample_shift", sample_shifts)
            if len(set(saturation_thresholds_uV)) == 1:
                self.annotate(saturation_threshold_uV=saturation_thresholds_uV[0])
            else:
                warnings.warn("Multiple saturation thresholds found for different probes, unable to annotate.")

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


read_spikegadgets = define_function_from_class(source_class=SpikeGadgetsRecordingExtractor, name="read_spikegadgets")
