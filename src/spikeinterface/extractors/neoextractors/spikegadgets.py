from __future__ import annotations

from pathlib import Path

import probeinterface

from spikeinterface.core.core_tools import define_function_from_class

from .neobaseextractor import NeoBaseRecordingExtractor


class SpikeGadgetsRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading rec files from spikegadgets.

    Based on :py:class:`neo.rawio.SpikeGadgetsRawIO`

    Parameters
    ----------
    file_path: str
        The file path to load the recordings from.
    stream_id: str or None, default: None
        If there are several streams, specify the stream id you want to load.
    stream_name: str or None, default: None
        If there are several streams, specify the stream name you want to load.
    all_annotations: bool, default: False
        Load exhaustively all annotations from neo.
    """

    mode = "file"
    NeoRawIOClass = "SpikeGadgetsRawIO"
    name = "spikegadgets"

    def __init__(self, file_path, stream_id=None, stream_name=None, block_index=None, all_annotations=False):
        neo_kwargs = self.map_to_neo_kwargs(file_path)
        NeoBaseRecordingExtractor.__init__(
            self, stream_id=stream_id, stream_name=stream_name, all_annotations=all_annotations, **neo_kwargs
        )
        self._kwargs.update(dict(file_path=str(Path(file_path).absolute()), stream_id=stream_id))

        # probe = probeinterface.read_spikegadgets(file_path, 1)
        from xml.etree import ElementTree
        import numpy as np
        from probeinterface import Probe, ProbeGroup

        # ------------------------- #
        #     Npix 1.0 constants    #
        # ------------------------- #
        TOTAL_NPIX_ELECTRODES = 960
        MAX_ACTIVE_CHANNELS = 384
        CONTACT_WIDTH = 16  # um
        CONTACT_HEIGHT = 20  # um
        # ------------------------- #

        header_txt = probeinterface.parse_spikegadgets_header(file_path)
        root = ElementTree.fromstring(header_txt)
        gconf = sr = root.find("GlobalConfiguration")
        hconf = root.find("HardwareConfiguration")
        sconf = root.find("SpikeConfiguration")

        probe_configs = [device for device in hconf if device.attrib['name'] == 'NeuroPixels1']
        n_probes = len(probe_configs)

        probegroup = ProbeGroup()

        for curr_probe in range(1, n_probes+1):
            probe_config = probe_configs[curr_probe-1]
            active_channel_str = [option for option in probe_config if option.attrib['name']=='channelsOn'][0].attrib['data']
            active_channels = [int(ch) for ch in active_channel_str.split(' ') if ch]
            n_active_channels = sum(active_channels)
            assert len(active_channels) == TOTAL_NPIX_ELECTRODES
            assert n_active_channels <= MAX_ACTIVE_CHANNELS

            """
                <SpikeConfiguration chanPerChip="1889715760" device="neuropixels1" categories="">
                <SpikeNTrode viewLFPBand="0"
                viewStimBand="0"
                id="1384"  # @USE: The first digit is the probe, and the second digit is the electrode number
                lfpScalingToUv="0.018311105685598315"
                LFPChan="1"
                notchFreq="60"
                rawRefOn="0"
                refChan="1"
                viewSpikeBand="1"
                rawScalingToUv="0.018311105685598315"
                spikeScalingToUv="0.018311105685598315"
                refNTrodeID="1"
                notchBW="10"
                color="#c83200"
                refGroup="2"
                filterOn="1"
                LFPHighFilter="200"
                moduleDataOn="0"
                groupRefOn="0"
                lowFilter="600"
                refOn="0"
                notchFilterOn="0"
                lfpRefOn="0"
                lfpFilterOn="0"
                highFilter="6000"
                >
                <SpikeChannel thresh="60"
                coord_dv="-480"  # @USE: dorsal-ventral coordinate in um (in pairs for staggered probe)
                spikeSortingGroup="1782505664"
                triggerOn="1"
                stimCapable="0"
                coord_ml="3192"  # @USE: medial-lateral coordinate in um
                coord_ap="3700"  # doesn't vary, assuming the shank's flat face along ML axis
                maxDisp="400"
                hwChan="735"  # @USE: don't know
                />
                </SpikeNTrode>
                ...
                </SpikeConfiguration>
                """
            # positions = np.stack((x_pos, y_pos), axis=1)
            contact_ids = []
            device_channels = []
            positions = np.zeros((n_active_channels, 2))
            nt_i = 0
            for ntrode in sconf:
                electrode_id = ntrode.attrib['id']
                if int(electrode_id[0]) == curr_probe:
                    contact_ids.append(electrode_id)
                    positions[nt_i,:] = (ntrode[0].attrib['coord_ml'], ntrode[0].attrib['coord_dv'])
                    device_channels.append(ntrode[0].attrib['hwChan'])
                    nt_i += 1
            assert len(contact_ids) == n_active_channels


            # construct Probe object
            probe = Probe(ndim=2, si_units="um", model_name="Neuropixels 1.0", manufacturer="IMEC")
            probe.set_contacts(
                contact_ids=contact_ids,
                positions=positions,
                shapes="square",
                shank_ids=None,
                shape_params={"width": CONTACT_WIDTH, "height": CONTACT_HEIGHT},
            )
            # wire it
            # probe.set_device_channel_indices(np.arange(n_active_channels))  # @TODO: shouldn't this be device_channels?
            probe.set_device_channel_indices(device_channels)
            # polygon_default = [
            #     (0, 10000),
            #     (0, 0),
            #     (35, -175),
            #     (70, 0),
            #     (70, 10000),
            # ]
            # probe.set_planar_contour(np.array(polygon_default))
            probe.move([250 * (curr_probe-1), 0])
            print(f'probe{curr_probe}.get_contact_count()', probe.get_contact_count())
            probegroup.add_probe(probe)

        print('probegroup.get_contact_count()', probegroup.get_contact_count())

        self.set_probes(probegroup, in_place=True)

    @classmethod
    def map_to_neo_kwargs(cls, file_path):
        neo_kwargs = {"filename": str(file_path)}
        return neo_kwargs


read_spikegadgets = define_function_from_class(source_class=SpikeGadgetsRecordingExtractor, name="read_spikegadgets")
